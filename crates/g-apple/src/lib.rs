//! Optional Apple GPU cells. v1: FP32 GEMM via MPSGraph when the work is large.
//! Enabling this crate does not change default placement for small/medium ops.

use std::cell::RefCell;

use g_core::{Device, Dtype, Error, Result, Tensor};
use objc2::rc::{autoreleasepool, Retained};
use objc2::runtime::ProtocolObject;
use objc2::AnyThread;
use objc2_foundation::{NSArray, NSDictionary, NSNumber};
use objc2_metal::{MTLBuffer, MTLCommandQueue, MTLCopyAllDevices, MTLDevice, MTLResourceOptions};
use objc2_metal_performance_shaders::MPSDataType;
use objc2_metal_performance_shaders_graph::{
    MPSGraph, MPSGraphCompilationDescriptor, MPSGraphDevice, MPSGraphExecutableExecutionDescriptor,
    MPSGraphOptimization, MPSGraphOptions, MPSGraphShapedType, MPSGraphTensor, MPSGraphTensorData,
};

struct GpuState {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
}

thread_local! {
    static STATE: RefCell<Option<GpuState>> = const { RefCell::new(None) };
}

fn with_state<T>(f: impl FnOnce(&GpuState) -> Result<T>) -> Result<T> {
    STATE.with(|cell| {
        if cell.borrow().is_none() {
            let devices = MTLCopyAllDevices();
            if devices.is_empty() {
                return Err(Error::unsupported(
                    "gpu",
                    Dtype::F32,
                    Device::Gpu,
                    "no Metal device",
                ));
            }
            let device = devices.objectAtIndex(0);
            let queue = device.newCommandQueue().ok_or_else(|| {
                Error::unsupported("gpu", Dtype::F32, Device::Gpu, "no command queue")
            })?;
            *cell.borrow_mut() = Some(GpuState { device, queue });
        }
        let g = cell.borrow();
        f(g.as_ref().expect("gpu state"))
    })
}

pub fn gpu_available() -> bool {
    !MTLCopyAllDevices().is_empty()
}

/// Offload 2-D FP32 GEMM when arithmetic intensity beats the CPU tax.
pub fn should_offload_matmul(a: &Tensor, b: &Tensor) -> bool {
    if a.dtype() != Dtype::F32 || b.dtype() != Dtype::F32 {
        return false;
    }
    if a.rank() != 2 || b.rank() != 2 {
        return false;
    }
    let m = a.shape()[0];
    let k = a.shape()[1];
    let n = b.shape()[1];
    if b.shape()[0] != k {
        return false;
    }
    // Train/PC sizes stay CPU. Wide/4096-class goes GPU.
    m.max(n).max(k) >= 1024 && (m as u64) * (n as u64) * (k as u64) >= 256 * 256 * 256
}

pub fn matmul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    if !should_offload_matmul(a, b) {
        return Err(Error::unsupported(
            "gpu.matmul",
            a.dtype(),
            Device::Gpu,
            "below crossover; use CPU",
        ));
    }
    let m = a.shape()[0];
    let k = a.shape()[1];
    let n = b.shape()[1];
    let av = a.to_contiguous()?.to_vec_f32()?;
    let bv = b.to_contiguous()?.to_vec_f32()?;
    let cv = autoreleasepool(|_| with_state(|st| unsafe { gemm_mpsgraph(st, m, n, k, &av, &bv) }))?;
    Tensor::from_slice_f32(&cv, &[m, n])
}

unsafe fn gemm_mpsgraph(
    st: &GpuState,
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
) -> Result<Vec<f32>> {
    let device = &st.device;
    let queue = &st.queue;

    let ashape = ns_shape(&[m, k]);
    let bshape = ns_shape(&[k, n]);
    let cshape = ns_shape(&[m, n]);
    let graph = unsafe { MPSGraph::new() };
    let xa = unsafe {
        graph.placeholderWithShape_dataType_name(Some(&ashape), MPSDataType::Float32, None)
    };
    let xb = unsafe {
        graph.placeholderWithShape_dataType_name(Some(&bshape), MPSDataType::Float32, None)
    };
    let yc =
        unsafe { graph.matrixMultiplicationWithPrimaryTensor_secondaryTensor_name(&xa, &xb, None) };

    let sa = shaped(&ashape);
    let sb = shaped(&bshape);
    let feeds: Retained<NSDictionary<MPSGraphTensor, MPSGraphShapedType>> =
        NSDictionary::from_retained_objects(&[&*xa, &*xb], &[sa, sb]);
    let targets = NSArray::from_slice(&[&*yc]);
    let gdev = unsafe { MPSGraphDevice::deviceWithMTLDevice(device) };
    let cdesc = unsafe { MPSGraphCompilationDescriptor::new() };
    unsafe {
        cdesc.setOptimizationLevel(MPSGraphOptimization::Level0);
        cdesc.setWaitForCompilationCompletion(true);
    }
    let exec = unsafe {
        graph.compileWithDevice_feeds_targetTensors_targetOperations_compilationDescriptor(
            Some(&gdev),
            &feeds,
            &targets,
            None,
            Some(&cdesc),
        )
    };
    unsafe { exec.setOptions(MPSGraphOptions::SynchronizeResults) };

    let abuf = buffer(device, a);
    let bbuf = buffer(device, b);
    let mut cout = vec![0.0f32; m * n];
    let cbuf = buffer(device, &cout);
    let ad = tensor_data(&abuf, &ashape);
    let bd = tensor_data(&bbuf, &bshape);
    let cd = tensor_data(&cbuf, &cshape);

    let feed_order = unsafe { exec.feedTensors() }.expect("feeds");
    let mut iv = Vec::new();
    for i in 0..feed_order.len() {
        let t = feed_order.objectAtIndex(i);
        if core::ptr::eq(&*t, &*xa) {
            iv.push(ad.clone());
        } else {
            iv.push(bd.clone());
        }
    }
    let inputs = NSArray::from_retained_slice(&iv);
    let outputs = NSArray::from_retained_slice(std::slice::from_ref(&cd));
    let desc = unsafe { MPSGraphExecutableExecutionDescriptor::new() };
    unsafe { desc.setWaitUntilCompleted(true) };
    let _ = unsafe {
        exec.runWithMTLCommandQueue_inputsArray_resultsArray_executionDescriptor(
            queue,
            &inputs,
            Some(&outputs),
            Some(&desc),
        )
    };
    let slice =
        unsafe { core::slice::from_raw_parts(cbuf.contents().cast::<f32>().as_ptr(), m * n) };
    cout.copy_from_slice(slice);
    Ok(cout)
}

fn ns_shape(dims: &[usize]) -> Retained<NSArray<NSNumber>> {
    let nums: Vec<Retained<NSNumber>> = dims.iter().copied().map(NSNumber::new_usize).collect();
    NSArray::from_retained_slice(&nums)
}

unsafe fn shaped(shape: &NSArray<NSNumber>) -> Retained<MPSGraphShapedType> {
    unsafe {
        MPSGraphShapedType::initWithShape_dataType(
            MPSGraphShapedType::alloc(),
            Some(shape),
            MPSDataType::Float32,
        )
    }
}

fn buffer(
    device: &ProtocolObject<dyn MTLDevice>,
    data: &[f32],
) -> Retained<ProtocolObject<dyn MTLBuffer>> {
    let bytes = std::mem::size_of_val(data);
    let buf = device
        .newBufferWithLength_options(bytes, MTLResourceOptions::StorageModeShared)
        .expect("mtlbuffer");
    unsafe {
        core::ptr::copy_nonoverlapping(
            data.as_ptr(),
            buf.contents().cast::<f32>().as_ptr(),
            data.len(),
        );
    }
    buf
}

unsafe fn tensor_data(
    buffer: &ProtocolObject<dyn MTLBuffer>,
    shape: &NSArray<NSNumber>,
) -> Retained<MPSGraphTensorData> {
    unsafe {
        MPSGraphTensorData::initWithMTLBuffer_shape_dataType(
            MPSGraphTensorData::alloc(),
            buffer,
            shape,
            MPSDataType::Float32,
        )
    }
}
