//! Optional Metal GPU cells. v1: FP32 GEMM via MPSGraph when the work is large.
//!
//! This backend works on Apple Silicon **and** on Intel Macs with Metal
//! devices (for example the Intel UHD 630 iGPU and the AMD Radeon Pro 5300M
//! dGPU). It scores the visible devices and prefers a discrete GPU for the
//! single-device path, and it can also split a large GEMM across several Metal
//! devices in parallel.
//!
//! Enabling this crate does not change default placement for small/medium ops.
//!
//! # Device policy
//!
//! When a discrete GPU exists, integrated Intel GPUs are excluded by default
//! because they are slower and MPSGraph on UHD-class iGPUs can be unstable when
//! driven concurrently with the dGPU. Opt them back in with
//! [`set_include_integrated`]. On Intel-only machines they are used
//! automatically. Query the effective set with [`gpu_device_names`].
//!
//! # Multi-device execution
//!
//! [`matmul_multi_device`] partitions batched GEMMs along the batch axis and
//! rank-2 GEMMs along their rows, then runs one MPSGraph executable per device
//! in parallel. Execution on each device is serialized by an internal lock,
//! which is what makes concurrent multi-device runs safe.
//!
//! # CPU + GPU
//!
//! `g-ad`'s matmul backend calls [`should_offload_matmul`] to decide whether
//! to use Metal, and when the shape splits cleanly it runs the CPU and every
//! eligible GPU in the same `std::thread::scope`. See `g_ad::matmul`.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use g_core::{broadcast_shapes, numel, Device, Dtype, Error, Result, Tensor};
use objc2::rc::{autoreleasepool, Retained};
use objc2::runtime::ProtocolObject;
use objc2::AnyThread;
use objc2_foundation::{NSArray, NSDictionary, NSNumber};
use objc2_metal::{MTLBuffer, MTLCommandQueue, MTLCopyAllDevices, MTLDevice, MTLResourceOptions};
use objc2_metal_performance_shaders::MPSDataType;
use objc2_metal_performance_shaders_graph::{
    MPSGraph, MPSGraphCompilationDescriptor, MPSGraphDevice, MPSGraphExecutable,
    MPSGraphExecutableExecutionDescriptor, MPSGraphOptimization, MPSGraphOptions,
    MPSGraphShapedType, MPSGraphTensor, MPSGraphTensorData,
};

/// A compiled GEMM, cached so training loops do not recompile per step.
#[derive(Clone)]
struct CachedGemm {
    exec: Retained<MPSGraphExecutable>,
    a_idx: usize,
    b_idx: usize,
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct GemmKey {
    m: usize,
    n: usize,
    k: usize,
    batch: Vec<usize>,
}

/// One Metal device plus its command queue and compiled-GEMM cache.
struct GpuState {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    name: String,
    gemms: Mutex<HashMap<GemmKey, CachedGemm>>,
    /// Serializes `MPSGraphExecutable` execution on this device. Metal command
    /// queues are thread-safe, but MPSGraph execution is not safe to drive from
    /// several user threads at once on the same device.
    run: Mutex<()>,
}

/// Every Metal device the process can see, best compute device first.
fn states() -> &'static [Arc<GpuState>] {
    static STATES: OnceLock<Vec<Arc<GpuState>>> = OnceLock::new();
    STATES.get_or_init(|| {
        let devices = MTLCopyAllDevices();
        let mut scored: Vec<(i32, usize, String)> = (0..devices.len())
            .map(|i| {
                let d = devices.objectAtIndex(i);
                let name = d.name().to_string();
                let score = device_score(&d, &name);
                (score, i, name)
            })
            .collect();
        // Highest score first. This makes the single-device path use the dGPU
        // on a dual-GPU Intel MacBook.
        scored.sort_by_key(|b| std::cmp::Reverse(b.0));
        scored
            .into_iter()
            .map(|(_, i, name)| {
                let d = devices.objectAtIndex(i);
                let queue = d.newCommandQueue().expect("metal command queue");
                Arc::new(GpuState {
                    device: d,
                    queue,
                    name,
                    gemms: Mutex::new(HashMap::new()),
                    run: Mutex::new(()),
                })
            })
            .collect()
    })
}

/// Score a device so discrete AMD/NVIDIA parts beat integrated Intel parts.
fn device_score(d: &ProtocolObject<dyn MTLDevice>, name: &str) -> i32 {
    let lower = name.to_lowercase();
    let mut score = 0i32;
    if !d.isLowPower() {
        score += 1000;
    }
    if lower.contains("amd") || lower.contains("radeon") {
        score += 500;
    }
    if lower.contains("apple") {
        score += 300;
    }
    if lower.contains("nvidia") || lower.contains("geforce") || lower.contains("quadro") {
        score += 400;
    }
    if lower.contains("intel") {
        score -= 300;
    }
    score
}

fn best_state() -> Option<&'static Arc<GpuState>> {
    eligible_states().into_iter().next()
}

fn is_intel(name: &str) -> bool {
    name.to_lowercase().contains("intel")
}

/// When a discrete GPU is present, Intel iGPUs are excluded from compute by
/// default: on these dual-GPU laptops the iGPU is slower and MPSGraph on the
/// UHD-class iGPU can be unstable when driven concurrently with the dGPU.
static INCLUDE_INTEGRATED: AtomicBool = AtomicBool::new(false);

/// Opt into running large GEMMs on Intel iGPUs as well as the discrete GPU.
///
/// On by default only when no discrete GPU exists (Intel-only machines).
pub fn set_include_integrated(include: bool) {
    INCLUDE_INTEGRATED.store(include, Ordering::Relaxed);
}

/// The devices used for GEMM compute, best first.
fn eligible_states() -> Vec<&'static Arc<GpuState>> {
    let all = states();
    let include = INCLUDE_INTEGRATED.load(Ordering::Relaxed);
    let has_discrete = all.iter().any(|st| !is_intel(&st.name));
    all.iter()
        .filter(|st| include || !(is_intel(&st.name) && has_discrete))
        .collect()
}

/// Whether at least one Metal device is visible.
pub fn gpu_available() -> bool {
    !MTLCopyAllDevices().is_empty()
}

/// Number of Metal devices the GEMM path will use in parallel.
pub fn gpu_device_count() -> usize {
    eligible_states().len()
}

/// Name of the device the single-device path will use for GEMMs.
pub fn gpu_device_name() -> Option<String> {
    best_state().map(|st| st.name.clone())
}

/// Names of the Metal devices used for GEMM compute, best first.
pub fn gpu_device_names() -> Vec<String> {
    eligible_states().iter().map(|st| st.name.clone()).collect()
}

/// Names of every Metal device visible to the process, best first.
pub fn gpu_all_device_names() -> Vec<String> {
    states().iter().map(|st| st.name.clone()).collect()
}

/// Normalized matmul dimensions. Mirrors the CPU `matmul` shape algebra
/// (rank-0 rejection, 1-D promotion, right-aligned batch broadcast, squeeze).
pub struct MatmulDims {
    /// Output rows after 1-D promotion.
    pub m: usize,
    /// Output columns after 1-D promotion.
    pub n: usize,
    /// Contracting (inner) dimension.
    pub k: usize,
    /// Broadcasted batch shape (empty for rank-2).
    pub batch: Vec<usize>,
    /// `a` was a rank-1 vector promoted to `[1, k]`.
    pub left: bool,
    /// `b` was a rank-1 vector promoted to `[k, 1]`.
    pub right: bool,
}

fn promote_left(a: &Tensor) -> Result<(Tensor, bool)> {
    if a.rank() == 1 {
        Ok((a.reshape(&[1, a.shape()[0] as isize])?, true))
    } else {
        Ok((a.clone(), false))
    }
}

fn promote_right(b: &Tensor) -> Result<(Tensor, bool)> {
    if b.rank() == 1 {
        Ok((b.reshape(&[b.shape()[0] as isize, 1])?, true))
    } else {
        Ok((b.clone(), false))
    }
}

fn matmul_dims(a: &Tensor, b: &Tensor) -> Option<MatmulDims> {
    if a.dtype() != Dtype::F32 || b.dtype() != Dtype::F32 {
        return None;
    }
    if a.rank() == 0 || b.rank() == 0 {
        return None;
    }
    let (a_b, left) = promote_left(a).ok()?;
    let (b_b, right) = promote_right(b).ok()?;
    let ar = a_b.rank();
    let br = b_b.rank();
    let m = *a_b.shape().get(ar - 2)?;
    let k = *a_b.shape().get(ar - 1)?;
    if b_b.shape().get(br - 2).copied()? != k {
        return None;
    }
    let n = *b_b.shape().get(br - 1)?;
    if m == 0 || n == 0 || k == 0 {
        return None;
    }
    let batch = broadcast_shapes(&a_b.shape()[..ar - 2], &b_b.shape()[..br - 2]).ok()?;
    if numel(&batch).ok()? == 0 {
        return None;
    }
    Some(MatmulDims {
        m,
        n,
        k,
        batch,
        left,
        right,
    })
}

/// Public shape normalization used by the CPU+GPU splitter in `g-ad`.
pub fn matmul_shape(a: &Tensor, b: &Tensor) -> Option<MatmulDims> {
    matmul_dims(a, b)
}

/// Offload an FP32 GEMM when the GPU wins over the CPU tax.
///
/// The dGPU round-trip (buffer upload + kernel + readback) has a fixed cost,
/// so the per-matrix arithmetic intensity must be nontrivial and the total
/// work must amortize it. Batched `[B,M,K] @ [K,N]` linear layers count the
/// whole batch, which is exactly the shape of most training matmuls.
pub fn should_offload_matmul(a: &Tensor, b: &Tensor) -> bool {
    let Some(d) = matmul_dims(a, b) else {
        return false;
    };
    let single = (d.m as u64)
        .saturating_mul(d.n as u64)
        .saturating_mul(d.k as u64);
    let n_batch = numel(&d.batch).unwrap_or(1) as u64;
    let total = single.saturating_mul(n_batch);
    let max_dim = d.m.max(d.n).max(d.k);
    single >= (1 << 20) && total >= (1 << 28) && max_dim >= 512
}

/// Materialize both operands, broadcast to the full batched shape, and return
/// the full row-major `f32` buffers plus the output shape (pre-squeeze).
fn materialize(a: &Tensor, b: &Tensor, d: &MatmulDims) -> Result<(Vec<f32>, Vec<f32>, Vec<usize>)> {
    let (a_b, _) = promote_left(a)?;
    let (b_b, _) = promote_right(b)?;

    let mut a_shape = d.batch.clone();
    a_shape.push(d.m);
    a_shape.push(d.k);
    let mut b_shape = d.batch.clone();
    b_shape.push(d.k);
    b_shape.push(d.n);

    let a_full = a_b.broadcast_to(&a_shape)?.to_contiguous()?;
    let b_full = b_b.broadcast_to(&b_shape)?.to_contiguous()?;
    let av = a_full.to_vec_f32()?;
    let bv = b_full.to_vec_f32()?;

    let mut out_shape = d.batch.clone();
    out_shape.push(d.m);
    out_shape.push(d.n);
    Ok((av, bv, out_shape))
}

/// Single-device GEMM on the best Metal device.
pub fn matmul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let Some(d) = matmul_dims(a, b) else {
        return Err(Error::unsupported(
            "gpu.matmul",
            a.dtype(),
            Device::Gpu,
            "unsupported shape/dtype",
        ));
    };
    if !should_offload_matmul(a, b) {
        return Err(Error::unsupported(
            "gpu.matmul",
            a.dtype(),
            Device::Gpu,
            "below crossover; use CPU",
        ));
    }
    let Some(st) = best_state() else {
        return Err(Error::unsupported(
            "gpu.matmul",
            Dtype::F32,
            Device::Gpu,
            "no Metal device",
        ));
    };
    let (av, bv, out_shape) = materialize(a, b, &d)?;

    let mut a_dims = d.batch.clone();
    a_dims.push(d.m);
    a_dims.push(d.k);
    let mut b_dims = d.batch.clone();
    b_dims.push(d.k);
    b_dims.push(d.n);
    let mut c_dims = d.batch.clone();
    c_dims.push(d.m);
    c_dims.push(d.n);

    let key = GemmKey {
        m: d.m,
        n: d.n,
        k: d.k,
        batch: d.batch.clone(),
    };
    let cv = autoreleasepool(|_| unsafe {
        gemm_mpsgraph(st, key, &a_dims, &b_dims, &c_dims, &av, &bv)
    })?;
    finish(&d, cv, &out_shape)
}

/// Split a GEMM across every eligible Metal device in parallel.
///
/// Batched work is split along the (flattened) batch axis; a single rank-2
/// GEMM is split along its rows. This is the cheapest correct partition for
/// GEMMs and keeps the per-device compile cache hot.
pub fn matmul_multi_device(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let Some(d) = matmul_dims(a, b) else {
        return Err(Error::unsupported(
            "gpu.matmul_multi",
            a.dtype(),
            Device::Gpu,
            "unsupported shape/dtype",
        ));
    };
    if !should_offload_matmul(a, b) {
        return Err(Error::unsupported(
            "gpu.matmul_multi",
            a.dtype(),
            Device::Gpu,
            "below crossover; use CPU",
        ));
    }
    let devs = eligible_states();
    if devs.is_empty() {
        return Err(Error::unsupported(
            "gpu.matmul_multi",
            Dtype::F32,
            Device::Gpu,
            "no Metal device",
        ));
    }
    let (av, bv, out_shape) = materialize(a, b, &d)?;

    let flat_batch = numel(&d.batch)?;
    let device_count = devs.len();
    let results: Vec<Result<Vec<f32>>> = std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(device_count);

        if !d.batch.is_empty() {
            // Split the flat batch axis across devices. A single batch element
            // (batch numel == 1) still takes this branch so it stays rank-3.
            let parts = device_count.min(flat_batch);
            for (pi, st) in devs.iter().take(parts).enumerate() {
                let start = pi * flat_batch / parts;
                let end = (pi + 1) * flat_batch / parts;
                let blen = end - start;
                let a_range = start * d.m * d.k..end * d.m * d.k;
                let b_range = start * d.k * d.n..end * d.k * d.n;
                let a_dims = vec![blen, d.m, d.k];
                let b_dims = vec![blen, d.k, d.n];
                let c_dims = vec![blen, d.m, d.n];
                let key = GemmKey {
                    m: d.m,
                    n: d.n,
                    k: d.k,
                    batch: vec![blen],
                };
                let av = &av;
                let bv = &bv;
                handles.push(scope.spawn(move || unsafe {
                    gemm_mpsgraph(
                        st,
                        key,
                        &a_dims,
                        &b_dims,
                        &c_dims,
                        &av[a_range],
                        &bv[b_range],
                    )
                }));
            }
        } else {
            // Split the M (row) axis of a single rank-2 GEMM.
            let parts = device_count.min(d.m);
            for (pi, st) in devs.iter().take(parts).enumerate() {
                let start = pi * d.m / parts;
                let end = (pi + 1) * d.m / parts;
                let mlen = end - start;
                let a_range = start * d.k..end * d.k;
                let a_dims = vec![mlen, d.k];
                let b_dims = vec![d.k, d.n];
                let c_dims = vec![mlen, d.n];
                let key = GemmKey {
                    m: mlen,
                    n: d.n,
                    k: d.k,
                    batch: Vec::new(),
                };
                let av = &av;
                let bv = &bv;
                handles.push(scope.spawn(move || unsafe {
                    gemm_mpsgraph(st, key, &a_dims, &b_dims, &c_dims, &av[a_range], bv)
                }));
            }
        }

        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });

    let mut pieces = Vec::with_capacity(results.len());
    for r in results {
        pieces.push(r?);
    }
    let cv = pieces.concat();
    finish(&d, cv, &out_shape)
}

fn finish(d: &MatmulDims, cv: Vec<f32>, out_shape: &[usize]) -> Result<Tensor> {
    let mut shape = out_shape.to_vec();
    if d.right {
        shape.pop();
    }
    if d.left {
        shape.pop();
    }
    if cv.len() != numel(&shape)? {
        return Err(Error::shape(
            "gpu.matmul",
            format!(
                "GPU produced {} elements, expected {}",
                cv.len(),
                numel(&shape)?
            ),
        ));
    }
    Tensor::from_slice_f32(&cv, &shape)
}

unsafe fn compile_gemm(
    device: &ProtocolObject<dyn MTLDevice>,
    a_shape: &NSArray<NSNumber>,
    b_shape: &NSArray<NSNumber>,
) -> Result<CachedGemm> {
    let graph = unsafe { MPSGraph::new() };
    let xa = unsafe {
        graph.placeholderWithShape_dataType_name(Some(a_shape), MPSDataType::Float32, None)
    };
    let xb = unsafe {
        graph.placeholderWithShape_dataType_name(Some(b_shape), MPSDataType::Float32, None)
    };
    let yc =
        unsafe { graph.matrixMultiplicationWithPrimaryTensor_secondaryTensor_name(&xa, &xb, None) };

    let sa = shaped(a_shape);
    let sb = shaped(b_shape);
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

    let feed_order = unsafe { exec.feedTensors() }.expect("feeds");
    let mut a_idx = 0usize;
    let mut b_idx = 1usize;
    for i in 0..feed_order.len() {
        let t = feed_order.objectAtIndex(i);
        if core::ptr::eq(&*t, &*xa) {
            a_idx = i;
        } else if core::ptr::eq(&*t, &*xb) {
            b_idx = i;
        }
    }
    Ok(CachedGemm { exec, a_idx, b_idx })
}

unsafe fn gemm_mpsgraph(
    st: &GpuState,
    key: GemmKey,
    a_dims: &[usize],
    b_dims: &[usize],
    c_dims: &[usize],
    a: &[f32],
    b: &[f32],
) -> Result<Vec<f32>> {
    let device = &st.device;
    let queue = &st.queue;

    let ashape = ns_shape(a_dims);
    let bshape = ns_shape(b_dims);
    let cshape = ns_shape(c_dims);
    let out_n = numel(c_dims)?;

    let cached = {
        let mut cache = st.gemms.lock().unwrap();
        if let Some(c) = cache.get(&key) {
            c.clone()
        } else {
            let c = compile_gemm(device, &ashape, &bshape)?;
            cache.insert(key, c.clone());
            c
        }
    };

    let abuf = buffer(device, a);
    let bbuf = buffer(device, b);
    let mut cout = vec![0.0f32; out_n];
    let cbuf = buffer(device, &cout);
    let ad = tensor_data(&abuf, &ashape);
    let bd = tensor_data(&bbuf, &bshape);
    let cd = tensor_data(&cbuf, &cshape);

    let mut iv: Vec<Retained<MPSGraphTensorData>> = vec![ad.clone(), ad.clone()];
    iv[cached.a_idx] = ad;
    iv[cached.b_idx] = bd;
    let inputs = NSArray::from_retained_slice(&iv);
    let outputs = NSArray::from_retained_slice(std::slice::from_ref(&cd));
    let desc = unsafe { MPSGraphExecutableExecutionDescriptor::new() };
    unsafe { desc.setWaitUntilCompleted(true) };
    let _run = st.run.lock().unwrap();
    let _ = unsafe {
        cached
            .exec
            .runWithMTLCommandQueue_inputsArray_resultsArray_executionDescriptor(
                queue,
                &inputs,
                Some(&outputs),
                Some(&desc),
            )
    };
    let slice =
        unsafe { core::slice::from_raw_parts(cbuf.contents().cast::<f32>().as_ptr(), out_n) };
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
