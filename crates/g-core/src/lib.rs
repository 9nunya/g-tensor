//! Core tensor types, errors, and view/construction ops.

mod backward;
mod construct;
mod device;
mod dtype;
mod error;
mod shape;
mod storage;
mod tensor;

pub use backward::Backward;
pub use construct::{arange_f32, eye, full_f32, linspace_f32, randn_f32};
pub use device::Device;
pub use dtype::Dtype;
pub use error::{Error, ErrorKind, Result};
pub use shape::{
    broadcast_shapes, broadcast_strides, for_each_index, normalize_axis, numel, offset_of,
    resolve_reshape, row_major_strides,
};
pub use storage::{Storage, StorageData};
pub use tensor::{Leaf, Tensor};

/// Zeros of `dtype` on CPU.
pub fn zeros(shape: &[usize], dtype: Dtype) -> Result<Tensor> {
    Tensor::zeros(shape, dtype)
}

/// Ones of `dtype` on CPU.
pub fn ones(shape: &[usize], dtype: Dtype) -> Result<Tensor> {
    Tensor::ones(shape, dtype)
}

/// Copy a host `f32` slice into a CPU tensor of `shape`.
pub fn from_slice_f32(data: &[f32], shape: &[usize]) -> Result<Tensor> {
    Tensor::from_slice_f32(data, shape)
}

/// Copy a host `f64` slice into a CPU tensor of `shape`.
pub fn from_slice_f64(data: &[f64], shape: &[usize]) -> Result<Tensor> {
    Tensor::from_slice_f64(data, shape)
}

/// Copy a host `i64` slice into a CPU tensor of `shape`.
pub fn from_slice_i64(data: &[i64], shape: &[usize]) -> Result<Tensor> {
    Tensor::from_slice_i64(data, shape)
}
