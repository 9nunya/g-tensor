use std::sync::Arc;

use crate::dtype::Dtype;
use crate::error::{Error, Result};

/// One typed, contiguous backing buffer.
#[derive(Debug)]
pub enum StorageData {
    /// `f32` buffer.
    F32(Vec<f32>),
    /// `f64` buffer.
    F64(Vec<f64>),
    /// `i64` buffer (indices).
    I64(Vec<i64>),
}

/// Typed, reference-counted backing storage for [`crate::Tensor`] views.
///
/// Multiple tensors may share one `Storage`; the tensor's offset and strides
/// select the region each view observes.
#[derive(Debug)]
pub struct Storage {
    /// The typed bytes.
    pub data: StorageData,
}

impl Storage {
    /// Element type of the buffer.
    pub fn dtype(&self) -> Dtype {
        match self.data {
            StorageData::F32(_) => Dtype::F32,
            StorageData::F64(_) => Dtype::F64,
            StorageData::I64(_) => Dtype::I64,
        }
    }

    /// Number of elements in the buffer.
    pub fn len(&self) -> usize {
        match &self.data {
            StorageData::F32(v) => v.len(),
            StorageData::F64(v) => v.len(),
            StorageData::I64(v) => v.len(),
        }
    }

    /// Whether the buffer holds zero elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// View the buffer as `f32` (errors on dtype mismatch).
    pub fn f32s(&self) -> Result<&[f32]> {
        match &self.data {
            StorageData::F32(v) => Ok(v),
            _ => Err(Error::dtype("storage", "expected f32")),
        }
    }

    /// View the buffer as `f64` (errors on dtype mismatch).
    pub fn f64s(&self) -> Result<&[f64]> {
        match &self.data {
            StorageData::F64(v) => Ok(v),
            _ => Err(Error::dtype("storage", "expected f64")),
        }
    }

    /// View the buffer as `i64` (errors on dtype mismatch).
    pub fn i64s(&self) -> Result<&[i64]> {
        match &self.data {
            StorageData::I64(v) => Ok(v),
            _ => Err(Error::dtype("storage", "expected i64")),
        }
    }

    /// Mutably view the buffer as `f32` (errors on dtype mismatch).
    pub fn f32s_mut(&mut self) -> Result<&mut [f32]> {
        match &mut self.data {
            StorageData::F32(v) => Ok(v),
            _ => Err(Error::dtype("storage", "expected f32")),
        }
    }
}

/// Shared handle to a [`Storage`].
pub type StorageRef = Arc<Storage>;
