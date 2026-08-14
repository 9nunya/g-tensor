use std::sync::Arc;

use crate::dtype::Dtype;
use crate::error::{Error, Result};

#[derive(Debug)]
pub enum StorageData {
    F32(Vec<f32>),
    F64(Vec<f64>),
    I64(Vec<i64>),
}

#[derive(Debug)]
pub struct Storage {
    pub data: StorageData,
}

impl Storage {
    pub fn dtype(&self) -> Dtype {
        match self.data {
            StorageData::F32(_) => Dtype::F32,
            StorageData::F64(_) => Dtype::F64,
            StorageData::I64(_) => Dtype::I64,
        }
    }

    pub fn len(&self) -> usize {
        match &self.data {
            StorageData::F32(v) => v.len(),
            StorageData::F64(v) => v.len(),
            StorageData::I64(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn f32s(&self) -> Result<&[f32]> {
        match &self.data {
            StorageData::F32(v) => Ok(v),
            _ => Err(Error::dtype("storage", "expected f32")),
        }
    }

    pub fn f64s(&self) -> Result<&[f64]> {
        match &self.data {
            StorageData::F64(v) => Ok(v),
            _ => Err(Error::dtype("storage", "expected f64")),
        }
    }

    pub fn i64s(&self) -> Result<&[i64]> {
        match &self.data {
            StorageData::I64(v) => Ok(v),
            _ => Err(Error::dtype("storage", "expected i64")),
        }
    }

    pub fn f32s_mut(&mut self) -> Result<&mut [f32]> {
        match &mut self.data {
            StorageData::F32(v) => Ok(v),
            _ => Err(Error::dtype("storage", "expected f32")),
        }
    }
}

pub type StorageRef = Arc<Storage>;
