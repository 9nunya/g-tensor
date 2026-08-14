use std::fmt;
use std::sync::{Arc, Mutex};

use crate::backward::Backward;
use crate::device::Device;
use crate::dtype::Dtype;
use crate::error::{Error, ErrorKind, Result};
use crate::shape::{
    broadcast_shapes, broadcast_strides, for_each_index, is_contiguous, normalize_axis, numel,
    offset_of, resolve_reshape, row_major_strides,
};
use crate::storage::{Storage, StorageData, StorageRef};

#[derive(Debug)]
pub struct Leaf {
    pub grad: Mutex<Option<Tensor>>,
}

/// Runtime-rank tensor. `Clone` is a cheap handle (refcount), not a byte copy.
///
/// Use [`Tensor::copy`] for a deep copy and [`Tensor::with_requires_grad`] to
/// mark a leaf for reverse AD.
#[derive(Clone)]
pub struct Tensor {
    pub(crate) storage: StorageRef,
    pub(crate) offset: usize,
    pub(crate) shape: Vec<usize>,
    pub(crate) strides: Vec<isize>,
    pub(crate) dtype: Dtype,
    pub(crate) device: Device,
    pub(crate) requires_grad: bool,
    pub(crate) leaf: Option<Arc<Leaf>>,
    pub(crate) grad_fn: Option<Arc<dyn Backward>>,
    pub(crate) placement: &'static str,
}

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Tensor")
            .field("shape", &self.shape)
            .field("dtype", &self.dtype)
            .field("device", &self.device)
            .field("requires_grad", &self.requires_grad)
            .field("placement", &self.placement)
            .finish()
    }
}

impl Tensor {
    #[doc(hidden)]
    pub fn from_storage(
        storage: Storage,
        shape: Vec<usize>,
        dtype: Dtype,
        device: Device,
    ) -> Result<Self> {
        let n = numel(&shape)?;
        if storage.len() != n {
            return Err(Error::shape(
                "from_storage",
                format!("storage len {} != numel {n}", storage.len()),
            ));
        }
        if storage.dtype() != dtype {
            return Err(Error::dtype("from_storage", "storage/dtype mismatch"));
        }
        if device != Device::Cpu {
            return Err(Error::unsupported(
                "from_storage",
                dtype,
                device,
                "v1 default build is CPU-only; enable feature gpu for the stub",
            ));
        }
        Ok(Self {
            storage: Arc::new(storage),
            offset: 0,
            strides: row_major_strides(&shape),
            shape,
            dtype,
            device,
            requires_grad: false,
            leaf: None,
            grad_fn: None,
            placement: "cpu",
        })
    }

    /// Copy `data` (row-major) into a new CPU `f32` tensor.
    pub fn from_slice_f32(data: &[f32], shape: &[usize]) -> Result<Self> {
        let n = numel(shape)?;
        if data.len() != n {
            return Err(Error::shape(
                "from_slice",
                format!("len {} != numel {n}", data.len()),
            ));
        }
        Self::from_storage(
            Storage {
                data: StorageData::F32(data.to_vec()),
            },
            shape.to_vec(),
            Dtype::F32,
            Device::Cpu,
        )
    }

    /// Copy `data` into a new CPU `f64` tensor.
    pub fn from_slice_f64(data: &[f64], shape: &[usize]) -> Result<Self> {
        let n = numel(shape)?;
        if data.len() != n {
            return Err(Error::shape(
                "from_slice",
                format!("len {} != numel {n}", data.len()),
            ));
        }
        Self::from_storage(
            Storage {
                data: StorageData::F64(data.to_vec()),
            },
            shape.to_vec(),
            Dtype::F64,
            Device::Cpu,
        )
    }

    /// Copy `data` into a new CPU `i64` tensor.
    pub fn from_slice_i64(data: &[i64], shape: &[usize]) -> Result<Self> {
        let n = numel(shape)?;
        if data.len() != n {
            return Err(Error::shape(
                "from_slice",
                format!("len {} != numel {n}", data.len()),
            ));
        }
        Self::from_storage(
            Storage {
                data: StorageData::I64(data.to_vec()),
            },
            shape.to_vec(),
            Dtype::I64,
            Device::Cpu,
        )
    }

    /// New CPU tensor filled with zeros.
    pub fn zeros(shape: &[usize], dtype: Dtype) -> Result<Self> {
        let n = numel(shape)?;
        let data = match dtype {
            Dtype::F32 => StorageData::F32(vec![0.0; n]),
            Dtype::F64 => StorageData::F64(vec![0.0; n]),
            Dtype::I64 => StorageData::I64(vec![0; n]),
        };
        Self::from_storage(Storage { data }, shape.to_vec(), dtype, Device::Cpu)
    }

    /// New CPU tensor filled with ones.
    pub fn ones(shape: &[usize], dtype: Dtype) -> Result<Self> {
        let n = numel(shape)?;
        let data = match dtype {
            Dtype::F32 => StorageData::F32(vec![1.0; n]),
            Dtype::F64 => StorageData::F64(vec![1.0; n]),
            Dtype::I64 => StorageData::I64(vec![1; n]),
        };
        Self::from_storage(Storage { data }, shape.to_vec(), dtype, Device::Cpu)
    }

    /// New CPU `f32` tensor filled with `value`.
    pub fn full_f32(shape: &[usize], value: f32) -> Result<Self> {
        let n = numel(shape)?;
        Self::from_storage(
            Storage {
                data: StorageData::F32(vec![value; n]),
            },
            shape.to_vec(),
            Dtype::F32,
            Device::Cpu,
        )
    }

    /// Rank-0 `f32` tensor.
    pub fn scalar_f32(value: f32) -> Result<Self> {
        Self::from_slice_f32(&[value], &[])
    }

    /// Rank-0 `f64` tensor.
    pub fn scalar_f64(value: f64) -> Result<Self> {
        Self::from_slice_f64(&[value], &[])
    }

    /// Dimension sizes. Rank-0 is `[]`.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// `shape().len()`.
    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    /// Product of dimensions (0 if any dim is 0).
    pub fn numel(&self) -> usize {
        numel(&self.shape).unwrap_or(0)
    }

    /// Element strides (may be zero or negative for views).
    pub fn strides(&self) -> &[isize] {
        &self.strides
    }

    /// Element type.
    pub fn dtype(&self) -> Dtype {
        self.dtype
    }

    /// Storage device.
    pub fn device(&self) -> Device {
        self.device
    }

    /// Where the last kernel ran (`"cpu"` / `"gpu"`).
    pub fn placement_trace(&self) -> &'static str {
        self.placement
    }

    /// Row-major contiguous with offset 0.
    pub fn is_contiguous(&self) -> bool {
        is_contiguous(&self.shape, &self.strides, self.offset)
    }

    /// Whether this value is in the reverse graph.
    pub fn requires_grad(&self) -> bool {
        self.requires_grad
    }

    /// Unique untracked storage — required for in-place ops.
    pub fn is_unique(&self) -> bool {
        Arc::strong_count(&self.storage) == 1 && self.grad_fn.is_none() && !self.requires_grad
    }

    #[doc(hidden)]
    pub fn require_unique(&self, op: &'static str) -> Result<()> {
        if !self.is_unique() {
            return Err(Error::new(
                ErrorKind::NotUnique,
                op,
                "in-place requires a unique untracked tensor",
            ));
        }
        Ok(())
    }

    /// Mark this handle as an AD leaf (floats only).
    pub fn with_requires_grad(mut self) -> Self {
        if self.dtype.is_float() {
            self.requires_grad = true;
            if self.leaf.is_none() {
                self.leaf = Some(Arc::new(Leaf {
                    grad: Mutex::new(None),
                }));
            }
        }
        self
    }

    #[doc(hidden)]
    pub fn set_grad_fn(&mut self, gf: Arc<dyn Backward>) {
        self.requires_grad = true;
        self.grad_fn = Some(gf);
        // Intermediates must not share the leaf accumulator of a view clone.
        self.leaf = None;
    }

    #[doc(hidden)]
    pub fn grad_fn(&self) -> Option<&Arc<dyn Backward>> {
        self.grad_fn.as_ref()
    }

    #[doc(hidden)]
    pub fn leaf(&self) -> Option<&Arc<Leaf>> {
        self.leaf.as_ref()
    }

    /// Same storage, dropped from the graph.
    pub fn detach(&self) -> Self {
        let mut t = self.clone();
        t.requires_grad = false;
        t.leaf = None;
        t.grad_fn = None;
        t
    }

    /// Deep copy of the values; result is untracked.
    pub fn copy(&self) -> Result<Self> {
        let packed = self.to_contiguous()?;
        let mut out = packed;
        out.requires_grad = false;
        out.leaf = None;
        out.grad_fn = None;
        Ok(out)
    }

    /// Move/copy to `device`. GPU is unsupported unless the `gpu` feature is on.
    pub fn to(&self, device: Device) -> Result<Self> {
        if device == self.device {
            return Ok(self.clone());
        }
        if device == Device::Gpu {
            return Err(Error::unsupported(
                "to",
                self.dtype,
                device,
                "gpu feature off by default; CPU-first v1",
            ));
        }
        Ok(self.clone())
    }

    /// Extract the single element. Errors if `numel != 1`. Named CPU sync.
    pub fn item_f32(&self) -> Result<f32> {
        if self.numel() != 1 {
            return Err(Error::shape("item", "numel must be 1"));
        }
        match self.dtype {
            Dtype::F32 => Ok(self.read_f32_at(&vec![0; self.rank()])?),
            Dtype::F64 => Ok(self.read_f64_at(&vec![0; self.rank()])? as f32),
            Dtype::I64 => Ok(self.read_i64_at(&vec![0; self.rank()])? as f32),
        }
    }

    /// Pack values as row-major `f32`. Named CPU sync.
    pub fn to_vec_f32(&self) -> Result<Vec<f32>> {
        if self.dtype != Dtype::F32 {
            return Err(Error::dtype("to_vec", "expected f32"));
        }
        let mut out = Vec::with_capacity(self.numel());
        for_each_index(&self.shape, |idx| {
            out.push(self.read_f32_at(idx).unwrap());
        });
        Ok(out)
    }

    /// Pack values as row-major `f64`. Named CPU sync.
    pub fn to_vec_f64(&self) -> Result<Vec<f64>> {
        if self.dtype != Dtype::F64 {
            return Err(Error::dtype("to_vec", "expected f64"));
        }
        let mut out = Vec::with_capacity(self.numel());
        for_each_index(&self.shape, |idx| {
            out.push(self.read_f64_at(idx).unwrap());
        });
        Ok(out)
    }

    /// Pack values as row-major `i64`. Named CPU sync.
    pub fn to_vec_i64(&self) -> Result<Vec<i64>> {
        if self.dtype != Dtype::I64 {
            return Err(Error::dtype("to_vec", "expected i64"));
        }
        let mut out = Vec::with_capacity(self.numel());
        for_each_index(&self.shape, |idx| {
            out.push(self.read_i64_at(idx).unwrap());
        });
        Ok(out)
    }

    #[doc(hidden)]
    pub fn read_f32_at(&self, index: &[usize]) -> Result<f32> {
        let off = offset_of(index, self.offset, &self.strides);
        Ok(self.storage.f32s()?[off])
    }

    #[doc(hidden)]
    pub fn read_f64_at(&self, index: &[usize]) -> Result<f64> {
        let off = offset_of(index, self.offset, &self.strides);
        Ok(self.storage.f64s()?[off])
    }

    #[doc(hidden)]
    pub fn read_i64_at(&self, index: &[usize]) -> Result<i64> {
        let off = offset_of(index, self.offset, &self.strides);
        Ok(self.storage.i64s()?[off])
    }

    /// Materialize a contiguous copy if needed.
    pub fn to_contiguous(&self) -> Result<Self> {
        if self.is_contiguous() {
            return Ok(self.clone());
        }
        match self.dtype {
            Dtype::F32 => Self::from_slice_f32(&self.to_vec_f32()?, &self.shape),
            Dtype::F64 => Self::from_slice_f64(&self.to_vec_f64()?, &self.shape),
            Dtype::I64 => Self::from_slice_i64(&self.to_vec_i64()?, &self.shape),
        }
    }

    /// View-only reshape; errors if not contiguous.
    pub fn view(&self, new_shape: &[isize]) -> Result<Self> {
        if !self.is_contiguous() {
            return Err(Error::shape("view", "view requires a contiguous tensor"));
        }
        let shape = resolve_reshape(&self.shape, new_shape, "view")?;
        let mut t = self.clone();
        t.shape = shape;
        t.strides = row_major_strides(&t.shape);
        Ok(t)
    }

    /// Reshape, copying if a view is impossible. One `-1` allowed.
    pub fn reshape(&self, new_shape: &[isize]) -> Result<Self> {
        if self.is_contiguous() {
            self.view(new_shape)
        } else {
            self.to_contiguous()?.view(new_shape)
        }
    }

    /// Permute axes (view).
    pub fn permute(&self, axes: &[isize]) -> Result<Self> {
        if axes.len() != self.rank() {
            return Err(Error::shape("permute", "axes rank mismatch"));
        }
        let mut seen = vec![false; self.rank()];
        let mut new_shape = Vec::with_capacity(self.rank());
        let mut new_strides = Vec::with_capacity(self.rank());
        for &a in axes {
            let ax = normalize_axis(a, self.rank(), "permute")?;
            if seen[ax] {
                return Err(Error::shape("permute", "duplicate axis"));
            }
            seen[ax] = true;
            new_shape.push(self.shape[ax]);
            new_strides.push(self.strides[ax]);
        }
        let mut t = self.clone();
        t.shape = new_shape;
        t.strides = new_strides;
        Ok(t)
    }

    /// Swap the last two axes (view; for AD use `g::transpose`).
    pub fn transpose(&self) -> Result<Self> {
        if self.rank() < 2 {
            return Err(Error::shape("transpose", "rank must be >= 2"));
        }
        let mut axes: Vec<isize> = (0..self.rank() as isize).collect();
        let r = axes.len();
        axes.swap(r - 1, r - 2);
        self.permute(&axes)
    }

    /// Basic slice. `ranges` is (start, end, step) per axis; None means full axis.
    /// Basic slice. Each triple is `(start, end, step)`; `None` means default. Negative step allowed.
    pub fn slice(&self, ranges: &[(Option<isize>, Option<isize>, Option<isize>)]) -> Result<Self> {
        if ranges.len() > self.rank() {
            return Err(Error::shape("slice", "too many ranges"));
        }
        let mut shape = Vec::new();
        let mut strides = Vec::new();
        let mut offset = self.offset as isize;
        for ax in 0..self.rank() {
            let dim = self.shape[ax] as isize;
            let (start, end, step) = if ax < ranges.len() {
                ranges[ax]
            } else {
                (None, None, None)
            };
            let step = step.unwrap_or(1);
            if step == 0 {
                return Err(Error::index("slice", "step must not be 0"));
            }
            let def_start = if step > 0 { 0 } else { dim - 1 };
            let def_end = if step > 0 { dim } else { -1 };
            let mut s = start.unwrap_or(def_start);
            let mut e = end.unwrap_or(def_end);
            if s < 0 {
                s += dim;
            }
            if e < 0 && end.is_some() {
                e += dim;
            }
            if step > 0 {
                s = s.clamp(0, dim);
                e = e.clamp(0, dim);
                if e < s {
                    e = s;
                }
                let len = if s >= e { 0 } else { (e - s + step - 1) / step };
                offset += s * self.strides[ax];
                shape.push(len as usize);
                strides.push(self.strides[ax] * step);
            } else {
                s = s.clamp(-1, dim - 1);
                e = e.clamp(-1, dim - 1);
                let len = if s <= e {
                    0
                } else {
                    (s - e + (-step) - 1) / (-step)
                };
                offset += s * self.strides[ax];
                shape.push(len as usize);
                strides.push(self.strides[ax] * step);
            }
        }
        if offset < 0 {
            return Err(Error::index("slice", "negative storage offset"));
        }
        let mut t = self.clone();
        t.offset = offset as usize;
        t.shape = shape;
        t.strides = strides;
        Ok(t)
    }

    /// Integer index on one axis (drops the axis).
    /// Integer index on one axis (drops that axis). OOB errors.
    pub fn select(&self, axis: isize, index: isize) -> Result<Self> {
        let ax = normalize_axis(axis, self.rank(), "select")?;
        let dim = self.shape[ax] as isize;
        let mut i = index;
        if i < 0 {
            i += dim;
        }
        if i < 0 || i >= dim {
            return Err(Error::index(
                "select",
                format!("index {index} out of {dim}"),
            ));
        }
        let mut ranges: Vec<(Option<isize>, Option<isize>, Option<isize>)> =
            vec![(None, None, None); self.rank()];
        ranges[ax] = (Some(i), Some(i + 1), Some(1));
        let sl = self.slice(&ranges)?;
        let mut shape = Vec::new();
        let mut strides = Vec::new();
        for (k, &d) in sl.shape.iter().enumerate() {
            if k != ax {
                shape.push(d);
                strides.push(sl.strides[k]);
            }
        }
        let mut t = sl;
        t.shape = shape;
        t.strides = strides;
        Ok(t)
    }

    /// Expand to `target` with zero strides on size-1 axes.
    pub fn broadcast_to(&self, target: &[usize]) -> Result<Self> {
        let _ = broadcast_shapes(&self.shape, target)?;
        // If any expanded dim is not 1 in self, error already. Writing later checks overlap.
        let strides = broadcast_strides(&self.shape, &self.strides, target)?;
        let mut t = self.clone();
        t.shape = target.to_vec();
        t.strides = strides;
        Ok(t)
    }

    #[doc(hidden)]
    pub fn has_zero_stride(&self) -> bool {
        self.strides
            .iter()
            .zip(self.shape.iter())
            .any(|(&s, &d)| d > 1 && s == 0)
    }

    #[doc(hidden)]
    pub fn same_storage(&self, other: &Tensor) -> bool {
        Arc::ptr_eq(&self.storage, &other.storage)
    }

    /// Insert a size-1 axis.
    pub fn unsqueeze(&self, axis: isize) -> Result<Self> {
        let rank = self.rank() as isize;
        let a = if axis < 0 { axis + rank + 1 } else { axis };
        if a < 0 || a > rank {
            return Err(Error::shape(
                "unsqueeze",
                format!("axis {axis} out of 0..={rank}"),
            ));
        }
        let ax = a as usize;
        let mut shape = self.shape.clone();
        let mut strides = self.strides.clone();
        shape.insert(ax, 1);
        strides.insert(ax, 0);
        let mut t = self.clone();
        t.shape = shape;
        t.strides = strides;
        Ok(t)
    }

    /// Remove size-1 axes (`None` = all of them).
    pub fn squeeze(&self, axis: Option<isize>) -> Result<Self> {
        let mut shape = Vec::new();
        let mut strides = Vec::new();
        if let Some(ax) = axis {
            let a = crate::shape::normalize_axis(ax, self.rank(), "squeeze")?;
            if self.shape[a] != 1 {
                return Err(Error::shape("squeeze", "axis size is not 1"));
            }
            for (i, (&d, &s)) in self.shape.iter().zip(self.strides.iter()).enumerate() {
                if i != a {
                    shape.push(d);
                    strides.push(s);
                }
            }
        } else {
            for (&d, &s) in self.shape.iter().zip(self.strides.iter()) {
                if d != 1 {
                    shape.push(d);
                    strides.push(s);
                }
            }
        }
        let mut t = self.clone();
        t.shape = shape;
        t.strides = strides;
        Ok(t)
    }

    /// Reshape to rank-1.
    pub fn flatten(&self) -> Result<Self> {
        self.reshape(&[-1])
    }

    /// Convert dtype (new storage unless identical).
    pub fn cast(&self, dtype: Dtype) -> Result<Self> {
        if self.dtype == dtype {
            return Ok(self.clone());
        }
        match (self.dtype, dtype) {
            (Dtype::F32, Dtype::F64) => {
                let v: Vec<f64> = self.to_vec_f32()?.into_iter().map(f64::from).collect();
                Tensor::from_slice_f64(&v, &self.shape)
            }
            (Dtype::F64, Dtype::F32) => {
                let v: Vec<f32> = self.to_vec_f64()?.into_iter().map(|x| x as f32).collect();
                Tensor::from_slice_f32(&v, &self.shape)
            }
            (Dtype::F32, Dtype::I64) => {
                let v: Vec<i64> = self.to_vec_f32()?.into_iter().map(|x| x as i64).collect();
                Tensor::from_slice_i64(&v, &self.shape)
            }
            (Dtype::I64, Dtype::F32) => {
                let v: Vec<f32> = self.to_vec_i64()?.into_iter().map(|x| x as f32).collect();
                Tensor::from_slice_f32(&v, &self.shape)
            }
            (Dtype::F64, Dtype::I64) => {
                let v: Vec<i64> = self.to_vec_f64()?.into_iter().map(|x| x as i64).collect();
                Tensor::from_slice_i64(&v, &self.shape)
            }
            (Dtype::I64, Dtype::F64) => {
                let v: Vec<f64> = self.to_vec_i64()?.into_iter().map(|x| x as f64).collect();
                Tensor::from_slice_f64(&v, &self.shape)
            }
            _ => Ok(self.clone()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_slice_and_item() {
        let t = Tensor::from_slice_f32(&[3.0], &[]).unwrap();
        assert_eq!(t.item_f32().unwrap(), 3.0);
        assert_eq!(t.rank(), 0);
    }

    #[test]
    fn unique_after_clone_is_false() {
        let t = Tensor::from_slice_f32(&[1.0, 2.0], &[2]).unwrap();
        let _u = t.clone();
        assert!(!t.is_unique());
    }

    #[test]
    fn reshape_empty() {
        let t = Tensor::zeros(&[0, 3], Dtype::F32).unwrap();
        let r = t.reshape(&[0, -1]).unwrap();
        assert_eq!(r.shape(), &[0, 3]);
    }
}
