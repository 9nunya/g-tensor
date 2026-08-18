use g_core::{for_each_index, normalize_axis, numel, Dtype, Error, Result, Tensor};

/// Concatenate tensors along `axis`.
///
/// All tensors must share a dtype, rank, and every dimension except `axis`.
pub fn cat(tensors: &[&Tensor], axis: isize) -> Result<Tensor> {
    if tensors.is_empty() {
        return Err(Error::shape("cat", "need at least one tensor"));
    }
    let dtype = tensors[0].dtype();
    let rank = tensors[0].rank();
    let ax = normalize_axis(axis, rank, "cat")?;
    for t in tensors {
        if t.dtype() != dtype {
            return Err(Error::dtype("cat", "mixed dtypes"));
        }
        if t.rank() != rank {
            return Err(Error::shape("cat", "mixed ranks"));
        }
        for (i, (&d, &e)) in t.shape().iter().zip(tensors[0].shape()).enumerate() {
            if i != ax && d != e {
                return Err(Error::shape("cat", "non-cat dims must match"));
            }
        }
    }
    let mut out_shape = tensors[0].shape().to_vec();
    out_shape[ax] = tensors.iter().map(|t| t.shape()[ax]).sum();
    match dtype {
        Dtype::F32 => crate::fast::cat_f32(tensors, ax, &out_shape),
        Dtype::F64 => {
            let mut out = vec![0.0f64; numel(&out_shape)?];
            let mut cursor = 0usize;
            for t in tensors {
                let take = t.shape()[ax];
                for_each_index(t.shape(), |idx| {
                    let mut oidx = idx.to_vec();
                    oidx[ax] += cursor;
                    let mut off = 0usize;
                    let mut st = 1usize;
                    for i in (0..out_shape.len()).rev() {
                        off += oidx[i] * st;
                        st *= out_shape[i];
                    }
                    out[off] = t.read_f64_at(idx).unwrap();
                });
                cursor += take;
            }
            Tensor::from_slice_f64(&out, &out_shape)
        }
        Dtype::I64 => Err(Error::dtype("cat", "v1 float cat")),
    }
}

/// Stack tensors along a new `axis` (rank increases by one).
pub fn stack(tensors: &[&Tensor], axis: isize) -> Result<Tensor> {
    let unsqueezed: Result<Vec<Tensor>> = tensors.iter().map(|t| t.unsqueeze(axis)).collect();
    let u = unsqueezed?;
    let refs: Vec<&Tensor> = u.iter().collect();
    cat(&refs, axis)
}

/// Reduce `x` by max over `axis` (`keepdims` retains it as size 1).
pub fn amax(x: &Tensor, axis: isize, keepdims: bool) -> Result<Tensor> {
    let ax = normalize_axis(axis, x.rank(), "amax")?;
    if x.shape()[ax] == 0 {
        return Err(Error::shape("amax", "empty reduction has no identity"));
    }
    let mut out_shape = x.shape().to_vec();
    if keepdims {
        out_shape[ax] = 1;
    } else {
        out_shape.remove(ax);
    }
    match x.dtype() {
        Dtype::F32 => crate::fast::amax_f32(x, ax, &out_shape),
        Dtype::F64 => {
            let mut acc = vec![f64::NEG_INFINITY; numel(&out_shape)?.max(1)];
            for_each_index(x.shape(), |idx| {
                let mut oidx = Vec::new();
                for (i, &ix) in idx.iter().enumerate() {
                    if i == ax {
                        if keepdims {
                            oidx.push(0);
                        }
                    } else {
                        oidx.push(ix);
                    }
                }
                let mut off = 0usize;
                let mut st = 1usize;
                if !out_shape.is_empty() {
                    for i in (0..out_shape.len()).rev() {
                        off += oidx[i] * st;
                        st *= out_shape[i];
                    }
                }
                let v = x.read_f64_at(idx).unwrap();
                if v > acc[off] {
                    acc[off] = v;
                }
            });
            Tensor::from_slice_f64(&acc, &out_shape)
        }
        Dtype::I64 => Err(Error::dtype("amax", "float")),
    }
}
