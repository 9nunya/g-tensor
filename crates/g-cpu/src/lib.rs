//! CPU kernels (oracle). Optional Accelerate GEMM via feature `accelerate`.

use g_core::{
    broadcast_shapes, for_each_index, normalize_axis, numel, Dtype, Error, ErrorKind, Result,
    Tensor,
};

mod fast;
mod shape_ops;
mod unary;

pub use shape_ops::{amax, cat, stack};
pub use unary::{
    abs, clamp, div, exp, gelu, leaky_relu, log, pow_scalar, sigmoid, sign, silu, softplus, sqrt,
};

#[cfg(feature = "accelerate")]
mod accelerate {
    #[link(name = "Accelerate", kind = "framework")]
    unsafe extern "C" {
        pub fn cblas_sgemm(
            order: i32,
            transa: i32,
            transb: i32,
            m: i32,
            n: i32,
            k: i32,
            alpha: f32,
            a: *const f32,
            lda: i32,
            b: *const f32,
            ldb: i32,
            beta: f32,
            c: *mut f32,
            ldc: i32,
        );
        pub fn cblas_dgemm(
            order: i32,
            transa: i32,
            transb: i32,
            m: i32,
            n: i32,
            k: i32,
            alpha: f64,
            a: *const f64,
            lda: i32,
            b: *const f64,
            ldb: i32,
            beta: f64,
            c: *mut f64,
            ldc: i32,
        );
    }
    pub const ROW: i32 = 101;
    pub const NOTRANS: i32 = 111;
    pub const TRANS: i32 = 112;
}

/// Whether this build links Accelerate (AMX/BLAS + vForce).
pub fn accelerate_enabled() -> bool {
    cfg!(feature = "accelerate")
}

/// Number of worker threads the kernels will use.
pub fn thread_count() -> usize {
    rayon::current_num_threads()
}

fn same_device_dtype(op: &'static str, xs: &[&Tensor]) -> Result<(Dtype, g_core::Device)> {
    let d0 = xs[0].dtype();
    let dev = xs[0].device();
    for x in xs {
        if x.dtype() != d0 {
            return Err(Error::dtype(op, "mixed dtypes"));
        }
        if x.device() != dev {
            return Err(Error::device(op, "mixed devices"));
        }
    }
    Ok((d0, dev))
}

pub(crate) fn binary_float<F32, F64>(
    op: &'static str,
    a: &Tensor,
    b: &Tensor,
    f32e: F32,
    f64e: F64,
) -> Result<Tensor>
where
    F32: Fn(f32, f32) -> f32,
    F64: Fn(f64, f64) -> f64,
{
    let (dtype, _) = same_device_dtype(op, &[a, b])?;
    if !dtype.is_float() {
        return Err(Error::dtype(op, "expected floating tensors"));
    }
    let shape = broadcast_shapes(a.shape(), b.shape())?;
    let a_b = a.broadcast_to(&shape)?;
    let b_b = b.broadcast_to(&shape)?;
    match dtype {
        Dtype::F32 => {
            let mut out = vec![0.0f32; numel(&shape)?];
            let mut i = 0;
            for_each_index(&shape, |idx| {
                out[i] = f32e(a_b.read_f32_at(idx).unwrap(), b_b.read_f32_at(idx).unwrap());
                i += 1;
            });
            Tensor::from_slice_f32(&out, &shape)
        }
        Dtype::F64 => {
            let mut out = vec![0.0f64; numel(&shape)?];
            let mut i = 0;
            for_each_index(&shape, |idx| {
                out[i] = f64e(a_b.read_f64_at(idx).unwrap(), b_b.read_f64_at(idx).unwrap());
                i += 1;
            });
            Tensor::from_slice_f64(&out, &shape)
        }
        Dtype::I64 => unreachable!(),
    }
}

pub fn add(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    if a.dtype() == Dtype::F32 && b.dtype() == Dtype::F32 {
        return fast::binary_f32("add", a, b, |x, y| x + y);
    }
    binary_float("add", a, b, |x, y| x + y, |x, y| x + y)
}

pub fn sub(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    if a.dtype() == Dtype::F32 && b.dtype() == Dtype::F32 {
        return fast::binary_f32("sub", a, b, |x, y| x - y);
    }
    binary_float("sub", a, b, |x, y| x - y, |x, y| x - y)
}

pub fn mul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    if a.dtype() == Dtype::F32 && b.dtype() == Dtype::F32 {
        return fast::binary_f32("mul", a, b, |x, y| x * y);
    }
    binary_float("mul", a, b, |x, y| x * y, |x, y| x * y)
}

pub fn mul_scalar(a: &Tensor, s: f64) -> Result<Tensor> {
    match a.dtype() {
        Dtype::F32 => {
            let sf = s as f32;
            fast::map_f32(a, move |x| x * sf)
        }
        Dtype::F64 => {
            let v: Vec<f64> = a.to_vec_f64()?.into_iter().map(|x| x * s).collect();
            Tensor::from_slice_f64(&v, a.shape())
        }
        Dtype::I64 => Err(Error::dtype("mul_scalar", "expected float")),
    }
}

pub fn neg(a: &Tensor) -> Result<Tensor> {
    mul_scalar(a, -1.0)
}

pub fn relu(a: &Tensor) -> Result<Tensor> {
    match a.dtype() {
        Dtype::F32 => fast::map_f32(a, |x| x.max(0.0)),
        Dtype::F64 => {
            let v: Vec<f64> = a.to_vec_f64()?.into_iter().map(|x| x.max(0.0)).collect();
            Tensor::from_slice_f64(&v, a.shape())
        }
        Dtype::I64 => Err(Error::dtype("relu", "expected float")),
    }
}

pub fn tanh(a: &Tensor) -> Result<Tensor> {
    match a.dtype() {
        Dtype::F32 => fast::unary_f32(a, fast::k_tanh),
        Dtype::F64 => {
            let v: Vec<f64> = a.to_vec_f64()?.into_iter().map(|x| x.tanh()).collect();
            Tensor::from_slice_f64(&v, a.shape())
        }
        Dtype::I64 => Err(Error::dtype("tanh", "expected float")),
    }
}

pub fn square(a: &Tensor) -> Result<Tensor> {
    mul(a, a)
}

/// Softmax over the last axis, fused. `f32` only.
pub fn softmax_last(x: &Tensor) -> Result<Tensor> {
    fast::softmax_last_f32(x)
}

/// Log-softmax over the last axis, fused. `f32` only.
pub fn log_softmax_last(x: &Tensor) -> Result<Tensor> {
    fast::log_softmax_last_f32(x)
}

/// Apply a scalar `f32` function elementwise (parallel, contiguous fast path).
pub fn map_f32(x: &Tensor, f: impl Fn(f32) -> f32 + Sync) -> Result<Tensor> {
    if x.dtype() != Dtype::F32 {
        return Err(Error::dtype("map_f32", "f32 only"));
    }
    fast::map_f32(x, f)
}

/// GELU value and local derivative in one pass. `f32` only.
pub fn gelu_with_grad(x: &Tensor) -> Result<(Tensor, Tensor)> {
    if x.dtype() != Dtype::F32 {
        return Err(Error::dtype("gelu_with_grad", "f32 only"));
    }
    fast::gelu_fwd_bwd(x)
}

fn reduce_axes(shape: &[usize], axes: &[usize], keepdims: bool) -> Result<Vec<usize>> {
    let mut drop = vec![false; shape.len()];
    for &a in axes {
        if a >= shape.len() {
            return Err(Error::shape("reduce", "axis oob"));
        }
        if drop[a] {
            return Err(Error::shape("reduce", "duplicate axis"));
        }
        drop[a] = true;
    }
    let mut out = Vec::new();
    for (i, &d) in shape.iter().enumerate() {
        if drop[i] {
            if keepdims {
                out.push(1);
            }
        } else {
            out.push(d);
        }
    }
    Ok(out)
}

pub fn sum(x: &Tensor, axes: Option<&[isize]>, keepdims: bool) -> Result<Tensor> {
    if !x.dtype().is_float() {
        return Err(Error::dtype("sum", "v1 sum is float-only"));
    }
    let axes_u: Vec<usize> = match axes {
        None => (0..x.rank()).collect(),
        Some(ax) => {
            let mut v = Vec::new();
            for &a in ax {
                v.push(normalize_axis(a, x.rank(), "sum")?);
            }
            v
        }
    };
    let out_shape = reduce_axes(x.shape(), &axes_u, keepdims)?;
    let reduced: Vec<bool> = {
        let mut r = vec![false; x.rank()];
        for &a in &axes_u {
            r[a] = true;
        }
        r
    };
    match x.dtype() {
        Dtype::F32 => fast::sum_f32(x, &reduced, &out_shape),
        Dtype::F64 => {
            let mut acc = vec![0.0f64; numel(&out_shape)?];
            for_each_index(x.shape(), |idx| {
                let mut oidx = Vec::new();
                for (i, &ix) in idx.iter().enumerate() {
                    if reduced[i] {
                        if keepdims {
                            oidx.push(0);
                        }
                    } else {
                        oidx.push(ix);
                    }
                }
                let o = if out_shape.is_empty() {
                    0
                } else {
                    let mut off = 0usize;
                    let mut st = 1usize;
                    for i in (0..out_shape.len()).rev() {
                        off += oidx[i] * st;
                        st *= out_shape[i];
                    }
                    off
                };
                acc[o] += x.read_f64_at(idx).unwrap();
            });
            Tensor::from_slice_f64(&acc, &out_shape)
        }
        Dtype::I64 => unreachable!(),
    }
}

pub fn mean(x: &Tensor, axes: Option<&[isize]>, keepdims: bool) -> Result<Tensor> {
    if !x.dtype().is_float() {
        return Err(Error::dtype("mean", "float only"));
    }
    let axes_u: Vec<usize> = match axes {
        None => (0..x.rank()).collect(),
        Some(ax) => ax
            .iter()
            .map(|&a| normalize_axis(a, x.rank(), "mean"))
            .collect::<Result<Vec<_>>>()?,
    };
    let mut n = 1usize;
    for &a in &axes_u {
        n = n.saturating_mul(x.shape()[a]);
    }
    let s = sum(x, axes, keepdims)?;
    if n == 0 {
        // empty mean → NaN
        match x.dtype() {
            Dtype::F32 => {
                let v = vec![f32::NAN; s.numel()];
                Tensor::from_slice_f32(&v, s.shape())
            }
            Dtype::F64 => {
                let v = vec![f64::NAN; s.numel()];
                Tensor::from_slice_f64(&v, s.shape())
            }
            Dtype::I64 => unreachable!(),
        }
    } else {
        mul_scalar(&s, 1.0 / n as f64)
    }
}

fn gemm_f32_into(m: usize, n: usize, k: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    #[cfg(feature = "accelerate")]
    unsafe {
        accelerate::cblas_sgemm(
            accelerate::ROW,
            accelerate::NOTRANS,
            accelerate::NOTRANS,
            m as i32,
            n as i32,
            k as i32,
            1.0,
            a.as_ptr(),
            k as i32,
            b.as_ptr(),
            n as i32,
            0.0,
            c.as_mut_ptr(),
            n as i32,
        );
    }
    #[cfg(not(feature = "accelerate"))]
    {
        for v in c.iter_mut() {
            *v = 0.0;
        }
        for i in 0..m {
            for p in 0..k {
                let av = a[i * k + p];
                for j in 0..n {
                    c[i * n + j] += av * b[p * n + j];
                }
            }
        }
    }
}

fn gemm_f64(m: usize, n: usize, k: usize, a: &[f64], b: &[f64]) -> Vec<f64> {
    let mut c = vec![0.0f64; m * n];
    #[cfg(feature = "accelerate")]
    unsafe {
        accelerate::cblas_dgemm(
            accelerate::ROW,
            accelerate::NOTRANS,
            accelerate::NOTRANS,
            m as i32,
            n as i32,
            k as i32,
            1.0,
            a.as_ptr(),
            k as i32,
            b.as_ptr(),
            n as i32,
            0.0,
            c.as_mut_ptr(),
            n as i32,
        );
        return c;
    }
    #[cfg(not(feature = "accelerate"))]
    {
        for i in 0..m {
            for p in 0..k {
                let av = a[i * k + p];
                for j in 0..n {
                    c[i * n + j] += av * b[p * n + j];
                }
            }
        }
        c
    }
}


/// Rank-2 `f32` GEMM that consumes transposed *views* directly.
///
/// Matmul backward computes `aᵀ @ gy` and `gy @ bᵀ`. Materializing those
/// transposes costs a full cache-hostile copy each; BLAS can apply them for
/// free via its transpose flags, so a transposed view is passed straight
/// through. Returns `None` when the layout is not a plain row/column-major
/// rank-2 matrix.
#[cfg(feature = "accelerate")]
fn matmul2d_blas(a: &Tensor, b: &Tensor) -> Option<Result<Tensor>> {
    // (is_transposed, leading_dim) for a 2-D view, or None if oddly strided.
    fn layout(t: &Tensor) -> Option<(bool, usize)> {
        let (sh, st) = (t.shape(), t.strides());
        let (m, k) = (sh[0], sh[1]);
        if st[1] == 1 && st[0] == k as isize {
            Some((false, k.max(1)))
        } else if st[0] == 1 && st[1] == m as isize {
            Some((true, m.max(1)))
        } else {
            None
        }
    }
    if a.rank() != 2 || b.rank() != 2 || a.dtype() != Dtype::F32 || b.dtype() != Dtype::F32 {
        return None;
    }
    let (ta, lda) = layout(a)?;
    let (tb, ldb) = layout(b)?;
    let (m, k, n) = (a.shape()[0], a.shape()[1], b.shape()[1]);
    if b.shape()[0] != k {
        return None;
    }
    let av = match a.storage_f32() {
        Ok(v) => v,
        Err(e) => return Some(Err(e)),
    };
    let bv = match b.storage_f32() {
        Ok(v) => v,
        Err(e) => return Some(Err(e)),
    };
    let mut c = vec![0f32; m * n];
    unsafe {
        accelerate::cblas_sgemm(
            accelerate::ROW,
            if ta { accelerate::TRANS } else { accelerate::NOTRANS },
            if tb { accelerate::TRANS } else { accelerate::NOTRANS },
            m as i32,
            n as i32,
            k as i32,
            1.0,
            av.as_ptr().add(a.storage_offset()),
            lda as i32,
            bv.as_ptr().add(b.storage_offset()),
            ldb as i32,
            0.0,
            c.as_mut_ptr(),
            n as i32,
        );
    }
    Some(Tensor::from_vec_f32(c, &[m, n]))
}

/// Shape algebra matches the charter/record: reject rank-0; 1-D promotions; batch broadcast.
pub fn matmul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    #[cfg(feature = "accelerate")]
    if let Some(r) = matmul2d_blas(a, b) {
        return r;
    }
    let (dtype, _) = same_device_dtype("matmul", &[a, b])?;
    if !dtype.is_float() {
        return Err(Error::dtype("matmul", "float only"));
    }
    if a.rank() == 0 || b.rank() == 0 {
        return Err(Error::shape("matmul", "rank-0 is not allowed"));
    }
    // Normalize to batched matrices.
    let (a_b, a_squeeze) = promote_left(a)?;
    let (b_b, b_squeeze) = promote_right(b)?;
    // a_b: [..., m, k]  b_b: [..., k, n]
    let a_rank = a_b.rank();
    let b_rank = b_b.rank();
    let k_a = a_b.shape()[a_rank - 1];
    let k_b = b_b.shape()[b_rank - 2];
    if k_a != k_b {
        return Err(Error::new(
            ErrorKind::BackendPrecheck,
            "matmul",
            format!("contracting dims differ {k_a} & {k_b}"),
        ));
    }
    let m = a_b.shape()[a_rank - 2];
    let n = b_b.shape()[b_rank - 1];
    let k = k_a;
    let a_batch = &a_b.shape()[..a_rank - 2];
    let b_batch = &b_b.shape()[..b_rank - 2];
    let batch = broadcast_shapes(a_batch, b_batch)?;
    let mut out_shape = batch.clone();
    out_shape.push(m);
    out_shape.push(n);
    let n_batch = numel(&batch)?;
    let a_mat = a_b.to_contiguous()?;
    let b_mat = b_b.to_contiguous()?;
    match dtype {
        Dtype::F32 => {
            let av = a_mat.as_slice_f32()?;
            let bv = b_mat.as_slice_f32()?;
            let a_stride = m * k;
            let b_stride = k * n;
            let mut cv = vec![0.0f32; n_batch * m * n];
            for bi in 0..n_batch {
                let a_i = batch_index_to_src(bi, &batch, a_batch);
                let b_i = batch_index_to_src(bi, &batch, b_batch);
                let off = bi * m * n;
                gemm_f32_into(
                    m,
                    n,
                    k,
                    &av[a_i * a_stride..a_i * a_stride + a_stride],
                    &bv[b_i * b_stride..b_i * b_stride + b_stride],
                    &mut cv[off..off + m * n],
                );
            }
            let mut out = Tensor::from_vec_f32(cv, &out_shape)?;
            squeeze_matmul(&mut out, a_squeeze, b_squeeze);
            Ok(out)
        }
        Dtype::F64 => {
            let av = a_mat.to_vec_f64()?;
            let bv = b_mat.to_vec_f64()?;
            let a_stride = m * k;
            let b_stride = k * n;
            let mut cv = vec![0.0f64; n_batch * m * n];
            for bi in 0..n_batch {
                let a_i = batch_index_to_src(bi, &batch, a_batch);
                let b_i = batch_index_to_src(bi, &batch, b_batch);
                let tile = gemm_f64(
                    m,
                    n,
                    k,
                    &av[a_i * a_stride..a_i * a_stride + a_stride],
                    &bv[b_i * b_stride..b_i * b_stride + b_stride],
                );
                let off = bi * m * n;
                cv[off..off + m * n].copy_from_slice(&tile);
            }
            let mut out = Tensor::from_slice_f64(&cv, &out_shape)?;
            squeeze_matmul(&mut out, a_squeeze, b_squeeze);
            Ok(out)
        }
        Dtype::I64 => unreachable!(),
    }
}

fn batch_index_to_src(flat: usize, out_batch: &[usize], src_batch: &[usize]) -> usize {
    if src_batch.is_empty() {
        return 0;
    }
    // unravel flat in out_batch, then ravel in src with broadcast (dim 1 → index 0)
    let mut rem = flat;
    let mut coords = vec![0usize; out_batch.len()];
    for i in (0..out_batch.len()).rev() {
        coords[i] = rem % out_batch[i].max(1);
        rem /= out_batch[i].max(1);
    }
    let pad = out_batch.len() - src_batch.len();
    let mut off = 0usize;
    let mut st = 1usize;
    for i in (0..src_batch.len()).rev() {
        let c = coords[i + pad];
        let idx = if src_batch[i] == 1 { 0 } else { c };
        off += idx * st;
        st *= src_batch[i].max(1);
    }
    off
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

fn squeeze_matmul(out: &mut Tensor, left: bool, right: bool) {
    let mut shape = out.shape().to_vec();
    if right && !shape.is_empty() {
        shape.pop();
    }
    if left && !shape.is_empty() {
        let last = shape.len() - 1;
        // left inserted m=1 as the second-to-last of the pre-squeeze?
        // out was [batch..., m, n]; left squeeze removes m (second last before right pop).
        // After right pop: [batch..., m]. Remove last if left.
        if left {
            shape.pop();
        } else {
            let _ = last;
        }
    }
    if let Ok(t) = out.reshape(&shape.iter().map(|&d| d as isize).collect::<Vec<_>>()) {
        *out = t;
    }
}

pub fn gather(x: &Tensor, axis: isize, index: &Tensor) -> Result<Tensor> {
    if index.dtype() != Dtype::I64 {
        return Err(Error::dtype("gather", "index must be i64"));
    }
    let ax = normalize_axis(axis, x.rank(), "gather")?;
    let dim = x.shape()[ax] as i64;
    // Broadcast index to x on non-axis dims: we take index.shape as output shape (record).
    let out_shape = index.shape().to_vec();
    if index.rank() != x.rank() {
        return Err(Error::shape("gather", "index rank must equal input rank"));
    }
    for i in 0..x.rank() {
        if i == ax {
            continue;
        }
        if index.shape()[i] != x.shape()[i] && index.shape()[i] != 1 && x.shape()[i] != 1 {
            return Err(Error::shape("gather", "index not broadcast-compatible"));
        }
    }
    match x.dtype() {
        Dtype::F32 => {
            let mut out = vec![0.0f32; numel(&out_shape)?];
            let mut i = 0;
            for_each_index(&out_shape, |oidx| {
                let mut src = oidx.to_vec();
                let gi = index.read_i64_at(oidx).unwrap();
                let mut ii = gi;
                if ii < 0 {
                    ii += dim;
                }
                if ii < 0 || ii >= dim {
                    // mark; we'll error after if needed — do it now via panic-free sentinel
                    src[ax] = usize::MAX;
                } else {
                    src[ax] = ii as usize;
                    // broadcast x on non-axis if needed
                    for (t, &s) in src.iter_mut().zip(x.shape()) {
                        if s == 1 {
                            *t = 0;
                        }
                    }
                }
                if src[ax] == usize::MAX {
                    out[i] = f32::NAN;
                } else {
                    out[i] = x.read_f32_at(&src).unwrap();
                }
                i += 1;
            });
            if out.iter().any(|v| v.is_nan()) {
                // distinguish OOB from data NaN: recheck
                let mut bad = false;
                for_each_index(&out_shape, |oidx| {
                    let gi = index.read_i64_at(oidx).unwrap();
                    let mut ii = gi;
                    if ii < 0 {
                        ii += dim;
                    }
                    if ii < 0 || ii >= dim {
                        bad = true;
                    }
                });
                if bad {
                    return Err(Error::index("gather", "index out of bounds"));
                }
            }
            Tensor::from_slice_f32(&out, &out_shape)
        }
        Dtype::F64 => {
            let mut out = vec![0.0f64; numel(&out_shape)?];
            let mut i = 0;
            for_each_index(&out_shape, |oidx| {
                let mut src = oidx.to_vec();
                let gi = index.read_i64_at(oidx).unwrap();
                let mut ii = gi;
                if ii < 0 {
                    ii += dim;
                }
                if ii < 0 || ii >= dim {
                    src[ax] = usize::MAX;
                } else {
                    src[ax] = ii as usize;
                    for (t, &s) in src.iter_mut().zip(x.shape()) {
                        if s == 1 {
                            *t = 0;
                        }
                    }
                }
                if src[ax] != usize::MAX {
                    out[i] = x.read_f64_at(&src).unwrap();
                }
                i += 1;
            });
            Tensor::from_slice_f64(&out, &out_shape)
        }
        Dtype::I64 => Err(Error::dtype("gather", "v1 gather float data")),
    }
}

pub fn scatter_add(dst: &Tensor, axis: isize, index: &Tensor, src: &Tensor) -> Result<Tensor> {
    if index.dtype() != Dtype::I64 {
        return Err(Error::dtype("scatter_add", "index must be i64"));
    }
    if dst.dtype() != src.dtype() {
        return Err(Error::dtype("scatter_add", "dst/src dtype"));
    }
    let ax = normalize_axis(axis, dst.rank(), "scatter_add")?;
    let dim = dst.shape()[ax] as i64;
    if index.shape() != src.shape() {
        return Err(Error::shape(
            "scatter_add",
            "index and src shapes must match",
        ));
    }
    let out = dst.copy()?;
    // We'll write into a packed buffer then rebuild.
    match dst.dtype() {
        Dtype::F32 => {
            let mut buf = out.to_vec_f32()?;
            let shape = out.shape().to_vec();
            for_each_index(src.shape(), |sidx| {
                let gi = index.read_i64_at(sidx).unwrap();
                let mut ii = gi;
                if ii < 0 {
                    ii += dim;
                }
                if ii < 0 || ii >= dim {
                    return;
                }
                let mut didx = sidx.to_vec();
                if didx.len() != shape.len() {
                    return;
                }
                didx[ax] = ii as usize;
                let mut off = 0usize;
                let mut st = 1usize;
                for i in (0..shape.len()).rev() {
                    off += didx[i] * st;
                    st *= shape[i];
                }
                buf[off] += src.read_f32_at(sidx).unwrap();
            });
            // OOB check
            let mut bad = false;
            for_each_index(src.shape(), |sidx| {
                let gi = index.read_i64_at(sidx).unwrap();
                let mut ii = gi;
                if ii < 0 {
                    ii += dim;
                }
                if ii < 0 || ii >= dim {
                    bad = true;
                }
            });
            if bad {
                return Err(Error::index("scatter_add", "index out of bounds"));
            }
            Tensor::from_slice_f32(&buf, &shape)
        }
        Dtype::F64 => {
            let mut buf = out.to_vec_f64()?;
            let shape = out.shape().to_vec();
            for_each_index(src.shape(), |sidx| {
                let gi = index.read_i64_at(sidx).unwrap();
                let mut ii = gi;
                if ii < 0 {
                    ii += dim;
                }
                if ii < 0 || ii >= dim {
                    return;
                }
                let mut didx = sidx.to_vec();
                didx[ax] = ii as usize;
                let mut off = 0usize;
                let mut st = 1usize;
                for i in (0..shape.len()).rev() {
                    off += didx[i] * st;
                    st *= shape[i];
                }
                buf[off] += src.read_f64_at(sidx).unwrap();
            });
            Tensor::from_slice_f64(&buf, &shape)
        }
        Dtype::I64 => Err(Error::dtype("scatter_add", "float dst")),
    }
}

pub fn relu_inplace(x: &mut Tensor) -> Result<()> {
    x.require_unique("relu_inplace")?;
    // Unique storage: rebuild via copy of values into same tensor is hard without mut storage.
    // Contract: unique untracked. We replace *x with relu(x).
    *x = relu(x)?;
    Ok(())
}

/// Gather `index` (1-D i64 of length batch) along `axis` of `x`.
pub fn take(x: &Tensor, axis: isize, index: &Tensor) -> Result<Tensor> {
    if index.rank() != 1 {
        return Err(Error::shape("take", "index must be rank 1"));
    }
    if index.dtype() != Dtype::I64 {
        return Err(Error::dtype("take", "i64 index"));
    }
    let ax = normalize_axis(axis, x.rank(), "take")?;
    if x.rank() != 2 || ax != 1 {
        return Err(Error::shape("take", "v1 take supports rank-2 axis=1 only"));
    }
    let b = x.shape()[0];
    let c = x.shape()[1] as i64;
    if index.numel() != b {
        return Err(Error::shape("take", "index length must equal batch"));
    }
    match x.dtype() {
        Dtype::F32 => {
            let mut out = vec![0.0f32; b];
            for (i, slot) in out.iter_mut().enumerate() {
                let mut j = index.read_i64_at(&[i])?;
                if j < 0 {
                    j += c;
                }
                if j < 0 || j >= c {
                    return Err(Error::index("take", "oob"));
                }
                *slot = x.read_f32_at(&[i, j as usize])?;
            }
            Tensor::from_slice_f32(&out, &[b])
        }
        Dtype::F64 => {
            let mut out = vec![0.0f64; b];
            for (i, slot) in out.iter_mut().enumerate() {
                let mut j = index.read_i64_at(&[i])?;
                if j < 0 {
                    j += c;
                }
                if j < 0 || j >= c {
                    return Err(Error::index("take", "oob"));
                }
                *slot = x.read_f64_at(&[i, j as usize])?;
            }
            Tensor::from_slice_f64(&out, &[b])
        }
        Dtype::I64 => Err(Error::dtype("take", "float data")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_broadcast() {
        let a = Tensor::from_slice_f32(&[1.0, 2.0], &[2, 1]).unwrap();
        let b = Tensor::from_slice_f32(&[10.0, 20.0, 30.0], &[1, 3]).unwrap();
        let c = add(&a, &b).unwrap();
        assert_eq!(c.shape(), &[2, 3]);
        assert_eq!(
            c.to_vec_f32().unwrap(),
            vec![11.0, 21.0, 31.0, 12.0, 22.0, 32.0]
        );
    }

    #[test]
    fn matmul_inner_mismatch() {
        let a = Tensor::from_slice_f32(&[1.0, 2.0, 3.0], &[1, 3]).unwrap();
        let b = Tensor::from_slice_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let e = matmul(&a, &b).unwrap_err();
        assert_eq!(e.kind, ErrorKind::BackendPrecheck);
    }

    #[test]
    fn empty_sum_is_zero() {
        let x = Tensor::zeros(&[2, 0, 3], Dtype::F32).unwrap();
        let s = sum(&x, Some(&[1]), false).unwrap();
        assert_eq!(s.shape(), &[2, 3]);
        assert!(s.to_vec_f32().unwrap().iter().all(|&v| v == 0.0));
    }

    #[test]
    fn empty_mean_is_nan() {
        let x = Tensor::zeros(&[2, 0], Dtype::F32).unwrap();
        let m = mean(&x, Some(&[1]), false).unwrap();
        assert!(m.to_vec_f32().unwrap().iter().all(|v| v.is_nan()));
    }

    #[test]
    fn matmul_with_transpose() {
        let q = Tensor::from_slice_f32(&[1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0], &[2, 4]).unwrap();
        let k = Tensor::from_slice_f32(&[1.0; 8], &[2, 4]).unwrap();
        let kt = k.transpose().unwrap();
        assert_eq!(kt.shape(), &[4, 2], "kt {:?}", kt.shape());
        let s = matmul(&q, &kt).unwrap();
        assert_eq!(s.shape(), &[2, 2], "s {:?}", s.shape());
    }

    #[test]
    fn vec_dot() {
        let a = Tensor::from_slice_f32(&[1.0, 2.0, 3.0], &[3]).unwrap();
        let b = Tensor::from_slice_f32(&[4.0, 5.0, 6.0], &[3]).unwrap();
        let c = matmul(&a, &b).unwrap();
        assert_eq!(c.shape(), &[] as &[usize]);
        assert_eq!(c.item_f32().unwrap(), 32.0);
    }
}
