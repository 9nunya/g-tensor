//! Fast CPU kernels.
//!
//! The baseline kernels in this crate were correct but pathologically slow:
//! every elementwise op round-tripped through `to_vec`/`from_slice` (two full
//! copies) and every element paid `offset_of` stride math plus a `Result`
//! unwrap. Reductions allocated a `Vec` *per element*.
//!
//! Four ideas carry the speed here:
//!   1. zero-copy contiguous fast paths ([`Tensor::as_slice_f32`]),
//!   2. slice-level inner loops the compiler can auto-vectorize,
//!   3. work splitting across cores above a size threshold,
//!   4. Accelerate vForce for transcendentals (exp/log/tanh/sqrt).

use g_core::{broadcast_shapes, broadcast_strides, numel, Error, Result, Tensor};
use rayon::prelude::*;

/// Below this element count, threading costs more than it saves.
const PAR_MIN: usize = 1 << 15;

#[cfg(feature = "accelerate")]
mod vforce {
    // Accelerate's vForce vector math. `y` and `x` may alias when equal.
    extern "C" {
        pub fn vvexpf(y: *mut f32, x: *const f32, n: *const i32);
        pub fn vvlogf(y: *mut f32, x: *const f32, n: *const i32);
        pub fn vvtanhf(y: *mut f32, x: *const f32, n: *const i32);
        pub fn vvsqrtf(y: *mut f32, x: *const f32, n: *const i32);
    }
}

/// Run `f(start_index, chunk)` over `out`, in parallel above [`PAR_MIN`].
#[inline]
pub(crate) fn par_chunks<T: Send>(out: &mut [T], f: impl Fn(usize, &mut [T]) + Sync) {
    let n = out.len();
    if n < PAR_MIN {
        f(0, out);
        return;
    }
    let threads = rayon::current_num_threads().max(1);
    let mut chunk = n.div_ceil(threads).max(1024);
    chunk = chunk.div_ceil(16) * 16; // keep SIMD lanes aligned
    out.par_chunks_mut(chunk)
        .enumerate()
        .for_each(|(i, c)| f(i * chunk, c));
}

// ---------------------------------------------------------------- chunk maths

macro_rules! vforce_kernel {
    ($name:ident, $acc:ident, $scalar:expr) => {
        #[inline]
        pub(crate) fn $name(src: &[f32], dst: &mut [f32]) {
            #[cfg(feature = "accelerate")]
            {
                let n = src.len() as i32;
                unsafe { vforce::$acc(dst.as_mut_ptr(), src.as_ptr(), &n) };
            }
            #[cfg(not(feature = "accelerate"))]
            {
                let f: fn(f32) -> f32 = $scalar;
                for (d, &s) in dst.iter_mut().zip(src) {
                    *d = f(s);
                }
            }
        }
    };
}

vforce_kernel!(k_exp, vvexpf, |x| x.exp());
vforce_kernel!(k_log, vvlogf, |x| x.ln());
vforce_kernel!(k_tanh, vvtanhf, |x| x.tanh());
vforce_kernel!(k_sqrt, vvsqrtf, |x| x.sqrt());

/// GELU, tanh approximation. Matches the scalar reference to <1e-6.
#[inline]
pub(crate) fn k_gelu(src: &[f32], dst: &mut [f32]) {
    const C: f32 = 0.797_884_6; // sqrt(2/pi)
    for (d, &s) in dst.iter_mut().zip(src) {
        *d = C * (s + 0.044715 * s * s * s);
    }
    let tmp = dst.as_ptr() as *const f32;
    k_tanh(unsafe { std::slice::from_raw_parts(tmp, dst.len()) }, dst);
    for (d, &s) in dst.iter_mut().zip(src) {
        *d = 0.5 * s * (1.0 + *d);
    }
}

#[inline]
pub(crate) fn k_sigmoid(src: &[f32], dst: &mut [f32]) {
    for (d, &s) in dst.iter_mut().zip(src) {
        *d = -s;
    }
    let tmp = dst.as_ptr() as *const f32;
    k_exp(unsafe { std::slice::from_raw_parts(tmp, dst.len()) }, dst);
    for d in dst.iter_mut() {
        *d = 1.0 / (1.0 + *d);
    }
}

#[inline]
pub(crate) fn k_silu(src: &[f32], dst: &mut [f32]) {
    k_sigmoid(src, dst);
    for (d, &s) in dst.iter_mut().zip(src) {
        *d *= s;
    }
}

// ------------------------------------------------------------------- unary

/// Apply a chunk-level kernel `f(src, dst)` over all of `x`.
pub(crate) fn unary_f32(x: &Tensor, f: impl Fn(&[f32], &mut [f32]) + Sync) -> Result<Tensor> {
    let n = x.numel();
    let mut out = vec![0f32; n];
    match x.as_slice_f32() {
        Ok(src) => par_chunks(&mut out, |off, dst| f(&src[off..off + dst.len()], dst)),
        Err(_) => {
            let src = x.to_vec_f32()?;
            par_chunks(&mut out, |off, dst| f(&src[off..off + dst.len()], dst));
        }
    }
    Tensor::from_vec_f32(out, x.shape())
}

/// Apply a scalar function elementwise over `x`.
pub(crate) fn map_f32(x: &Tensor, f: impl Fn(f32) -> f32 + Sync) -> Result<Tensor> {
    unary_f32(x, |s, d| {
        for (dd, &ss) in d.iter_mut().zip(s) {
            *dd = f(ss);
        }
    })
}

// ------------------------------------------------------------------ binary

/// Broadcasting elementwise binary op.
///
/// Works on arbitrary strides without materializing broadcast copies: the
/// innermost axis becomes a slice-level loop specialized on whether each side
/// is unit-stride (contiguous) or zero-stride (broadcast).
pub(crate) fn binary_f32(
    op: &'static str,
    a: &Tensor,
    b: &Tensor,
    f: impl Fn(f32, f32) -> f32 + Sync,
) -> Result<Tensor> {
    if a.dtype() != b.dtype() {
        return Err(Error::dtype(op, "mixed dtypes"));
    }
    let out_shape = broadcast_shapes(a.shape(), b.shape())?;
    let n = numel(&out_shape)?;
    let asrc = a.storage_f32()?;
    let bsrc = b.storage_f32()?;
    let (ab, bb) = (a.storage_offset(), b.storage_offset());

    if out_shape.is_empty() {
        let v = f(asrc[ab], bsrc[bb]);
        return Tensor::from_vec_f32(vec![v], &out_shape);
    }
    let mut out = vec![0f32; n];
    if n == 0 {
        return Tensor::from_vec_f32(out, &out_shape);
    }

    let sa = broadcast_strides(a.shape(), a.strides(), &out_shape)?;
    let sb = broadcast_strides(b.shape(), b.strides(), &out_shape)?;
    let rank = out_shape.len();
    let inner = out_shape[rank - 1];
    let (ia, ib) = (sa[rank - 1], sb[rank - 1]);
    let outer = n / inner.max(1);

    // Offsets of outer row `r`, by unraveling r over the leading axes.
    let row_offsets = |r: usize| -> (isize, isize) {
        let (mut ao, mut bo) = (ab as isize, bb as isize);
        let mut rem = r;
        for k in (0..rank - 1).rev() {
            let d = out_shape[k];
            let i = rem % d;
            rem /= d;
            ao += i as isize * sa[k];
            bo += i as isize * sb[k];
        }
        (ao, bo)
    };

    let row = |r: usize, dst: &mut [f32]| {
        let (ao, bo) = row_offsets(r);
        match (ia, ib) {
            (1, 1) => {
                let av = &asrc[ao as usize..ao as usize + inner];
                let bv = &bsrc[bo as usize..bo as usize + inner];
                for i in 0..inner {
                    dst[i] = f(av[i], bv[i]);
                }
            }
            (1, 0) => {
                let av = &asrc[ao as usize..ao as usize + inner];
                let s = bsrc[bo as usize];
                for i in 0..inner {
                    dst[i] = f(av[i], s);
                }
            }
            (0, 1) => {
                let s = asrc[ao as usize];
                let bv = &bsrc[bo as usize..bo as usize + inner];
                for i in 0..inner {
                    dst[i] = f(s, bv[i]);
                }
            }
            _ => {
                for i in 0..inner {
                    let av = asrc[(ao + i as isize * ia) as usize];
                    let bv = bsrc[(bo + i as isize * ib) as usize];
                    dst[i] = f(av, bv);
                }
            }
        }
    };

    if n < PAR_MIN {
        for (r, dst) in out.chunks_mut(inner).enumerate() {
            row(r, dst);
        }
    } else {
        let threads = rayon::current_num_threads().max(1);
        let rows_per = outer.div_ceil(threads).max(1);
        out.par_chunks_mut(rows_per * inner)
            .enumerate()
            .for_each(|(c, buf)| {
                for (j, dst) in buf.chunks_mut(inner).enumerate() {
                    row(c * rows_per + j, dst);
                }
            });
    }
    Tensor::from_vec_f32(out, &out_shape)
}

// --------------------------------------------------------------- reductions

/// Sum `x` over `reduced` axes into an output of `out_shape`.
///
/// Three paths: parallel tree-reduce for full reductions, parallel row sums
/// when only the last axis is reduced, and a dual-odometer walk otherwise
/// (which keeps both the input and output offsets incremental, so no stride
/// math or allocation happens per element).
pub(crate) fn sum_f32(x: &Tensor, reduced: &[bool], out_shape: &[usize]) -> Result<Tensor> {
    let rank = x.rank();
    let n = x.numel();
    let out_n = numel(out_shape)?;
    let all = reduced.iter().all(|&r| r);

    // Full reduction.
    if all {
        let total = match x.as_slice_f32() {
            Ok(s) => {
                if n < PAR_MIN {
                    s.iter().copied().sum::<f32>()
                } else {
                    s.par_chunks(1 << 14).map(|c| c.iter().copied().sum::<f32>()).sum()
                }
            }
            Err(_) => x.to_vec_f32()?.iter().copied().sum::<f32>(),
        };
        return Tensor::from_vec_f32(vec![total], out_shape);
    }

    // Last-axis-only reduction over a contiguous tensor: rows are independent.
    let last_only = rank > 0 && reduced[rank - 1] && !reduced[..rank - 1].iter().any(|&r| r);
    if last_only {
        if let Ok(src) = x.as_slice_f32() {
            let inner = x.shape()[rank - 1];
            let mut out = vec![0f32; out_n];
            par_chunks(&mut out, |off, dst| {
                for (j, d) in dst.iter_mut().enumerate() {
                    let r = off + j;
                    *d = src[r * inner..(r + 1) * inner].iter().copied().sum::<f32>();
                }
            });
            return Tensor::from_vec_f32(out, out_shape);
        }
    }

    // General case: dual odometer over input and output offsets.
    let mut ostr = vec![0isize; rank];
    {
        // Row-major strides of the *reduced* output, mapped back onto input dims.
        let mut kept: Vec<usize> = Vec::new();
        for k in 0..rank {
            if !reduced[k] {
                kept.push(k);
            }
        }
        let mut st = 1isize;
        for &k in kept.iter().rev() {
            ostr[k] = st;
            st *= x.shape()[k] as isize;
        }
    }
    let src = x.storage_f32()?;
    let shape = x.shape().to_vec();
    let xstr = x.strides().to_vec();
    let mut acc = vec![0f32; out_n];
    if n > 0 {
        let mut idx = vec![0usize; rank];
        let mut ioff = x.storage_offset() as isize;
        let mut ooff = 0isize;
        loop {
            acc[ooff as usize] += src[ioff as usize];
            let mut k = rank - 1;
            loop {
                idx[k] += 1;
                ioff += xstr[k];
                ooff += ostr[k];
                if idx[k] < shape[k] {
                    break;
                }
                ioff -= xstr[k] * shape[k] as isize;
                ooff -= ostr[k] * shape[k] as isize;
                idx[k] = 0;
                if k == 0 {
                    return Tensor::from_vec_f32(acc, out_shape);
                }
                k -= 1;
            }
        }
    }
    Tensor::from_vec_f32(acc, out_shape)
}


/// Fused GELU forward + local derivative in a single pass.
///
/// Training needs both `y` and `dy/dx`; computing them separately costs two
/// full passes plus two `tanh` evaluations. This shares one vForce `tanh`.
pub(crate) fn gelu_fwd_bwd(x: &Tensor) -> Result<(Tensor, Tensor)> {
    const K: f32 = 0.797_884_6; // sqrt(2/pi)
    const A: f32 = 0.044715;
    let n = x.numel();
    let owned;
    let src: &[f32] = match x.as_slice_f32() {
        Ok(s) => s,
        Err(_) => {
            owned = x.to_vec_f32()?;
            &owned
        }
    };
    let mut y = vec![0f32; n];
    let mut d = vec![0f32; n];

    let body = |src: &[f32], yc: &mut [f32], dc: &mut [f32]| {
        // dc <- k * (x + a x^3)
        for (dd, &s) in dc.iter_mut().zip(src) {
            *dd = K * (s + A * s * s * s);
        }
        // dc <- tanh(dc)
        let p = dc.as_ptr();
        k_tanh(unsafe { std::slice::from_raw_parts(p, dc.len()) }, dc);
        // y  <- 0.5 x (1 + th)
        for ((yy, &s), &th) in yc.iter_mut().zip(src).zip(dc.iter()) {
            *yy = 0.5 * s * (1.0 + th);
        }
        // dc <- 0.5(1+th) + 0.5 x (1-th^2) k (1 + 3a x^2)
        for (dd, &s) in dc.iter_mut().zip(src) {
            let th = *dd;
            *dd = 0.5 * (1.0 + th) + 0.5 * s * (1.0 - th * th) * K * (1.0 + 3.0 * A * s * s);
        }
    };

    if n < PAR_MIN {
        body(src, &mut y, &mut d);
    } else {
        let threads = rayon::current_num_threads().max(1);
        let chunk = (n.div_ceil(threads)).max(1024).div_ceil(16) * 16;
        y.par_chunks_mut(chunk)
            .zip(d.par_chunks_mut(chunk))
            .enumerate()
            .for_each(|(i, (yc, dc))| {
                let off = i * chunk;
                body(&src[off..off + yc.len()], yc, dc);
            });
    }
    Ok((
        Tensor::from_vec_f32(y, x.shape())?,
        Tensor::from_vec_f32(d, x.shape())?,
    ))
}
