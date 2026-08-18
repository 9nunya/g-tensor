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

use g_core::{broadcast_shapes, broadcast_strides, numel, Dtype, Error, Result, Tensor};
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
    let tmp = dst.as_ptr();
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
    let tmp = dst.as_ptr();
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
                    s.par_chunks(1 << 14)
                        .map(|c| c.iter().copied().sum::<f32>())
                        .sum()
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
        for (k, is_reduced) in reduced.iter().enumerate().take(rank) {
            if !is_reduced {
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

/// Split `out` into row blocks and process them in parallel.
/// `f(first_row_index, block)` sees whole rows of length `inner`.
#[inline]
pub(crate) fn par_rows(out: &mut [f32], inner: usize, f: impl Fn(usize, &mut [f32]) + Sync) {
    if inner == 0 || out.len() < PAR_MIN {
        f(0, out);
        return;
    }
    let rows = out.len() / inner;
    let threads = rayon::current_num_threads().max(1);
    let rpc = rows.div_ceil(threads).max(1);
    out.par_chunks_mut(rpc * inner)
        .enumerate()
        .for_each(|(c, buf)| f(c * rpc, buf));
}

/// Borrow `x` as a contiguous f32 slice, materializing only if strided.
macro_rules! as_contig {
    ($x:expr, $owned:ident) => {
        match $x.as_slice_f32() {
            Ok(s) => s,
            Err(_) => {
                $owned = $x.to_vec_f32()?;
                &$owned
            }
        }
    };
}

/// Numerically stable softmax over the last axis, fused into one pass per row.
pub(crate) fn softmax_last_f32(x: &Tensor) -> Result<Tensor> {
    let rank = x.rank();
    if rank == 0 {
        return Err(Error::shape("softmax", "rank-0"));
    }
    let inner = x.shape()[rank - 1];
    let owned;
    let src = as_contig!(x, owned);
    let mut out = vec![0f32; x.numel()];
    par_rows(&mut out, inner, |r0, blk| {
        for (j, row) in blk.chunks_mut(inner).enumerate() {
            let s = &src[(r0 + j) * inner..(r0 + j + 1) * inner];
            let m = s.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            for (d, &v) in row.iter_mut().zip(s) {
                *d = v - m;
            }
            let p = row.as_ptr();
            k_exp(unsafe { std::slice::from_raw_parts(p, row.len()) }, row);
            let sum: f32 = row.iter().copied().sum();
            let inv = 1.0 / sum;
            for d in row.iter_mut() {
                *d *= inv;
            }
        }
    });
    Tensor::from_vec_f32(out, x.shape())
}

/// Numerically stable log-softmax over the last axis.
pub(crate) fn log_softmax_last_f32(x: &Tensor) -> Result<Tensor> {
    let rank = x.rank();
    if rank == 0 {
        return Err(Error::shape("log_softmax", "rank-0"));
    }
    let inner = x.shape()[rank - 1];
    let owned;
    let src = as_contig!(x, owned);
    let mut out = vec![0f32; x.numel()];
    par_rows(&mut out, inner, |r0, blk| {
        let mut scratch = vec![0f32; inner];
        for (j, row) in blk.chunks_mut(inner).enumerate() {
            let s = &src[(r0 + j) * inner..(r0 + j + 1) * inner];
            let m = s.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            for (d, &v) in row.iter_mut().zip(s) {
                *d = v - m;
            }
            k_exp(row, &mut scratch);
            let sum: f32 = scratch.iter().copied().sum();
            let lse = sum.ln();
            for d in row.iter_mut() {
                *d -= lse;
            }
        }
    });
    Tensor::from_vec_f32(out, x.shape())
}

/// Max over `ax`. Fast row path when `ax` is the last axis.
pub(crate) fn amax_f32(x: &Tensor, ax: usize, out_shape: &[usize]) -> Result<Tensor> {
    let rank = x.rank();
    let out_n = numel(out_shape)?;
    if ax == rank - 1 {
        let inner = x.shape()[rank - 1];
        let owned;
        let src = as_contig!(x, owned);
        let mut out = vec![0f32; out_n];
        par_chunks(&mut out, |off, dst| {
            for (j, d) in dst.iter_mut().enumerate() {
                let r = off + j;
                *d = src[r * inner..(r + 1) * inner]
                    .iter()
                    .copied()
                    .fold(f32::NEG_INFINITY, f32::max);
            }
        });
        return Tensor::from_vec_f32(out, out_shape);
    }
    // General axis: dual odometer, no per-element allocation.
    let mut ostr = vec![0isize; rank];
    let mut st = 1isize;
    for k in (0..rank).rev() {
        if k != ax {
            ostr[k] = st;
            st *= x.shape()[k] as isize;
        }
    }
    let src = x.storage_f32()?;
    let shape = x.shape().to_vec();
    let xstr = x.strides().to_vec();
    let mut acc = vec![f32::NEG_INFINITY; out_n];
    let mut idx = vec![0usize; rank];
    let mut ioff = x.storage_offset() as isize;
    let mut ooff = 0isize;
    loop {
        let v = src[ioff as usize];
        let slot = &mut acc[ooff as usize];
        if v > *slot {
            *slot = v;
        }
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

/// Embedding lookup: `out[i0..i0+inner] = table[idx[i0]]`.
///
/// Pure random-access gather, parallel over rows.
pub(crate) fn embedding_f32(table: &Tensor, idx: &Tensor) -> Result<Tensor> {
    if idx.dtype() != Dtype::I64 {
        return Err(Error::dtype("embedding", "indices must be i64"));
    }
    let inner = table.shape()[1];
    let v = table.shape()[0] as i64;
    let tv = table.as_slice_f32()?;
    let iv = idx.storage_i64()?;
    let shape = idx.shape();
    let mut out = vec![0f32; idx.numel() * inner];
    // Row-aligned parallel split: chunks must land on row boundaries.
    par_rows(&mut out, inner, |r0, blk| {
        for (j, d) in blk.chunks_mut(inner).enumerate() {
            let class = iv[idx.storage_offset() + r0 + j];
            if class < 0 || class >= v {
                d.fill(0.0);
                continue;
            }
            let s = class as usize * inner;
            d.copy_from_slice(&tv[s..s + inner]);
        }
    });
    Tensor::from_vec_f32(
        out,
        &shape.iter().chain(&[inner]).cloned().collect::<Vec<_>>(),
    )
}

/// Backward of [`embedding_f32`]: scatter-add `gy` into the table rows.
pub(crate) fn embedding_backward_f32(table: &Tensor, idx: &Tensor, gy: &Tensor) -> Result<Tensor> {
    let v = table.shape()[0];
    let inner = table.shape()[1];
    let iv = idx.to_vec_i64()?;
    let gv = gy.to_vec_f32()?;
    let mut out = vec![0f32; v * inner];
    // Positions per class, then one clean pass per class (parallel).
    let mut buckets: Vec<Vec<usize>> = (0..v).map(|_| Vec::new()).collect();
    for (p, &c) in iv.iter().enumerate() {
        if c >= 0 && (c as usize) < v {
            buckets[c as usize].push(p);
        }
    }
    out.par_chunks_mut(inner).enumerate().for_each(|(c, row)| {
        for &p in &buckets[c] {
            let g = &gv[p * inner..p * inner + inner];
            for (acc, gv_) in row.iter_mut().zip(g) {
                *acc += gv_;
            }
        }
    });
    Tensor::from_vec_f32(out, table.shape())
}

/// Argmax over the last axis -> `i64` indices.
pub(crate) fn argmax_last_f32(x: &Tensor) -> Result<Tensor> {
    let rank = x.rank();
    if rank == 0 {
        return Err(Error::shape("argmax", "rank >= 1"));
    }
    let inner = x.shape()[rank - 1];
    let owned;
    let src = as_contig!(x, owned);
    let out_shape: Vec<usize> = x.shape()[..rank - 1].to_vec();
    let mut out = vec![0i64; x.numel() / inner.max(1)];
    par_chunks(&mut out, |off, dst| {
        for (j, d) in dst.iter_mut().enumerate() {
            let r = off + j;
            let s = &src[r * inner..(r + 1) * inner];
            let mut best = 0usize;
            let mut bv = f32::NEG_INFINITY;
            for (k, &val) in s.iter().enumerate() {
                if val > bv {
                    bv = val;
                    best = k;
                }
            }
            *d = best as i64;
        }
    });
    Tensor::from_vec_i64(out, &out_shape)
}

/// Concatenate along `ax` with contiguous block copies.
pub(crate) fn cat_f32(tensors: &[&Tensor], ax: usize, out_shape: &[usize]) -> Result<Tensor> {
    let inner: usize = out_shape[ax + 1..].iter().product();
    let outer: usize = out_shape[..ax].iter().product();
    let out_ax = out_shape[ax];
    let mut out = vec![0f32; numel(out_shape)?];
    let mut cursor = 0usize;
    for t in tensors {
        let owned;
        let tv = as_contig!(t, owned);
        let tax = t.shape()[ax];
        let span = tax * inner;
        for o in 0..outer {
            let dst = o * out_ax * inner + cursor * inner;
            out[dst..dst + span].copy_from_slice(&tv[o * span..o * span + span]);
        }
        cursor += tax;
    }
    Tensor::from_vec_f32(out, out_shape)
}

/// Masked cross-entropy: `loss = -mean(mask_i * log p_i[target_i])`.
///
/// Returns the scalar loss and the softmax probabilities (needed by the
/// backward pass). `count == 0` gives loss 0 and zero gradient.
pub(crate) fn masked_ce_f32(
    logits: &Tensor,
    targets: &Tensor,
    mask: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let inner = logits.shape()[1];
    let owned;
    let src = as_contig!(logits, owned);
    let tg = targets.to_vec_i64()?;
    let mk = mask.to_vec_f32()?;
    let n = tg.len();
    if n * inner != logits.numel() || mk.len() != n {
        return Err(Error::shape(
            "masked_ce",
            format!(
                "targets/mask length {n} inconsistent with logits {}x{inner}",
                logits.shape()[0]
            ),
        ));
    }
    // |mask| normalization: RL passes advantage weights (possibly negative)
    // through the mask, and the count must stay positive.
    let count: f32 = mk.iter().map(|m| m.abs()).sum();
    let mut probs = vec![0f32; n * inner];
    par_rows(&mut probs, inner, |r0, blk| {
        let mut scratch = vec![0f32; inner];
        for (j, row) in blk.chunks_mut(inner).enumerate() {
            let i = r0 + j;
            let s = &src[i * inner..i * inner + inner];
            let m = s.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            for (d, &v) in row.iter_mut().zip(s) {
                *d = v - m;
            }
            k_exp(row, &mut scratch);
            let sum: f32 = scratch.iter().copied().sum();
            for (d, &v) in row.iter_mut().zip(scratch.iter()) {
                *d = v / sum;
            }
        }
    });
    // Sequential pass for exact, reproducible loss summation. Masks are
    // signed weights: ordinary supervised rows use 0/1, while policy-gradient
    // callers may use negative advantages. Normalize by |weight| so opposite
    // signs do not create an unstable near-zero denominator.
    let mut loss = 0f32;
    if count > 0.0 {
        for i in 0..n {
            if mk[i] != 0.0 {
                let t = tg[i];
                if t >= 0 && (t as usize) < inner {
                    // Clamp against softmax underflow: ln(0) = -inf would
                    // poison the loss and, through it, the gradients.
                    let p = probs[i * inner + t as usize].clamp(1e-30, 1.0);
                    loss -= mk[i] * p.ln();
                }
            }
        }
        loss /= count;
    }
    Ok((
        Tensor::from_vec_f32(vec![loss], &[])?,
        Tensor::from_vec_f32(probs, logits.shape())?,
    ))
}

/// Backward of [`masked_ce_f32`]: `dlogits_i = (p_i - onehot(t_i)) * mask_i / count`.
pub(crate) fn masked_ce_backward_f32(
    probs: &Tensor,
    targets: &Tensor,
    mask: &Tensor,
) -> Result<Tensor> {
    let inner = probs.shape()[1];
    let pv = probs.to_vec_f32()?;
    let tg = targets.to_vec_i64()?;
    let mk = mask.to_vec_f32()?;
    let n = tg.len();
    if n * inner != probs.numel() || mk.len() != n {
        return Err(Error::shape(
            "masked_ce_backward",
            format!(
                "targets/mask length {n} inconsistent with probs {}x{inner}",
                probs.shape()[0]
            ),
        ));
    }
    let count: f32 = mk.iter().map(|m| m.abs()).sum();
    let mut g = vec![0f32; n * inner];
    if count > 0.0 {
        let inv = 1.0 / count;
        par_rows(&mut g, inner, |r0, blk| {
            for (j, row) in blk.chunks_mut(inner).enumerate() {
                let i = r0 + j;
                if mk[i] == 0.0 {
                    continue;
                }
                let s = &pv[i * inner..i * inner + inner];
                for (k, (d, &p)) in row.iter_mut().zip(s).enumerate() {
                    let oh = if tg[i] as usize == k { 1.0 } else { 0.0 };
                    *d = (p - oh) * mk[i] * inv;
                }
            }
        });
    }
    Tensor::from_vec_f32(g, probs.shape())
}

/// Sigmoid value and local derivative in one pass (training pays one pass
/// instead of the fwd + a two-pass local + a backward mul).
pub(crate) fn sigmoid_fwd_bwd(x: &Tensor) -> Result<(Tensor, Tensor)> {
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
    // d <- -x, then vectorized exp via vForce, then the final division —
    // scalar .exp() in the hot loop is ~10x slower than vvexpf.
    let body = |src: &[f32], yc: &mut [f32], dc: &mut [f32]| {
        for (i, &s) in src.iter().enumerate() {
            dc[i] = -s;
        }
        let p = dc.as_ptr();
        k_exp(unsafe { std::slice::from_raw_parts(p, dc.len()) }, dc);
        for i in 0..src.len() {
            let v = 1.0 / (1.0 + dc[i]);
            yc[i] = v;
            dc[i] = v * (1.0 - v);
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

/// SiLU value and local derivative in one pass (one sigmoid shared between
/// the value and the derivative).
pub(crate) fn silu_fwd_bwd(x: &Tensor) -> Result<(Tensor, Tensor)> {
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
        for (i, &s) in src.iter().enumerate() {
            dc[i] = -s;
        }
        let p = dc.as_ptr();
        k_exp(unsafe { std::slice::from_raw_parts(p, dc.len()) }, dc);
        for (i, &s) in src.iter().enumerate() {
            let sg = 1.0 / (1.0 + dc[i]);
            yc[i] = s * sg;
            // silu' = sigmoid + x*sigmoid*(1-sigmoid)
            dc[i] = sg * (1.0 + s - s * sg);
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
