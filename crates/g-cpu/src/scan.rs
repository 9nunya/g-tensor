//! Gated linear recurrence (associative scan) as a single fused op.
//!
//! The recurrence `h_t = a_t * h_{t-1} + b_t` is the workhorse of linear-RNN /
//! selective-state-space sequence models. Expressing it with per-timestep
//! tensor ops would build one autodiff graph node per timestep, which is fatal
//! for both memory and speed at T in the hundreds. Here it is one op with a
//! hand-written backward.
//!
//! Layout is `[B, T, D]`. The scan is sequential in `T` (inherent) but fully
//! vectorized across `D` (contiguous inner loop) and parallel across `B`.

use crate::fast;
use g_core::{Dtype, Error, Result, Tensor};
use rayon::prelude::*;

fn check(a: &Tensor, b: &Tensor) -> Result<(usize, usize, usize)> {
    if a.dtype() != Dtype::F32 || b.dtype() != Dtype::F32 {
        return Err(Error::dtype("gated_scan", "f32 only"));
    }
    if a.rank() != 3 || b.rank() != 3 {
        return Err(Error::shape("gated_scan", "expected [B, T, D]"));
    }
    if a.shape() != b.shape() {
        return Err(Error::shape("gated_scan", "gate/input shape mismatch"));
    }
    Ok((a.shape()[0], a.shape()[1], a.shape()[2]))
}

/// Forward scan: `h_t = a_t * h_{t-1} + b_t`, with `h_{-1} = 0`.
pub fn gated_scan(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let (bs, t, d) = check(a, b)?;
    let (ao, bo);
    let av = match a.as_slice_f32() {
        Ok(s) => s,
        Err(_) => {
            ao = a.to_vec_f32()?;
            &ao
        }
    };
    let bv = match b.as_slice_f32() {
        Ok(s) => s,
        Err(_) => {
            bo = b.to_vec_f32()?;
            &bo
        }
    };
    let mut h = vec![0f32; bs * t * d];
    let step = t * d;
    h.par_chunks_mut(step).enumerate().for_each(|(bi, hb)| {
        let ab = &av[bi * step..bi * step + step];
        let bb = &bv[bi * step..bi * step + step];
        let mut state = vec![0f32; d];
        for ti in 0..t {
            let off = ti * d;
            let (ar, br, hr) = (&ab[off..off + d], &bb[off..off + d], &mut hb[off..off + d]);
            for j in 0..d {
                state[j] = ar[j] * state[j] + br[j];
                hr[j] = state[j];
            }
        }
    });
    Tensor::from_vec_f32(h, a.shape())
}

/// Backward for [`gated_scan`].
///
/// With `G_t = dL/dh_t` accumulated in reverse,
/// `G_t = gh_t + a_{t+1} G_{t+1}`, `dL/db_t = G_t`, `dL/da_t = G_t * h_{t-1}`.
pub fn gated_scan_backward(a: &Tensor, h: &Tensor, gh: &Tensor) -> Result<(Tensor, Tensor)> {
    let (bs, t, d) = check(a, h)?;
    let av = a.to_vec_f32()?;
    let hv = h.to_vec_f32()?;
    let gv = gh.to_vec_f32()?;
    let step = t * d;
    let mut ga = vec![0f32; bs * step];
    let mut gb = vec![0f32; bs * step];
    ga.par_chunks_mut(step)
        .zip(gb.par_chunks_mut(step))
        .enumerate()
        .for_each(|(bi, (gab, gbb))| {
            let ab = &av[bi * step..bi * step + step];
            let hb = &hv[bi * step..bi * step + step];
            let ghb = &gv[bi * step..bi * step + step];
            let mut carry = vec![0f32; d];
            for ti in (0..t).rev() {
                let off = ti * d;
                for j in 0..d {
                    let nxt = if ti + 1 < t {
                        ab[off + d + j] * carry[j]
                    } else {
                        0.0
                    };
                    let g = ghb[off + j] + nxt;
                    carry[j] = g;
                    gbb[off + j] = g;
                    gab[off + j] = if ti > 0 { g * hb[off - d + j] } else { 0.0 };
                }
            }
        });
    Ok((
        Tensor::from_vec_f32(ga, a.shape())?,
        Tensor::from_vec_f32(gb, a.shape())?,
    ))
}

/// Fused RMS normalization over the last axis: `x / sqrt(mean(x^2) + eps)`.
pub fn rms_norm(x: &Tensor, eps: f32) -> Result<Tensor> {
    let rank = x.rank();
    if rank == 0 {
        return Err(Error::shape("rms_norm", "rank >= 1"));
    }
    let inner = x.shape()[rank - 1];
    let owned;
    let src = match x.as_slice_f32() {
        Ok(s) => s,
        Err(_) => {
            owned = x.to_vec_f32()?;
            &owned
        }
    };
    let mut out = vec![0f32; x.numel()];
    fast::par_rows(&mut out, inner, |r0, blk| {
        for (j, row) in blk.chunks_mut(inner).enumerate() {
            let s = &src[(r0 + j) * inner..(r0 + j + 1) * inner];
            let ms = s.iter().map(|v| v * v).sum::<f32>() / inner as f32;
            let scale = 1.0 / (ms + eps).sqrt();
            for (d, &v) in row.iter_mut().zip(s) {
                *d = v * scale;
            }
        }
    });
    Tensor::from_vec_f32(out, x.shape())
}

/// Backward for [`rms_norm`], given the original input.
pub fn rms_norm_backward(x: &Tensor, gy: &Tensor, eps: f32) -> Result<Tensor> {
    let rank = x.rank();
    let inner = x.shape()[rank - 1];
    let xv = x.to_vec_f32()?;
    let gv = gy.to_vec_f32()?;
    let mut out = vec![0f32; x.numel()];
    fast::par_rows(&mut out, inner, |r0, blk| {
        for (j, row) in blk.chunks_mut(inner).enumerate() {
            let base = (r0 + j) * inner;
            let xs = &xv[base..base + inner];
            let gs = &gv[base..base + inner];
            let n = inner as f32;
            let ms = xs.iter().map(|v| v * v).sum::<f32>() / n;
            let inv = 1.0 / (ms + eps).sqrt();
            let dot = xs.iter().zip(gs).map(|(a, b)| a * b).sum::<f32>();
            let c = dot * inv * inv * inv / n;
            for k in 0..inner {
                row[k] = gs[k] * inv - xs[k] * c;
            }
        }
    });
    Tensor::from_vec_f32(out, x.shape())
}
