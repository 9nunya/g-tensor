//! Fused HELIX block: the whole layer as one kernel sequence and one
//! autodiff node.
//!
//! The composed block is ~30 graph nodes and ~30 live tensors per layer.
//! Most of that is transient (the forward's intermediate values only exist
//! to be re-read by the backward pass), and each node carries its own
//! allocation, backward dispatch, and clone bookkeeping. This module
//! computes the same function with the same primitives, saves only the
//! values the backward genuinely needs, and exposes one node per layer.

use crate::fast;
use g_core::{Dtype, Error, Result, Tensor};
use rayon::prelude::*;

/// Values saved by the forward for the backward pass.
pub struct FusedAux {
    /// Block input `[B,T,D]`.
    pub x: Tensor,
    /// Scan gates `[B,T,D]`.
    pub a: Tensor,
    /// Scan state `[B,T,D]`.
    pub h: Tensor,
    /// `silu(h)*g2` `[B,T,D]`.
    pub s2: Tensor,
    /// `x + s2@Wo` `[B,T,D]`.
    pub out1: Tensor,
    /// `rms(out1)*g3` `[B,T,D]`.
    pub xn2: Tensor,
    /// `xn2@Wf1` `[B,T,2D]` (SiLU recomputed in backward).
    pub f1: Tensor,
    /// `silu(f1)@Wf2` `[B,T,D]`.
    pub f2: Tensor,
}

fn check(x: &Tensor, d: usize) -> Result<(usize, usize)> {
    if x.dtype() != Dtype::F32 {
        return Err(Error::dtype("fused_block", "f32 only"));
    }
    if x.rank() != 3 || x.shape()[2] != d {
        return Err(Error::shape("fused_block", "expected [B, T, D]"));
    }
    Ok((x.shape()[0], x.shape()[1]))
}

/// Forward. `wa wb wo` are `[D,D]`, `wf1 [D,2D]`, `wf2 [2D,D]`, gains `[D]`.
#[allow(clippy::too_many_arguments)]
pub fn fused_block_fwd(
    x: &Tensor,
    wa: &Tensor,
    wb: &Tensor,
    wo: &Tensor,
    wf1: &Tensor,
    wf2: &Tensor,
    g1: &Tensor,
    g2: &Tensor,
    g3: &Tensor,
    g4: &Tensor,
    eps: f32,
) -> Result<(Tensor, FusedAux)> {
    let d = x.shape()[2];
    check(x, d)?;

    let r1 = crate::rms_norm(x, eps)?;
    let xn = fast::binary_f32("fused", &r1, g1, |a, b| a * b)?;
    // Gate projection includes a constant-1 column so `wa` is [D+1, D] and
    // its last row is a learnable gate bias. Initialized at +3 (sigmoid ~0.95)
    // this makes the recurrence retain by default instead of decaying every
    // ~2 timesteps, which is what lets multi-token computations chain.
    let ones_col = Tensor::from_vec_f32(
        vec![1.0; x.shape()[0] * x.shape()[1]],
        &[x.shape()[0], x.shape()[1], 1],
    )?;
    let xna = crate::cat(&[&xn, &ones_col], 2)?;
    let ua = crate::matmul(&xna, wa)?;
    let a = crate::sigmoid(&ua)?;
    let ub = crate::matmul(&xn, wb)?;
    let h = crate::gated_scan(&a, &ub)?;
    // Composed order: silu(h * g2) -- the gain applies BEFORE the silu.
    let hg = fast::binary_f32("fused", &h, g2, |a, b| a * b)?;
    let s2 = fast::unary_f32(&hg, fast::k_silu)?;
    let so = crate::matmul(&s2, wo)?;
    let out1 = fast::binary_f32("fused", x, &so, |a, b| a + b)?;
    let r2 = crate::rms_norm(&out1, eps)?;
    let xn2 = fast::binary_f32("fused", &r2, g3, |a, b| a * b)?;
    let f1 = crate::matmul(&xn2, wf1)?;
    let sf1 = fast::unary_f32(&f1, fast::k_silu)?;
    let f2 = crate::matmul(&sf1, wf2)?;
    let f = fast::binary_f32("fused", &f2, g4, |a, b| a * b)?;
    let y = fast::binary_f32("fused", &out1, &f, |a, b| a + b)?;

    let aux = FusedAux {
        x: x.clone(),
        a,
        h,
        s2,
        out1,
        xn2,
        f1,
        f2,
    };
    Ok((y, aux))
}

#[inline]
fn sum01(t: &Tensor) -> Result<Tensor> {
    crate::sum(t, Some(&[0, 1]), false)
}

/// Batch-summed `A^T @ B`: `sum_b a_b^T b_b`, via the rank-3 BLAS path.
/// The product is `[B, K, N]`; only the BATCH axis is reduced.
fn bt_b(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let at = a.transpose()?; // [B, K, M]
    let prod = crate::matmul(&at, b)?; // [B, K, N]
    crate::sum(&prod, Some(&[0]), false)
}

/// Silu local derivative into `out`: `s(1 + x - x*s)`, `s = sigmoid(x)`,
/// vectorized exp via vForce.
fn silu_local(x: &Tensor, out: &mut [f32]) {
    let xv = x.to_vec_f32().expect("f32");
    out.par_chunks_mut(1 << 14)
        .enumerate()
        .for_each(|(i, chunk)| {
            let off = i * (1 << 14);
            let mut tmp: Vec<f32> = chunk
                .iter()
                .enumerate()
                .map(|(j, _)| -xv[off + j])
                .collect();
            let p = tmp.as_ptr();
            fast::k_exp(
                unsafe { std::slice::from_raw_parts(p, tmp.len()) },
                &mut tmp,
            );
            for (j, o) in chunk.iter_mut().enumerate() {
                let s = 1.0 / (1.0 + tmp[j]);
                *o = s * (1.0 + xv[off + j] - xv[off + j] * s);
            }
        });
}

/// Backward. Returns grads in order (x, wa, wb, wo, wf1, wf2, g1, g2, g3, g4).
#[allow(clippy::too_many_arguments)]
#[allow(clippy::type_complexity)]
pub fn fused_block_bwd(
    aux: &FusedAux,
    wa: &Tensor,
    wb: &Tensor,
    wo: &Tensor,
    wf1: &Tensor,
    wf2: &Tensor,
    g1: &Tensor,
    g2: &Tensor,
    g3: &Tensor,
    g4: &Tensor,
    eps: f32,
    gy: &Tensor,
) -> Result<(
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
)> {
    let x = &aux.x;
    let _dbg = std::env::var("FUSED_PROF").is_ok();
    macro_rules! t0 {
        () => {
            if _dbg {
                std::time::Instant::now()
            } else {
                std::time::Instant::now()
            }
        };
    }
    macro_rules! tlog {
        ($name:expr, $t:ident) => {
            if _dbg {
                eprintln!("  {} {:?}", $name, $t.elapsed());
            }
        };
    }

    // --- tail: f = f2*g4, y = out1 + f ---
    let mut _tt = t0!();
    let gg4 = sum01(&fast::binary_f32("fused", gy, &aux.f2, |a, b| a * b)?)?;
    let dm = fast::binary_f32("fused", gy, g4, |a, b| a * b)?;
    tlog!("tail", _tt);
    let mut _tt = t0!();
    let sf1 = fast::unary_f32(&aux.f1, fast::k_silu)?;
    let gwf2 = bt_b(&sf1, &dm)?; // [2D, D]
    let dsf1 = crate::matmul(&dm, &wf2.transpose()?)?; // [B,T,D]@[D,2D]
    let mut sl = vec![0f32; aux.f1.numel()];
    silu_local(&aux.f1, &mut sl);
    let sl_t = Tensor::from_vec_f32(sl, aux.f1.shape())?;
    let d_pre = fast::binary_f32("fused", &dsf1, &sl_t, |a, b| a * b)?;
    let gwf1 = bt_b(&aux.xn2, &d_pre)?; // [D, 2D]
    let dxn2 = crate::matmul(&d_pre, &wf1.transpose()?)?; // [B,T,2D]@[2D,D]
    tlog!("ffn", _tt);
    let mut _tt = t0!();

    // --- rms #2 with gain g3 ---
    let r2 = crate::rms_norm(&aux.out1, eps)?;
    let gg3 = sum01(&fast::binary_f32("fused", &dxn2, &r2, |a, b| a * b)?)?;
    let dxn2g = fast::binary_f32("fused", &dxn2, g3, |a, b| a * b)?;
    // out1 receives TWO gradients: through the residual add y = out1 + f
    // (plain gy) and through rms_norm #2.
    let dout1 = fast::binary_f32(
        "fused",
        &crate::rms_norm_backward(&aux.out1, &dxn2g, eps)?,
        gy,
        |a, b| a + b,
    )?;
    tlog!("rms2", _tt);
    let mut _tt = t0!();

    // --- out1 = x + s2@wo ---
    let gs2 = crate::matmul(&dout1, &wo.transpose()?)?;
    let gwo = bt_b(&aux.s2, &dout1)?;
    let mut dx = dout1;
    tlog!("wo", _tt);
    let mut _tt = t0!();

    // --- gain 2 + silu(h): s2 = silu(h * g2) ---
    let hg = fast::binary_f32("fused", &aux.h, g2, |a, b| a * b)?;
    let gg2 = sum01(&fast::binary_f32("fused", &gs2, &hg, |a, b| a * b)?)?; // will redo below
    let _ = gg2;
    let mut sl_h = vec![0f32; aux.h.numel()];
    silu_local(&hg, &mut sl_h);
    let slh_t = Tensor::from_vec_f32(sl_h, hg.shape())?;
    let ghg = fast::binary_f32("fused", &gs2, &slh_t, |a, b| a * b)?;
    let dh = fast::binary_f32("fused", &ghg, g2, |a, b| a * b)?;
    let gg2 = sum01(&fast::binary_f32("fused", &ghg, &aux.h, |a, b| a * b)?)?;
    tlog!("silu2", _tt);
    let mut _tt = t0!();

    // --- scan backward ---
    let (ga, gb) = crate::gated_scan_backward(&aux.a, &aux.h, &dh)?;
    tlog!("scan", _tt);
    let mut _tt = t0!();

    // --- sigmoid gate: a = sigmoid(xn@wa) ---
    let one_minus_a = fast::unary_f32(&aux.a, |s, d| {
        for (dd, &v) in d.iter_mut().zip(s) {
            *dd = 1.0 - v;
        }
    })?;
    let a_a1 = fast::binary_f32("fused", &aux.a, &one_minus_a, |a, b| a * b)?;
    let gua = fast::binary_f32("fused", &ga, &a_a1, |a, b| a * b)?;
    let r1 = crate::rms_norm(x, eps)?;
    let xn = fast::binary_f32("fused", &r1, g1, |a, b| a * b)?;
    tlog!("gate", _tt);
    let mut _tt = t0!();
    let ones_col = Tensor::from_vec_f32(
        vec![1.0; x.shape()[0] * x.shape()[1]],
        &[x.shape()[0], x.shape()[1], 1],
    )?;
    let xna = crate::cat(&[&xn, &ones_col], 2)?;
    let gwa = bt_b(&xna, &gua)?; // [D+1, D]: the bias row learns too
    let gwb = bt_b(&xn, &gb)?;
    // dxn uses only the first D rows of wa (the bias column had grad 0 by
    // construction: the ones column is constant).
    let wa_top = wa.slice(&[
        (Some(0), Some((wa.shape()[0] - 1) as isize), None),
        (None, None, None),
    ])?;
    let dxn1 = fast::binary_f32(
        "fused",
        &crate::matmul(&gua, &wa_top.transpose()?)?,
        &crate::matmul(&gb, &wb.transpose()?)?,
        |a, b| a + b,
    )?;

    // --- rms #1 with gain g1 ---
    let gg1 = sum01(&fast::binary_f32("fused", &dxn1, &r1, |a, b| a * b)?)?;
    let dxn1g = fast::binary_f32("fused", &dxn1, g1, |a, b| a * b)?;
    let dx_rms = crate::rms_norm_backward(x, &dxn1g, eps)?;
    dx = fast::binary_f32("fused", &dx, &dx_rms, |a, b| a + b)?;
    tlog!("rms1+matmuls", _tt);

    Ok((dx, gwa, gwb, gwo, gwf1, gwf2, gg1, gg2, gg3, gg4))
}
