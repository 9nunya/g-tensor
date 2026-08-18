//! The fused block must produce identical values and gradients to the
//! composed block. The composed block's primitives are individually
//! finite-difference gradchecked, so agreement here validates the fusion.

use g::prelude::*;
use g::{add, fused_block, gated_scan, mul, rms_norm, sigmoid, silu};
use g_core::Tensor;

#[allow(clippy::too_many_arguments)]
fn composed(
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
) -> g::Result<Tensor> {
    let xn = mul(&rms_norm(x, eps)?, g1)?;
    let ones_col = Tensor::from_vec_f32(
        vec![1.0; x.shape()[0] * x.shape()[1]],
        &[x.shape()[0], x.shape()[1], 1],
    )
    .unwrap();
    let xna = g::cat(&[&xn, &ones_col], 2).unwrap();
    let a = sigmoid(&xna.linear(wa, None)?)?;
    let b = xn.linear(wb, None)?;
    let h = gated_scan(&a, &b)?;
    let out = silu(&mul(&h, g2)?)?.linear(wo, None)?;
    let x = add(x, &out)?;
    let xn2 = mul(&rms_norm(&x, eps)?, g3)?;
    let f1 = silu(&xn2.linear(wf1, None)?)?;
    let f2 = mul(&f1.linear(wf2, None)?, g4)?;
    add(&x, &f2)
}

fn rel_err(a: &[f32], b: &[f32]) -> f32 {
    let mut worst = 0f32;
    for (x, y) in a.iter().zip(b) {
        let d = (x - y).abs() / (x.abs().max(y.abs()).max(1e-4));
        worst = worst.max(d);
    }
    worst
}

#[test]
fn fused_block_matches_composed_values_and_grads() {
    let (b, t, d) = (2usize, 5usize, 8usize);
    let mk = |n: usize, seed: u64| {
        Tensor::from_vec_f32(
            (0..n)
                .map(|i| ((i * 31 + seed as usize * 7) % 17) as f32 / 9.0 - 0.9)
                .collect(),
            &[n],
        )
        .unwrap()
    };
    let x0 = Tensor::from_vec_f32(
        (0..b * t * d)
            .map(|i| ((i * 13 % 11) as f32) / 5.0 - 1.0)
            .collect(),
        &[b, t, d],
    )
    .unwrap();
    let (wa0, wb0, wo0) = (
        mk((d + 1) * d, 1)
            .reshape(&[(d + 1) as isize, d as isize])
            .unwrap(),
        mk(d * d, 2).reshape(&[d as isize, d as isize]).unwrap(),
        mk(d * d, 3).reshape(&[d as isize, d as isize]).unwrap(),
    );
    let (wf10, wf20) = (
        mk(d * 2 * d, 4)
            .reshape(&[d as isize, (2 * d) as isize])
            .unwrap(),
        mk(d * 2 * d, 5)
            .reshape(&[(2 * d) as isize, d as isize])
            .unwrap(),
    );
    let (g10, g20, g30, g40) = (mk(d, 6), mk(d, 7), mk(d, 8), mk(d, 9));
    for g0 in [&g10, &g20, &g30, &g40] {
        assert_eq!(g0.shape(), &[d]);
    }
    let eps = 1e-5f32;

    // Values: fused == composed
    let yc = composed(
        &x0, &wa0, &wb0, &wo0, &wf10, &wf20, &g10, &g20, &g30, &g40, eps,
    )
    .unwrap();
    let yf = fused_block(
        &x0, &wa0, &wb0, &wo0, &wf10, &wf20, &g10, &g20, &g30, &g40, eps,
    )
    .unwrap();
    let e_val = rel_err(&yc.to_vec_f32().unwrap(), &yf.to_vec_f32().unwrap());
    println!("value rel err: {e_val:.2e}");
    assert!(e_val < 1e-5, "fused forward mismatch {e_val}");

    // Gradients w.r.t. every input via a weighted scalar objective.
    let wt = Tensor::from_vec_f32(
        (0..b * t * d).map(|i| 0.5 + (i % 4) as f32 * 0.1).collect(),
        &[b, t, d],
    )
    .unwrap();
    #[allow(clippy::type_complexity)]
    let grad_all = |fwd: &dyn Fn(
        &Tensor,
        &Tensor,
        &Tensor,
        &Tensor,
        &Tensor,
        &Tensor,
        &Tensor,
        &Tensor,
        &Tensor,
        &Tensor,
        f32,
    ) -> g::Result<Tensor>|
     -> Vec<Vec<f32>> {
        let xr = x0.clone().with_requires_grad();
        let war = wa0.clone().with_requires_grad();
        let wbr = wb0.clone().with_requires_grad();
        let wor = wo0.clone().with_requires_grad();
        let wf1r = wf10.clone().with_requires_grad();
        let wf2r = wf20.clone().with_requires_grad();
        let g1r = g10.clone().with_requires_grad();
        let g2r = g20.clone().with_requires_grad();
        let g3r = g30.clone().with_requires_grad();
        let g4r = g40.clone().with_requires_grad();
        let y = fwd(
            &xr, &war, &wbr, &wor, &wf1r, &wf2r, &g1r, &g2r, &g3r, &g4r, 1e-5,
        )
        .unwrap();
        let loss = mul(&y, &wt).unwrap().sum(None, false).unwrap();
        let gs = grad(
            &loss,
            &[&xr, &war, &wbr, &wor, &wf1r, &wf2r, &g1r, &g2r, &g3r, &g4r],
        )
        .unwrap();
        gs.iter().map(|g| g.to_vec_f32().unwrap()).collect()
    };
    let gc = grad_all(&composed);
    // fused: needs requires_grad leaves
    let xr = x0.clone().with_requires_grad();
    let war = wa0.clone().with_requires_grad();
    let wbr = wb0.clone().with_requires_grad();
    let wor = wo0.clone().with_requires_grad();
    let wf1r = wf10.clone().with_requires_grad();
    let wf2r = wf20.clone().with_requires_grad();
    let g1r = g10.clone().with_requires_grad();
    let g2r = g20.clone().with_requires_grad();
    let g3r = g30.clone().with_requires_grad();
    let g4r = g40.clone().with_requires_grad();
    let y = fused_block(
        &xr, &war, &wbr, &wor, &wf1r, &wf2r, &g1r, &g2r, &g3r, &g4r, eps,
    )
    .unwrap();
    let loss = mul(&y, &wt).unwrap().sum(None, false).unwrap();
    let gs = grad(
        &loss,
        &[&xr, &war, &wbr, &wor, &wf1r, &wf2r, &g1r, &g2r, &g3r, &g4r],
    )
    .unwrap();
    let names = ["x", "wa", "wb", "wo", "wf1", "wf2", "g1", "g2", "g3", "g4"];
    let mut worst = 0f32;
    for (i, (g, name)) in gs.iter().zip(names).enumerate() {
        let e = rel_err(&g.to_vec_f32().unwrap(), &gc[i]);
        worst = worst.max(e);
        println!("  grad {name:4} rel err {e:.2e}");
    }
    assert!(worst < 2e-3, "fused backward mismatch {worst}");
}
