//! Step-by-step backward comparison: fused vs composed.
use g::prelude::*;
use g::{fused_block, gated_scan, mul, rms_norm, sigmoid, silu};
use g_core::{Dtype, Tensor};

fn diff(name: &str, a: &Tensor, b: &Tensor) {
    let (av, bv) = (a.to_vec_f32().unwrap(), b.to_vec_f32().unwrap());
    let mut m = 0f32;
    for (x, y) in av.iter().zip(bv.iter()) {
        m = m.max((x - y).abs() / (x.abs().max(y.abs()).max(1e-4)));
    }
    println!(
        "{name:22} shapes {:?} vs {:?} max rel {m:.2e}",
        a.shape(),
        b.shape()
    );
}

fn main() {
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
        mk(d * d, 1).reshape(&[d as isize, d as isize]).unwrap(),
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
    let eps = 1e-5f32;
    let wt = Tensor::from_vec_f32(
        (0..b * t * d).map(|i| 0.5 + (i % 4) as f32 * 0.1).collect(),
        &[b, t, d],
    )
    .unwrap();
    let gy = &wt;

    // composed forward intermediates
    let r1 = rms_norm(&x0, eps).unwrap();
    let xn = mul(&r1, &g10).unwrap();
    let a = sigmoid(&xn.linear(&wa0, None).unwrap()).unwrap();
    let bb = xn.linear(&wb0, None).unwrap();
    let h = gated_scan(&a, &bb).unwrap();
    let hg = mul(&h, &g20).unwrap();
    let s2 = silu(&hg).unwrap();
    let out1 = add(&x0, &s2.linear(&wo0, None).unwrap()).unwrap();
    let r2 = rms_norm(&out1, eps).unwrap();
    let xn2 = mul(&r2, &g30).unwrap();
    let f1 = xn2.linear(&wf10, None).unwrap();
    let sf1 = silu(&f1).unwrap();
    let f2 = sf1.linear(&wf20, None).unwrap();
    let y = add(&out1, &mul(&f2, &g40).unwrap()).unwrap();

    // composed backward chain (hand-built with kernels)
    let df = gy.clone();
    let gg4c = g_cpu::sum(&g_cpu::mul(&df, &f2).unwrap(), Some(&[0, 1]), false).unwrap();
    let dm = g_cpu::mul(&df, &g40).unwrap();
    let gwf2c = g_cpu::sum(
        &g_cpu::matmul(&sf1.transpose().unwrap(), &dm).unwrap(),
        Some(&[0, 1]),
        false,
    )
    .unwrap();
    let dsf1 = g_cpu::matmul(&dm, &wf20.transpose().unwrap()).unwrap();
    let (_, sl1) = g_cpu::silu_with_grad(&f1).unwrap();
    let d_pre = g_cpu::mul(&dsf1, &sl1).unwrap();
    let gwf1c = g_cpu::sum(
        &g_cpu::matmul(&xn2.transpose().unwrap(), &d_pre).unwrap(),
        Some(&[0, 1]),
        false,
    )
    .unwrap();
    let dxn2 = g_cpu::matmul(&d_pre, &wf10.transpose().unwrap()).unwrap();
    let gg3c = g_cpu::sum(&g_cpu::mul(&dxn2, &r2).unwrap(), Some(&[0, 1]), false).unwrap();
    let dout1 = g_cpu::rms_norm_backward(&out1, &g_cpu::mul(&dxn2, &g30).unwrap(), eps).unwrap();
    let gs2 = g_cpu::matmul(&dout1, &wo0.transpose().unwrap()).unwrap();
    let gwo_c = g_cpu::sum(
        &g_cpu::matmul(&s2.transpose().unwrap(), &dout1).unwrap(),
        Some(&[0, 1]),
        false,
    )
    .unwrap();
    let gg2c = g_cpu::sum(&g_cpu::mul(&gs2, &hg).unwrap(), Some(&[0, 1]), false).unwrap();
    let (_, slh) = g_cpu::silu_with_grad(&hg).unwrap();
    let ghg = g_cpu::mul(&gs2, &slh).unwrap();
    let dh = g_cpu::mul(&ghg, &g20).unwrap();
    let (ga, gb) = g_cpu::gated_scan_backward(&a, &h, &dh).unwrap();
    let a1ma = g_cpu::mul(
        &a,
        &g_cpu::sub(&Tensor::ones(a.shape(), Dtype::F32).unwrap(), &a).unwrap(),
    )
    .unwrap();
    let gua = g_cpu::mul(&ga, &a1ma).unwrap();
    let gwa_c = g_cpu::sum(
        &g_cpu::matmul(&xn.transpose().unwrap(), &gua).unwrap(),
        Some(&[0, 1]),
        false,
    )
    .unwrap();
    let gwb_c = g_cpu::sum(
        &g_cpu::matmul(&xn.transpose().unwrap(), &gb).unwrap(),
        Some(&[0, 1]),
        false,
    )
    .unwrap();
    let dxn1 = g_cpu::add(
        &g_cpu::matmul(&gua, &wa0.transpose().unwrap()).unwrap(),
        &g_cpu::matmul(&gb, &wb0.transpose().unwrap()).unwrap(),
    )
    .unwrap();
    let gg1c = g_cpu::sum(&g_cpu::mul(&dxn1, &r1).unwrap(), Some(&[0, 1]), false).unwrap();
    let dx = g_cpu::add(
        &dout1,
        &g_cpu::rms_norm_backward(&x0, &g_cpu::mul(&dxn1, &g10).unwrap(), eps).unwrap(),
    )
    .unwrap();

    // fused backward intermediates via the fused op grads
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
    let yf = fused_block(
        &xr, &war, &wbr, &wor, &wf1r, &wf2r, &g1r, &g2r, &g3r, &g4r, eps,
    )
    .unwrap();
    let loss = mul(&yf, &wt).unwrap().sum(None, false).unwrap();
    let gs = grad(
        &loss,
        &[&xr, &war, &wbr, &wor, &wf1r, &wf2r, &g1r, &g2r, &g3r, &g4r],
    )
    .unwrap();

    println!("=== value check ===");
    diff("y", &y, &yf);
    println!("=== backward piece checks ===");
    diff("gx", &dx, &gs[0]);
    diff("gwa", &gwa_c, &gs[1]);
    diff("gwb", &gwb_c, &gs[2]);
    diff("gwo", &gwo_c, &gs[3]);
    diff("gwf1", &gwf1c, &gs[4]);
    diff("gwf2", &gwf2c, &gs[5]);
    diff("gg1", &gg1c, &gs[6]);
    diff("gg2", &gg2c, &gs[7]);
    diff("gg3", &gg3c, &gs[8]);
    diff("gg4", &gg4c, &gs[9]);
    // check the intermediate tensors directly
    diff(
        "dout1 (chk)",
        &dout1,
        &g_cpu::rms_norm_backward(&out1, &g_cpu::mul(&dxn2, &g30).unwrap(), eps).unwrap(),
    );
    diff(
        "silu_local f1",
        &sl1,
        &g_cpu::silu_with_grad(&f1).unwrap().1,
    );
    // Finite-difference arbiter on the composed function.
    println!("=== FD arbiter ===");
    let composed = |x: &Tensor,
                    wa: &Tensor,
                    wb: &Tensor,
                    wo: &Tensor,
                    wf1: &Tensor,
                    wf2: &Tensor,
                    g1: &Tensor,
                    g2: &Tensor,
                    g3: &Tensor,
                    g4: &Tensor|
     -> f32 {
        let r1 = rms_norm(x, eps).unwrap();
        let xn = mul(&r1, g1).unwrap();
        let a = sigmoid(&xn.linear(wa, None).unwrap()).unwrap();
        let bb = xn.linear(wb, None).unwrap();
        let h = gated_scan(&a, &bb).unwrap();
        let hg = mul(&h, g2).unwrap();
        let s2 = silu(&hg).unwrap();
        let out1 = add(x, &s2.linear(wo, None).unwrap()).unwrap();
        let r2 = rms_norm(&out1, eps).unwrap();
        let xn2 = mul(&r2, g3).unwrap();
        let f1 = xn2.linear(wf1, None).unwrap();
        let sf1 = silu(&f1).unwrap();
        let f2 = sf1.linear(wf2, None).unwrap();
        let y = add(&out1, &mul(&f2, g4).unwrap()).unwrap();
        mul(&y, &wt)
            .unwrap()
            .sum(None, false)
            .unwrap()
            .to_vec_f32()
            .unwrap()[0]
    };
    let hstep = 1e-3f32;
    let fd = |i: usize, base: &Vec<f32>, shape: &[usize], f: &dyn Fn(&Tensor) -> f32| -> f32 {
        let mut pp = base.clone();
        pp[i] += hstep;
        let mut mm = base.clone();
        mm[i] -= hstep;
        let fp = f(&Tensor::from_vec_f32(pp, shape).unwrap());
        let fm = f(&Tensor::from_vec_f32(mm, shape).unwrap());
        (fp - fm) / (2.0 * hstep)
    };
    let xv = x0.to_vec_f32().unwrap();
    let _f0 = composed(&x0, &wa0, &wb0, &wo0, &wf10, &wf20, &g10, &g20, &g30, &g40);
    for idx in [0usize, 1, 40] {
        let fdx = |x: &Tensor| composed(x, &wa0, &wb0, &wo0, &wf10, &wf20, &g10, &g20, &g30, &g40);
        let fdv = fd(idx, &xv, &[b, t, d], &fdx);
        println!(
            "  FD dL/dx[{idx}] = {fdv:.6}   fused gx[{idx}] = {:.6}   hand dx[{idx}] = {:.6}",
            gs[0].to_vec_f32().unwrap()[idx],
            dx.to_vec_f32().unwrap()[idx]
        );
    }
    let g2v = g20.to_vec_f32().unwrap();
    let fg2 = |g2: &Tensor| composed(&x0, &wa0, &wb0, &wo0, &wf10, &wf20, &g10, g2, &g30, &g40);
    let fd2 = fd(0, &g2v, &[d], &fg2);
    println!(
        "  FD dL/dg2[0] = {fd2:.6}   fused gg2[0] = {:.6}",
        gs[7].to_vec_f32().unwrap()[0]
    );
}
