//! Compare fused-backward intermediates against the AD graph's own
//! intermediates (computed by re-running the tail of the graph with each
//! intermediate as a leaf).
use g::prelude::*;
use g::{fused_block, gated_scan, mul, rms_norm, sigmoid, silu};
use g_core::{Dtype, Tensor};

fn diff(name: &str, a: &Tensor, b: &Tensor) {
    let (av, bv) = (a.to_vec_f32().unwrap(), b.to_vec_f32().unwrap());
    let mut m = 0f32;
    for (x, y) in av.iter().zip(bv.iter()) {
        m = m.max((x - y).abs() / (x.abs().max(y.abs()).max(1e-4)));
    }
    println!("{name:16} max rel {m:.2e}");
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

    // forward intermediates (AD version)
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

    // tail function from a given intermediate to the loss, treating the
    // intermediate as a requires_grad leaf.
    let tail_dout1 = |out1v: &Tensor| -> Tensor {
        let r2 = rms_norm(out1v, eps).unwrap();
        let xn2 = mul(&r2, &g30).unwrap();
        let f1 = xn2.linear(&wf10, None).unwrap();
        let sf1 = silu(&f1).unwrap();
        let f2 = sf1.linear(&wf20, None).unwrap();
        let y = add(out1v, &mul(&f2, &g40).unwrap()).unwrap();
        mul(&y, &wt).unwrap().sum(None, false).unwrap()
    };
    let out1l = out1.clone().with_requires_grad();
    let l = tail_dout1(&out1l);
    let ad_dout1 = grad(&l, &[&out1l]).unwrap()[0].clone();
    // hand-built dout1
    let dm = g_cpu::mul(&wt, &g40).unwrap();
    let gsf1 = g_cpu::matmul(&dm, &wf20.transpose().unwrap()).unwrap();
    let (_, sl1) = g_cpu::silu_with_grad(&f1).unwrap();
    let d_pre = g_cpu::mul(&gsf1, &sl1).unwrap();
    let dxn2 = g_cpu::matmul(&d_pre, &wf10.transpose().unwrap()).unwrap();
    let dout1_hand =
        g_cpu::rms_norm_backward(&out1, &g_cpu::mul(&dxn2, &g30).unwrap(), eps).unwrap();
    diff("dout1", &ad_dout1, &dout1_hand);

    // now the head part from x to out1, given gy=dout1
    let gy = ad_dout1;
    // AD versions of the head grads:
    let tail_hg = |hgv: &Tensor| -> Tensor {
        let s2 = silu(hgv).unwrap();
        let out1 = add(&x0, &s2.linear(&wo0, None).unwrap()).unwrap();
        tail_dout1(&out1)
    };
    let hgl = hg.clone().with_requires_grad();
    let l = tail_hg(&hgl);
    let ad_ghg = grad(&l, &[&hgl]).unwrap()[0].clone();
    let (_, slh) = g_cpu::silu_with_grad(&hg).unwrap();
    let gs2 = g_cpu::matmul(&gy, &wo0.transpose().unwrap()).unwrap();
    let ghg_hand = g_cpu::mul(&gs2, &slh).unwrap();
    diff("ghg (silu'(hg)*gs2)", &ad_ghg, &ghg_hand);

    // scan input grads via AD
    let tail_h = |hv: &Tensor| -> Tensor {
        let hg = mul(hv, &g20).unwrap();
        tail_hg(&hg)
    };
    let hl = h.clone().with_requires_grad();
    let l = tail_h(&hl);
    let ad_dh = grad(&l, &[&hl]).unwrap()[0].clone();
    let dh_hand = g_cpu::mul(&ghg_hand, &g20).unwrap();
    diff("dh", &ad_dh, &dh_hand);

    // ga, gb via AD
    let tail_a = |av: &Tensor| -> Tensor {
        let h = gated_scan(av, &bb).unwrap();
        tail_h(&h)
    };
    let al = a.clone().with_requires_grad();
    let l = tail_a(&al);
    let ad_ga = grad(&l, &[&al]).unwrap()[0].clone();
    let tail_bb = |bv: &Tensor| -> Tensor {
        let h = gated_scan(&a, bv).unwrap();
        tail_h(&h)
    };
    let bbl = bb.clone().with_requires_grad();
    let l = tail_bb(&bbl);
    let ad_gb = grad(&l, &[&bbl]).unwrap()[0].clone();
    let (ga_h, gb_h) = g_cpu::gated_scan_backward(&a, &h, &dh_hand).unwrap();
    diff("ga", &ad_ga, &ga_h);
    diff("gb", &ad_gb, &gb_h);

    // gua via AD
    let tail_ua = |uav: &Tensor| -> Tensor {
        let a = sigmoid(uav).unwrap();
        tail_a(&a)
    };
    // ua = xn@wa ; we need grad wrt ua
    let ua = xn.linear(&wa0, None).unwrap();
    let ual = ua.clone().with_requires_grad();
    let l = tail_ua(&ual);
    let ad_gua = grad(&l, &[&ual]).unwrap()[0].clone();
    let a1ma = g_cpu::mul(
        &a,
        &g_cpu::sub(&Tensor::ones(a.shape(), Dtype::F32).unwrap(), &a).unwrap(),
    )
    .unwrap();
    let gua_hand = g_cpu::mul(&ga_h, &a1ma).unwrap();
    diff("gua", &ad_gua, &gua_hand);

    // xn grad via AD (both from wa path and wb path)
    let tail_xn = |xnv: &Tensor| -> Tensor {
        let a = sigmoid(&xnv.linear(&wa0, None).unwrap()).unwrap();
        let bb = xnv.linear(&wb0, None).unwrap();
        let h = gated_scan(&a, &bb).unwrap();
        tail_h(&h)
    };
    let xnl = xn.clone().with_requires_grad();
    let l = tail_xn(&xnl);
    let ad_dxn = grad(&l, &[&xnl]).unwrap()[0].clone();
    let dxn_hand = g_cpu::add(
        &g_cpu::matmul(&gua_hand, &wa0.transpose().unwrap()).unwrap(),
        &g_cpu::matmul(&gb_h, &wb0.transpose().unwrap()).unwrap(),
    )
    .unwrap();
    diff("dxn", &ad_dxn, &dxn_hand);

    // r1 grad via AD
    let tail_r1 = |r1v: &Tensor| -> Tensor {
        let xn = mul(r1v, &g10).unwrap();
        tail_xn(&xn)
    };
    let r1l = r1.clone().with_requires_grad();
    let l = tail_r1(&r1l);
    let ad_dr1 = grad(&l, &[&r1l]).unwrap()[0].clone();
    let dr1_hand = g_cpu::mul(&dxn_hand, &g10).unwrap();
    diff("dr1", &ad_dr1, &dr1_hand);

    // finally dx via AD
    let xl = x0.clone().with_requires_grad();
    let l = {
        let r1 = rms_norm(&xl, eps).unwrap();
        let xn = mul(&r1, &g10).unwrap();
        let a = sigmoid(&xn.linear(&wa0, None).unwrap()).unwrap();
        let bb = xn.linear(&wb0, None).unwrap();
        let h = gated_scan(&a, &bb).unwrap();
        let hg = mul(&h, &g20).unwrap();
        let s2 = silu(&hg).unwrap();
        let out1 = add(&xl, &s2.linear(&wo0, None).unwrap()).unwrap();
        tail_dout1(&out1)
    };
    let ad_dx = grad(&l, &[&xl]).unwrap()[0].clone();
    let dx_hand = g_cpu::add(&gy, &g_cpu::rms_norm_backward(&x0, &dr1_hand, eps).unwrap()).unwrap();
    diff("dx", &ad_dx, &dx_hand);
    // also fused
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
    let gs = grad(&loss, &[&xr]).unwrap();
    diff("fused dx vs ad", &ad_dx, &gs[0]);
}
