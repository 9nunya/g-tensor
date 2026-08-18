//! Finite-difference gradchecks for the fused recurrence and norm kernels.
use g::prelude::*;
use g_core::Tensor;

/// Central-difference gradient of `f` w.r.t. every element of `x`.
fn numeric_grad(x: &Tensor, f: &dyn Fn(&Tensor) -> f32) -> Vec<f32> {
    let base = x.to_vec_f32().unwrap();
    let h = 1e-3f32;
    let mut out = vec![0f32; base.len()];
    for i in 0..base.len() {
        let mut p = base.clone();
        p[i] += h;
        let mut m = base.clone();
        m[i] -= h;
        let fp = f(&Tensor::from_vec_f32(p, x.shape()).unwrap());
        let fm = f(&Tensor::from_vec_f32(m, x.shape()).unwrap());
        out[i] = (fp - fm) / (2.0 * h);
    }
    out
}

fn rel_err(a: &[f32], b: &[f32]) -> f32 {
    let mut worst = 0f32;
    for (x, y) in a.iter().zip(b) {
        let d = (x - y).abs() / (x.abs().max(y.abs()).max(1e-3));
        worst = worst.max(d);
    }
    worst
}

#[test]
fn gated_scan_gradients_match_finite_differences() {
    let (b, t, d) = (2usize, 5usize, 3usize);
    let n = b * t * d;
    // Gates in (0,1) via sigmoid of a raw tensor, inputs arbitrary.
    let araw: Vec<f32> = (0..n)
        .map(|i| ((i * 37 % 23) as f32 / 23.0) - 0.5)
        .collect();
    let bs: Vec<f32> = (0..n)
        .map(|i| ((i * 17 % 19) as f32 / 19.0) - 0.5)
        .collect();
    let a0 = Tensor::from_vec_f32(araw.clone(), &[b, t, d]).unwrap();
    let b0 = Tensor::from_vec_f32(bs.clone(), &[b, t, d]).unwrap();
    // weights make the scalar objective non-trivial
    let w: Vec<f32> = (0..n).map(|i| ((i * 13 % 7) as f32 / 7.0) + 0.2).collect();
    let wt = Tensor::from_vec_f32(w, &[b, t, d]).unwrap();

    let obj = |a: &Tensor, bb: &Tensor| -> f32 {
        let g = g::sigmoid(a).unwrap();
        let h = gated_scan(&g, bb).unwrap();
        g::mul(&h, &wt)
            .unwrap()
            .sum(None, false)
            .unwrap()
            .to_vec_f32()
            .unwrap()[0]
    };

    let ag = a0.clone().with_requires_grad();
    let bg = b0.clone().with_requires_grad();
    let gsig = g::sigmoid(&ag).unwrap();
    let h = gated_scan(&gsig, &bg).unwrap();
    let loss = g::mul(&h, &wt).unwrap().sum(None, false).unwrap();
    let gs = grad(&loss, &[&ag, &bg]).unwrap();

    let na = numeric_grad(&a0, &|a| obj(a, &b0));
    let nb = numeric_grad(&b0, &|bb| obj(&a0, bb));
    let ea = rel_err(&gs[0].to_vec_f32().unwrap(), &na);
    let eb = rel_err(&gs[1].to_vec_f32().unwrap(), &nb);
    println!("gated_scan rel err: gate={ea:.2e} input={eb:.2e}");
    assert!(ea < 2e-2, "gate grad mismatch {ea}");
    assert!(eb < 2e-2, "input grad mismatch {eb}");
}

#[test]
fn rms_norm_gradients_match_finite_differences() {
    let (r, d) = (3usize, 6usize);
    let v: Vec<f32> = (0..r * d)
        .map(|i| ((i * 29 % 17) as f32 / 17.0) - 0.4)
        .collect();
    let x0 = Tensor::from_vec_f32(v, &[r, d]).unwrap();
    let w: Vec<f32> = (0..r * d)
        .map(|i| ((i * 11 % 5) as f32 / 5.0) + 0.3)
        .collect();
    let wt = Tensor::from_vec_f32(w, &[r, d]).unwrap();
    let obj = |x: &Tensor| -> f32 {
        let y = rms_norm(x, 1e-5).unwrap();
        g::mul(&y, &wt)
            .unwrap()
            .sum(None, false)
            .unwrap()
            .to_vec_f32()
            .unwrap()[0]
    };
    let xg = x0.clone().with_requires_grad();
    let y = rms_norm(&xg, 1e-5).unwrap();
    let loss = g::mul(&y, &wt).unwrap().sum(None, false).unwrap();
    let gs = grad(&loss, &[&xg]).unwrap();
    let ng = numeric_grad(&x0, &obj);
    let e = rel_err(&gs[0].to_vec_f32().unwrap(), &ng);
    println!("rms_norm rel err: {e:.2e}");
    assert!(e < 2e-2, "rms_norm grad mismatch {e}");
}

#[test]
fn gated_scan_matches_explicit_recurrence() {
    let (b, t, d) = (2usize, 4usize, 3usize);
    let n = b * t * d;
    let av: Vec<f32> = (0..n).map(|i| 0.3 + (i % 5) as f32 * 0.1).collect();
    let bv: Vec<f32> = (0..n).map(|i| (i % 7) as f32 * 0.2 - 0.5).collect();
    let a = Tensor::from_vec_f32(av.clone(), &[b, t, d]).unwrap();
    let bb = Tensor::from_vec_f32(bv.clone(), &[b, t, d]).unwrap();
    let h = gated_scan(&a, &bb).unwrap().to_vec_f32().unwrap();
    for bi in 0..b {
        let mut state = vec![0f32; d];
        for ti in 0..t {
            for (j, sj) in state.iter_mut().enumerate().take(d) {
                let o = bi * t * d + ti * d + j;
                *sj = av[o] * *sj + bv[o];
                assert!((h[o] - *sj).abs() < 1e-6, "mismatch at {o}");
            }
        }
    }
}
