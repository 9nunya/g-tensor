//! Gradchecks for the fused embedding and masked cross-entropy nodes.
use g::prelude::*;
use g_core::Tensor;

fn rel_err(a: &[f32], b: &[f32]) -> f32 {
    let mut worst = 0f32;
    for (x, y) in a.iter().zip(b) {
        let d = (x - y).abs() / (x.abs().max(y.abs()).max(1e-3));
        worst = worst.max(d);
    }
    worst
}

#[test]
fn embedding_fused_grad_matches_finite_differences() {
    let (v, d, n) = (8usize, 4usize, 6usize);
    let table0 = Tensor::from_vec_f32(
        (0..v * d).map(|i| (i as f32 % 5.0) - 2.0).collect(),
        &[v, d],
    )
    .unwrap();
    let idx = Tensor::from_vec_i64(vec![0, 3, 1, 7, 2, 5], &[n]).unwrap();
    let wt =
        Tensor::from_vec_f32((0..n * d).map(|i| 0.5 + (i % 3) as f32).collect(), &[n, d]).unwrap();

    let obj = |t: &Tensor| -> f32 {
        let y = g_ad::embedding_fused(t, &idx).unwrap();
        g::mul(&y, &wt)
            .unwrap()
            .sum(None, false)
            .unwrap()
            .to_vec_f32()
            .unwrap()[0]
    };
    let tg = table0.clone().with_requires_grad();
    let y = g_ad::embedding_fused(&tg, &idx).unwrap();
    let loss = g::mul(&y, &wt).unwrap().sum(None, false).unwrap();
    let gs = grad(&loss, &[&tg]).unwrap();

    let h = 1e-3f32;
    let base = table0.to_vec_f32().unwrap();
    let mut ng = vec![0f32; base.len()];
    for i in 0..base.len() {
        let mut pp = base.clone();
        pp[i] += h;
        let mut mm = base.clone();
        mm[i] -= h;
        let fp = obj(&Tensor::from_vec_f32(pp, table0.shape()).unwrap());
        let fm = obj(&Tensor::from_vec_f32(mm, table0.shape()).unwrap());
        ng[i] = (fp - fm) / (2.0 * h);
    }
    let e = rel_err(&gs[0].to_vec_f32().unwrap(), &ng);
    println!("embedding rel err: {e:.2e}");
    assert!(e < 2e-2, "embedding grad mismatch {e}");
}

#[test]
fn masked_ce_grad_matches_finite_differences() {
    let (n, v) = (5usize, 6usize);
    let logits0 = Tensor::from_vec_f32(
        (0..n * v).map(|i| (i as f32 % 7.0) - 3.0).collect(),
        &[n, v],
    )
    .unwrap();
    let targets = Tensor::from_vec_i64(vec![0, 2, 5, 1, 4], &[n]).unwrap();
    let mask = Tensor::from_vec_f32(vec![1.0, 0.0, 1.0, 1.0, 0.0], &[n]).unwrap();

    // f64 objective: f32 finite differences lose ~12% to cancellation on the
    // tiny gradients of near-zero softmax entries.
    let mk: Vec<f32> = mask.to_vec_f32().unwrap();
    let tgv: Vec<i64> = targets.to_vec_i64().unwrap();
    let count: f64 = mk.iter().map(|&m| m as f64).sum();
    let obj = |l: &Tensor| -> f64 {
        let lv = l.to_vec_f32().unwrap();
        let mut loss = 0f64;
        for i in 0..n {
            if mk[i] == 0.0 {
                continue;
            }
            let row = &lv[i * v..i * v + v];
            let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let ex: Vec<f64> = row.iter().map(|x| ((x - m) as f64).exp()).collect();
            let s: f64 = ex.iter().sum();
            loss -= (ex[tgv[i] as usize] / s).ln();
        }
        loss / count
    };
    let lg = logits0.clone().with_requires_grad();
    let loss = g_ad::masked_ce(&lg, &targets, &mask).unwrap();
    let gs = grad(&loss, &[&lg]).unwrap();

    let h = 1e-3f32;
    let base = logits0.to_vec_f32().unwrap();
    let mut ng = vec![0f32; base.len()];
    for i in 0..base.len() {
        let mut pp = base.clone();
        pp[i] += h;
        let mut mm = base.clone();
        mm[i] -= h;
        let fp = obj(&Tensor::from_vec_f32(pp, logits0.shape()).unwrap());
        let fm = obj(&Tensor::from_vec_f32(mm, logits0.shape()).unwrap());
        ng[i] = ((fp - fm) / (2.0 * h as f64)) as f32;
    }
    let e = rel_err(&gs[0].to_vec_f32().unwrap(), &ng);
    println!("masked_ce rel err: {e:.2e}");
    assert!(e < 2e-2, "masked_ce grad mismatch {e}");

    // Masked positions must receive exactly zero gradient.
    let gv = gs[0].to_vec_f32().unwrap();
    for row in 0..n {
        if mask.to_vec_f32().unwrap()[row] == 0.0 {
            for k in 0..v {
                assert_eq!(gv[row * v + k], 0.0, "masked row {row} got gradient");
            }
        }
    }
}

#[test]
fn masked_ce_signed_weights_match_finite_differences() {
    let (n, v) = (4usize, 5usize);
    let base: Vec<f32> = (0..n * v)
        .map(|i| ((i * 7 % 13) as f32 - 6.0) * 0.2)
        .collect();
    let logits0 = Tensor::from_vec_f32(base.clone(), &[n, v]).unwrap();
    let targets = Tensor::from_vec_i64(vec![0, 3, 1, 4], &[n]).unwrap();
    let weights_v = vec![1.0f32, -0.75, 0.0, 0.4];
    let weights = Tensor::from_vec_f32(weights_v.clone(), &[n]).unwrap();
    let denom: f64 = weights_v.iter().map(|w| w.abs() as f64).sum();
    let objective = |values: &[f32]| -> f64 {
        let mut loss = 0.0f64;
        for i in 0..n {
            let row = &values[i * v..(i + 1) * v];
            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exp: Vec<f64> = row.iter().map(|x| ((x - max) as f64).exp()).collect();
            let sum: f64 = exp.iter().sum();
            loss -= weights_v[i] as f64 * (exp[[0, 3, 1, 4][i]] / sum).ln();
        }
        loss / denom
    };

    let logits = logits0.with_requires_grad();
    let loss = g_ad::masked_ce(&logits, &targets, &weights).unwrap();
    let analytical = grad(&loss, &[&logits]).unwrap()[0].to_vec_f32().unwrap();
    let h = 1e-3f32;
    let mut numerical = vec![0.0f32; base.len()];
    for i in 0..base.len() {
        let mut plus = base.clone();
        plus[i] += h;
        let mut minus = base.clone();
        minus[i] -= h;
        numerical[i] = ((objective(&plus) - objective(&minus)) / (2.0 * h as f64)) as f32;
    }
    let e = rel_err(&analytical, &numerical);
    assert!(e < 2e-2, "signed masked_ce grad mismatch {e}");
    assert!(
        analytical[v..2 * v].iter().any(|x| x.abs() > 1e-4),
        "negative-weight row was silently skipped"
    );
    assert!(
        analytical[2 * v..3 * v].iter().all(|x| *x == 0.0),
        "zero-weight row received gradient"
    );
}
