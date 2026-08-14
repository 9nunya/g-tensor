//! FD gradchecks for the fused sigmoid/silu forward+backward.
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
fn fused_sigmoid_and_silu_grads_match_fd() {
    let n = 48usize;
    let v: Vec<f32> = (0..n).map(|i| (i as f32 % 9.0) / 4.0 - 1.0).collect();
    let x0 = Tensor::from_vec_f32(v, &[n]).unwrap();
    let w: Vec<f32> = (0..n).map(|i| 0.3 + (i % 5) as f32 * 0.1).collect();
    let wt = Tensor::from_vec_f32(w, &[n]).unwrap();

    for (name, op) in [("sigmoid", g::sigmoid as fn(&Tensor) -> g::Result<Tensor>),
                       ("silu", g::silu as fn(&Tensor) -> g::Result<Tensor>)] {
        let obj = |x: &Tensor| -> f64 {
            let y = op(x).unwrap();
            g::mul(&y, &wt).unwrap().sum(None, false).unwrap().to_vec_f32().unwrap()[0] as f64
        };
        let xg = x0.clone().with_requires_grad();
        let y = op(&xg).unwrap();
        let loss = g::mul(&y, &wt).unwrap().sum(None, false).unwrap();
        let gs = grad(&loss, &[&xg]).unwrap();
        let h = 1e-3f32;
        let base = x0.to_vec_f32().unwrap();
        let mut ng = vec![0f32; n];
        for i in 0..n {
            let mut pp = base.clone(); pp[i] += h;
            let mut mm = base.clone(); mm[i] -= h;
            let fp = obj(&Tensor::from_vec_f32(pp, &[n]).unwrap());
            let fm = obj(&Tensor::from_vec_f32(mm, &[n]).unwrap());
            ng[i] = ((fp - fm) / (2.0 * h as f64)) as f32;
        }
        let e = rel_err(&gs[0].to_vec_f32().unwrap(), &ng);
        println!("{name} fused rel err: {e:.2e}");
        assert!(e < 2e-2, "{name} grad mismatch {e}");
    }
}
