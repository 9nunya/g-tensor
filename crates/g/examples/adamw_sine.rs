//! Fit y = sin(x) with a 1-32-32-1 MLP and AdamW.
use g::prelude::*;
use std::f32::consts::PI;

fn main() -> Result<()> {
    let n = 64;
    let mut xs = Vec::with_capacity(n);
    let mut ys = Vec::with_capacity(n);
    for i in 0..n {
        let x = -PI + 2.0 * PI * (i as f32) / (n as f32);
        xs.push(x);
        ys.push(x.sin());
    }
    let x = from_slice_f32(&xs, &[n, 1])?;
    let y = from_slice_f32(&ys, &[n, 1])?;
    let mut w1 = g::randn_f32(&[1, 32], 1)?
        .mul(&from_slice_f32(&[0.4], &[])?)?
        .with_requires_grad();
    let mut b1 = g::zeros(&[32], Dtype::F32)?.with_requires_grad();
    let mut w2 = g::randn_f32(&[32, 32], 2)?
        .mul(&from_slice_f32(&[0.2], &[])?)?
        .with_requires_grad();
    let mut b2 = g::zeros(&[32], Dtype::F32)?.with_requires_grad();
    let mut w3 = g::randn_f32(&[32, 1], 3)?
        .mul(&from_slice_f32(&[0.2], &[])?)?
        .with_requires_grad();
    let mut b3 = g::zeros(&[1], Dtype::F32)?.with_requires_grad();
    let mut opt = g::AdamW::new(&[&w1, &b1, &w2, &b2, &w3, &b3], 0.01, 0.9, 0.999, 1e-8, 0.0)?;
    for step in 0..200 {
        g::zero_grad(&[&w1, &b1, &w2, &b2, &w3, &b3]);
        let h = x
            .linear(&w1, Some(&b1))?
            .tanh()?
            .linear(&w2, Some(&b2))?
            .tanh()?;
        let pred = h.linear(&w3, Some(&b3))?;
        let loss = pred.mse_loss(&y, Reduce::Mean)?;
        let gs = grad(&loss, &[&w1, &b1, &w2, &b2, &w3, &b3])?;
        let u = opt.step(&[&w1, &b1, &w2, &b2, &w3, &b3], &gs)?;
        w1 = u[0].clone().with_requires_grad();
        b1 = u[1].clone().with_requires_grad();
        w2 = u[2].clone().with_requires_grad();
        b2 = u[3].clone().with_requires_grad();
        w3 = u[4].clone().with_requires_grad();
        b3 = u[5].clone().with_requires_grad();
        if step % 40 == 0 {
            println!("step {step} mse {:.5}", loss.item_f32()?);
        }
    }
    Ok(())
}
