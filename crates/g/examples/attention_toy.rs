//! Two-token scaled-dot-product attention + residual MLP.
use g::prelude::*;

fn main() -> Result<()> {
    let x = from_slice_f32(&[0.2, -0.1, 0.4, 0.3, 0.0, -0.2, 0.1, 0.5], &[2, 4])?;
    let mut wq = g::randn_f32(&[4, 4], 1)?
        .mul(&from_slice_f32(&[0.3], &[])?)?
        .with_requires_grad();
    let mut wk = g::randn_f32(&[4, 4], 2)?
        .mul(&from_slice_f32(&[0.3], &[])?)?
        .with_requires_grad();
    let mut wv = g::randn_f32(&[4, 4], 3)?
        .mul(&from_slice_f32(&[0.3], &[])?)?
        .with_requires_grad();
    let mut wo = g::randn_f32(&[4, 4], 4)?
        .mul(&from_slice_f32(&[0.3], &[])?)?
        .with_requires_grad();
    let target = from_slice_f32(&[0.1, 0.2, -0.1, 0.0, 0.3, -0.2, 0.0, 0.1], &[2, 4])?;
    let opt = g::Sgd { lr: 0.05 };
    let scale = from_slice_f32(&[0.5], &[])?; // 1/sqrt(4)
    for step in 0..40 {
        g::zero_grad(&[&wq, &wk, &wv, &wo]);
        let q = x.linear(&wq, None)?;
        let k = x.linear(&wk, None)?;
        let v = x.linear(&wv, None)?;
        let kt = g::transpose(&k)?;
        let scores = g::matmul(&q, &kt)?.mul(&scale)?;
        let att = softmax(&scores, -1)?;
        let y = g::matmul(&att, &v)?.linear(&wo, None)?;
        let loss = y.mse_loss(&target, Reduce::Mean)?;
        let gs = grad(&loss, &[&wq, &wk, &wv, &wo])?;
        let u = opt.step(&[&wq, &wk, &wv, &wo], &gs)?;
        wq = u[0].clone().with_requires_grad();
        wk = u[1].clone().with_requires_grad();
        wv = u[2].clone().with_requires_grad();
        wo = u[3].clone().with_requires_grad();
        if step % 10 == 0 {
            println!("step {step} attn_mse {:.5}", loss.item_f32()?);
        }
    }
    Ok(())
}
