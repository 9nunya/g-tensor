//! Undercomplete autoencoder on 8-D one-hot-ish vectors.
use g::prelude::*;

fn main() -> Result<()> {
    let n = 8;
    let mut data = vec![0.0f32; n * n];
    for i in 0..n {
        data[i * n + i] = 1.0;
    }
    let x = from_slice_f32(&data, &[n, n])?;
    let mut e = g::randn_f32(&[n, 3], 3)?
        .mul(&from_slice_f32(&[0.3], &[])?)?
        .with_requires_grad();
    let mut d = g::randn_f32(&[3, n], 4)?
        .mul(&from_slice_f32(&[0.3], &[])?)?
        .with_requires_grad();
    let opt = g::Sgd { lr: 0.25 };
    for step in 0..120 {
        g::zero_grad(&[&e, &d]);
        let z = x.linear(&e, None)?.tanh()?;
        let rec = z.linear(&d, None)?.sigmoid()?;
        let loss = rec.mse_loss(&x, Reduce::Mean)?;
        let gs = grad(&loss, &[&e, &d])?;
        let u = opt.step(&[&e, &d], &gs)?;
        e = u[0].clone().with_requires_grad();
        d = u[1].clone().with_requires_grad();
        if step % 30 == 0 {
            println!("step {step} recon {:.5}", loss.item_f32()?);
        }
    }
    Ok(())
}
