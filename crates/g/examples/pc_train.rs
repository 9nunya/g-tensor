//! Local predictive-coding classifier: infer hidden state, then Hebbian-like update.
use g::prelude::*;

fn main() -> Result<()> {
    let x = from_slice_f32(&[0.9, 0.1, 0.05, 0.1, 0.85, 0.1, 0.05, 0.1, 0.9], &[3, 3])?;
    let y = from_slice_f32(&[1.0, 0.0, 0.0, 1.0, 0.0, 1.0], &[3, 2])?;
    let mut w1 = g::randn_f32(&[3, 6], 1)?
        .mul(&from_slice_f32(&[0.3], &[])?)?
        .with_requires_grad();
    let mut w2 = g::randn_f32(&[6, 2], 2)?
        .mul(&from_slice_f32(&[0.3], &[])?)?
        .with_requires_grad();
    let opt = g::Sgd { lr: 0.1 };
    for step in 0..40 {
        let mut h = g::zeros(&[3, 6], Dtype::F32)?;
        for _k in 0..8 {
            let pred_h = x.linear(&w1.detach(), None)?.tanh()?;
            let pred_y = h.linear(&w2.detach(), None)?.tanh()?;
            let eh = h.sub(&pred_h)?;
            let ey = y.sub(&pred_y)?;
            // local: descend prediction error, no tape through K
            h = h.sub(&eh.mul(&from_slice_f32(&[0.3], &[])?)?)?;
            let _ = ey;
        }
        let hs = h.stop_gradient()?;
        g::zero_grad(&[&w1, &w2]);
        let e1 = hs.sub(&x.linear(&w1, None)?.tanh()?)?;
        let e2 = y.sub(&hs.linear(&w2, None)?.tanh()?)?;
        let energy = e1
            .mul(&e1)?
            .mean(None, false)?
            .add(&e2.mul(&e2)?.mean(None, false)?)?;
        let gs = grad(&energy, &[&w1, &w2])?;
        let u = opt.step(&[&w1, &w2], &gs)?;
        w1 = u[0].clone().with_requires_grad();
        w2 = u[1].clone().with_requires_grad();
        if step % 10 == 0 {
            println!("step {step} energy {:.4}", energy.item_f32()?);
        }
    }
    Ok(())
}
