//! Tiny MLP trained with MSE.
use g::prelude::*;

fn main() -> Result<()> {
    let x = from_slice_f32(&[0.2, -0.1, 0.4, 0.3, 0.1, -0.2], &[2, 3])?;
    let target = from_slice_f32(&[0.5, -0.25], &[2, 1])?;
    let mut w1 = g::randn_f32(&[3, 4], 1)?.with_requires_grad();
    let mut b1 = g::zeros(&[4], Dtype::F32)?.with_requires_grad();
    let mut w2 = g::randn_f32(&[4, 1], 2)?.with_requires_grad();
    let mut b2 = g::zeros(&[1], Dtype::F32)?.with_requires_grad();
    let opt = g::Sgd { lr: 0.05 };
    for step in 0..50 {
        g::zero_grad(&[&w1, &b1, &w2, &b2]);
        let y = x.linear(&w1, Some(&b1))?.relu()?.linear(&w2, Some(&b2))?;
        let loss = y.mse_loss(&target, Reduce::Mean)?;
        let gs = grad(&loss, &[&w1, &b1, &w2, &b2])?;
        let upd = opt.step(&[&w1, &b1, &w2, &b2], &gs)?;
        w1 = upd[0].clone().with_requires_grad();
        b1 = upd[1].clone().with_requires_grad();
        w2 = upd[2].clone().with_requires_grad();
        b2 = upd[3].clone().with_requires_grad();
        if step % 10 == 0 {
            println!("step {step} loss {}", loss.item_f32()?);
        }
    }
    Ok(())
}
