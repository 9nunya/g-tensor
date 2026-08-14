//! T8: tiny ordinary training step (W5 analogue).

use g::{from_slice_f32, grad, linear, mse_loss, relu, zero_grad, Reduce, Result, Sgd, Tensor};

fn mlp(x: &Tensor, w1: &Tensor, b1: &Tensor, w2: &Tensor, b2: &Tensor) -> Result<Tensor> {
    let h = relu(&linear(x, w1, Some(b1))?)?;
    linear(&h, w2, Some(b2))
}

#[test]
fn one_and_many_steps_descend() -> Result<()> {
    let x = from_slice_f32(&[0.1, -0.2, 0.3, 0.4, -0.5, 0.6], &[2, 3])?;
    let y = from_slice_f32(&[0.2, -0.1, 0.0, 0.5], &[2, 2])?;
    let mut w1 = from_slice_f32(&[0.2, -0.1, 0.05, 0.1, 0.0, -0.2], &[3, 2])?.with_requires_grad();
    let mut b1 = from_slice_f32(&[0.0, 0.0], &[2])?.with_requires_grad();
    let mut w2 = from_slice_f32(&[0.3, -0.2, 0.1, 0.25], &[2, 2])?.with_requires_grad();
    let mut b2 = from_slice_f32(&[0.0, 0.0], &[2])?.with_requires_grad();

    let pred0 = mlp(&x, &w1, &b1, &w2, &b2)?;
    let loss0 = mse_loss(&pred0, &y, Reduce::Mean)?.item_f32()?;
    assert!(loss0.is_finite());

    let sgd = Sgd { lr: 0.05 };
    let mut last = loss0;
    for _ in 0..25 {
        zero_grad(&[&w1, &b1, &w2, &b2]);
        let pred = mlp(&x, &w1, &b1, &w2, &b2)?;
        let loss = mse_loss(&pred, &y, Reduce::Mean)?;
        let gs = grad(&loss, &[&w1, &b1, &w2, &b2])?;
        let updated = sgd.step(&[&w1, &b1, &w2, &b2], &gs)?;
        w1 = updated[0].clone().with_requires_grad();
        b1 = updated[1].clone().with_requires_grad();
        w2 = updated[2].clone().with_requires_grad();
        b2 = updated[3].clone().with_requires_grad();
        last = loss.item_f32()?;
    }
    assert!(last < loss0, "expected descent: start {loss0} end {last}");
    Ok(())
}
