//! Tiny 4x4 grid DQN. Goal at (3,3), pits at (1,1).
use g::prelude::*;

fn one_hot_pos(x: i32, y: i32) -> [f32; 16] {
    let mut s = [0.0f32; 16];
    s[(y * 4 + x) as usize] = 1.0;
    s
}

fn main() -> Result<()> {
    let mut w1 = g::randn_f32(&[16, 32], 5)?
        .mul(&from_slice_f32(&[0.25], &[])?)?
        .with_requires_grad();
    let mut b1 = g::zeros(&[32], Dtype::F32)?.with_requires_grad();
    let mut w2 = g::randn_f32(&[32, 4], 6)?
        .mul(&from_slice_f32(&[0.25], &[])?)?
        .with_requires_grad();
    let mut b2 = g::zeros(&[4], Dtype::F32)?.with_requires_grad();
    let opt = g::Sgd { lr: 0.05 };
    let mut rng = 99u64;
    let mut nxt = || {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        (rng >> 33) as f32 / (u32::MAX as f32)
    };
    type Transition = ([f32; 16], usize, f32, [f32; 16], bool);
    let mut replay: Vec<Transition> = Vec::new();
    let deltas = [(0, -1), (0, 1), (-1, 0), (1, 0)];
    let mut returns = 0.0;
    for ep in 0..60 {
        let (mut x, mut y) = (0, 0);
        let mut gsum = 0.0;
        for _t in 0..24 {
            let s = one_hot_pos(x, y);
            let q = from_slice_f32(&s, &[1, 16])?
                .linear(&w1, Some(&b1))?
                .relu()?
                .linear(&w2, Some(&b2))?;
            let qv = q.to_vec_f32()?;
            let a = if nxt() < 0.2 {
                (nxt() * 4.0) as usize % 4
            } else {
                qv.iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .unwrap()
                    .0
            };
            let (dx, dy) = deltas[a];
            let nx = (x + dx).clamp(0, 3);
            let ny = (y + dy).clamp(0, 3);
            let (r, done) = if nx == 1 && ny == 1 {
                (-1.0, true)
            } else if nx == 3 && ny == 3 {
                (1.0, true)
            } else {
                (-0.02, false)
            };
            replay.push((s, a, r, one_hot_pos(nx, ny), done));
            if replay.len() > 400 {
                replay.remove(0);
            }
            gsum += r;
            x = nx;
            y = ny;
            if replay.len() >= 32 {
                let mut xs = Vec::new();
                let mut tg = Vec::new();
                for k in 0..32 {
                    let idx = ((nxt() * replay.len() as f32) as usize) % replay.len();
                    let (s0, a0, r0, s1, d1) = replay[idx];
                    xs.extend_from_slice(&s0);
                    let q1 = from_slice_f32(&s1, &[1, 16])?
                        .linear(&w1.detach(), Some(&b1.detach()))?
                        .relu()?
                        .linear(&w2.detach(), Some(&b2.detach()))?
                        .to_vec_f32()?;
                    let maxq = q1.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let mut yhat = from_slice_f32(&s0, &[1, 16])?
                        .linear(&w1.detach(), Some(&b1.detach()))?
                        .relu()?
                        .linear(&w2.detach(), Some(&b2.detach()))?
                        .to_vec_f32()?;
                    yhat[a0] = r0 + if d1 { 0.0 } else { 0.95 * maxq };
                    tg.extend_from_slice(&yhat);
                    let _ = k;
                }
                g::zero_grad(&[&w1, &b1, &w2, &b2]);
                let pred = from_slice_f32(&xs, &[32, 16])?
                    .linear(&w1, Some(&b1))?
                    .relu()?
                    .linear(&w2, Some(&b2))?;
                let loss = pred.mse_loss(&from_slice_f32(&tg, &[32, 4])?, Reduce::Mean)?;
                let gs = grad(&loss, &[&w1, &b1, &w2, &b2])?;
                let u = opt.step(&[&w1, &b1, &w2, &b2], &gs)?;
                w1 = u[0].clone().with_requires_grad();
                b1 = u[1].clone().with_requires_grad();
                w2 = u[2].clone().with_requires_grad();
                b2 = u[3].clone().with_requires_grad();
            }
            if done {
                break;
            }
        }
        returns = 0.9 * returns + 0.1 * gsum;
        if ep % 10 == 0 {
            println!("ep {ep} return {gsum:.2} ema {returns:.2}");
        }
    }
    Ok(())
}
