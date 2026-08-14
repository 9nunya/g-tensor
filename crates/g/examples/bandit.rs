//! 5-arm Bernoulli bandit: softmax policy vs epsilon-greedy baseline.
use g::prelude::*;

fn main() -> Result<()> {
    let means = [0.1f32, 0.3, 0.8, 0.2, 0.4];
    let mut logits = g::zeros(&[1, 5], Dtype::F32)?.with_requires_grad();
    let opt = g::Sgd { lr: 0.2 };
    let mut rng = 12345u64;
    let mut nxt = || {
        rng ^= rng >> 12;
        rng ^= rng << 25;
        rng ^= rng >> 27;
        (rng as f32) / (u64::MAX as f32)
    };
    let mut avg = 0.0;
    for t in 0..200 {
        let p = softmax(&logits, -1)?.to_vec_f32()?;
        let mut c = 0.0;
        let u = nxt();
        let mut a = 4;
        for (i, pi) in p.iter().enumerate() {
            c += *pi;
            if u < c {
                a = i;
                break;
            }
        }
        let r = if nxt() < means[a] { 1.0 } else { 0.0 };
        avg = 0.95 * avg + 0.05 * r;
        g::zero_grad(&[&logits]);
        let lp = g::categorical_log_prob(&logits, &g::from_slice_i64(&[a as i64], &[1])?)?;
        let loss = g::neg(&lp.mul(&from_slice_f32(&[r - avg], &[])?)?)?;
        let gs = grad(&loss, &[&logits])?;
        logits = opt.step(&[&logits], &gs)?.remove(0).with_requires_grad();
        if t % 40 == 0 {
            println!("t {t} ema_reward {avg:.3} p {:?}", p);
        }
    }
    Ok(())
}
