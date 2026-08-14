//! Neural primitives and a small AdamW/SGD.

use std::sync::Arc;

use g_core::{numel, Backward, Dtype, Error, Result, Tensor};

/// Affine map `x @ w + b` with `w` shaped `[in, out]` (not PyTorch's `[out, in]`).
pub fn linear(x: &Tensor, w: &Tensor, b: Option<&Tensor>) -> Result<Tensor> {
    if w.rank() != 2 {
        return Err(Error::shape("linear", "w must be rank 2 [in, out]"));
    }
    if x.rank() == 0 {
        return Err(Error::shape("linear", "x rank >= 1"));
    }
    let inn = x.shape()[x.rank() - 1];
    if inn != w.shape()[0] {
        return Err(Error::shape(
            "linear",
            format!("x[..., {inn}] vs w[{}, {}]", w.shape()[0], w.shape()[1]),
        ));
    }
    let y = g_ad::matmul(x, w)?;
    if let Some(bias) = b {
        g_ad::add(&y, bias)
    } else {
        Ok(y)
    }
}

pub fn relu(x: &Tensor) -> Result<Tensor> {
    g_ad::relu(x)
}

pub fn tanh(x: &Tensor) -> Result<Tensor> {
    g_ad::tanh(x)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// How a loss reduces over the batch.
pub enum Reduce {
    Mean,
    Sum,
    None,
}

/// Mean squared error. `target` is detached by default.
pub fn mse_loss(pred: &Tensor, target: &Tensor, reduction: Reduce) -> Result<Tensor> {
    let t = g_ad::detach(target);
    let diff = g_ad::sub(pred, &t)?;
    let sq = g_ad::mul(&diff, &diff)?;
    match reduction {
        Reduce::None => Ok(sq),
        Reduce::Sum => g_ad::sum(&sq, None, false),
        Reduce::Mean => g_ad::mean(&sq, None, false),
    }
}

/// Softmax along `axis`. Compute is at least fp32.
pub fn softmax(x: &Tensor, axis: isize) -> Result<Tensor> {
    let axn = g_core::normalize_axis(axis, x.rank(), "softmax")?;
    if x.dtype() == Dtype::F32 && axn + 1 == x.rank() {
        let out = g_cpu::softmax_last(x)?;
        return Ok(track_softmax(x, out));
    }
    log_softmax(x, axis).and_then(|z| {
        // exp(log_softmax)
        match z.dtype() {
            Dtype::F32 => {
                let v: Vec<f32> = z.to_vec_f32()?.into_iter().map(|v| v.exp()).collect();
                let out = Tensor::from_slice_f32(&v, z.shape())?;
                Ok(track_softmax(x, out))
            }
            Dtype::F64 => {
                let v: Vec<f64> = z.to_vec_f64()?.into_iter().map(|v| v.exp()).collect();
                let out = Tensor::from_slice_f64(&v, z.shape())?;
                Ok(track_softmax(x, out))
            }
            Dtype::I64 => Err(Error::dtype("softmax", "float")),
        }
    })
}

struct SoftmaxBw {
    parents: Vec<Tensor>,
    y: Tensor,
    axis: usize,
}
impl Backward for SoftmaxBw {
    fn name(&self) -> &'static str {
        "softmax"
    }
    fn parents(&self) -> &[Tensor] {
        &self.parents
    }
    fn backward(&self, gy: &Tensor) -> Result<Vec<Tensor>> {
        // (diag(s) - s s^T) gy = s * (gy - sum(gy*s))
        let dot = g_cpu::mul(gy, &self.y)?;
        let s = g_cpu::sum(&dot, Some(&[self.axis as isize]), true)?;
        let centered = g_cpu::sub(gy, &s)?;
        Ok(vec![g_cpu::mul(&self.y, &centered)?])
    }
}

fn track_softmax(x: &Tensor, mut y: Tensor) -> Tensor {
    if x.requires_grad() {
        let axis = x.rank().saturating_sub(1);
        y.set_grad_fn(Arc::new(SoftmaxBw {
            parents: vec![x.clone()],
            y: y.detach(),
            axis,
        }));
    }
    y
}

/// Log-softmax along `axis` (stable `x - max - logsumexp`).
pub fn log_softmax(x: &Tensor, axis: isize) -> Result<Tensor> {
    let ax = g_core::normalize_axis(axis, x.rank(), "log_softmax")?;
    if x.shape()[ax] == 0 {
        return Err(Error::shape("log_softmax", "empty axis"));
    }
    if x.dtype() == Dtype::F32 && ax + 1 == x.rank() {
        let out = g_cpu::log_softmax_last(x)?;
        if x.requires_grad() {
            let mut y = out;
            let saved = y.detach();
            y.set_grad_fn(Arc::new(LogSoftmaxBw {
                parents: vec![x.clone()],
                log_y: saved,
                axis: ax,
            }));
            return Ok(y);
        }
        return Ok(out);
    }
    // x - max - log(sum(exp(x-max)))
    let maxv = reduce_max(x, ax)?;
    let shifted = g_cpu::sub(x, &maxv)?;
    let ex = match x.dtype() {
        Dtype::F32 => {
            let v: Vec<f32> = shifted.to_vec_f32()?.into_iter().map(|v| v.exp()).collect();
            Tensor::from_slice_f32(&v, x.shape())?
        }
        Dtype::F64 => {
            let v: Vec<f64> = shifted.to_vec_f64()?.into_iter().map(|v| v.exp()).collect();
            Tensor::from_slice_f64(&v, x.shape())?
        }
        Dtype::I64 => return Err(Error::dtype("log_softmax", "float")),
    };
    let se = g_cpu::sum(&ex, Some(&[ax as isize]), true)?;
    let lse = match se.dtype() {
        Dtype::F32 => {
            let v: Vec<f32> = se.to_vec_f32()?.into_iter().map(|v| v.ln()).collect();
            Tensor::from_slice_f32(&v, se.shape())?
        }
        Dtype::F64 => {
            let v: Vec<f64> = se.to_vec_f64()?.into_iter().map(|v| v.ln()).collect();
            Tensor::from_slice_f64(&v, se.shape())?
        }
        Dtype::I64 => unreachable!(),
    };
    let out = g_cpu::sub(&shifted, &lse)?;
    if x.requires_grad() {
        // log_softmax VJP: gy - softmax * sum(gy)
        let mut y = out;
        y.set_grad_fn(Arc::new(LogSoftmaxBw {
            parents: vec![x.clone()],
            log_y: y.detach(),
            axis: ax,
        }));
        Ok(y)
    } else {
        Ok(out)
    }
}

struct LogSoftmaxBw {
    parents: Vec<Tensor>,
    log_y: Tensor,
    axis: usize,
}
impl Backward for LogSoftmaxBw {
    fn name(&self) -> &'static str {
        "log_softmax"
    }
    fn parents(&self) -> &[Tensor] {
        &self.parents
    }
    fn backward(&self, gy: &Tensor) -> Result<Vec<Tensor>> {
        let s = g_cpu::exp(&self.log_y)?;
        let sum_gy = g_cpu::sum(gy, Some(&[self.axis as isize]), true)?;
        let corr = g_cpu::mul(&s, &sum_gy)?;
        Ok(vec![g_cpu::sub(gy, &corr)?])
    }
}

fn reduce_max(x: &Tensor, axis: usize) -> Result<Tensor> {
    let mut out_shape = x.shape().to_vec();
    out_shape[axis] = 1;
    match x.dtype() {
        Dtype::F32 => {
            let mut acc = vec![f32::NEG_INFINITY; numel(&out_shape)?];
            g_core::for_each_index(x.shape(), |idx| {
                let mut oidx = idx.to_vec();
                oidx[axis] = 0;
                let mut off = 0usize;
                let mut st = 1usize;
                for i in (0..out_shape.len()).rev() {
                    off += oidx[i] * st;
                    st *= out_shape[i];
                }
                let v = x.read_f32_at(idx).unwrap();
                if v > acc[off] {
                    acc[off] = v;
                }
            });
            Tensor::from_slice_f32(&acc, &out_shape)
        }
        Dtype::F64 => {
            let mut acc = vec![f64::NEG_INFINITY; numel(&out_shape)?];
            g_core::for_each_index(x.shape(), |idx| {
                let mut oidx = idx.to_vec();
                oidx[axis] = 0;
                let mut off = 0usize;
                let mut st = 1usize;
                for i in (0..out_shape.len()).rev() {
                    off += oidx[i] * st;
                    st *= out_shape[i];
                }
                let v = x.read_f64_at(idx).unwrap();
                if v > acc[off] {
                    acc[off] = v;
                }
            });
            Tensor::from_slice_f64(&acc, &out_shape)
        }
        Dtype::I64 => Err(Error::dtype("max", "float")),
    }
}

/// `θ ← θ − lr · g`.
pub struct Sgd {
    pub lr: f64,
}

impl Sgd {
    pub fn step(&self, params: &[&Tensor], grads: &[Tensor]) -> Result<Vec<Tensor>> {
        if params.len() != grads.len() {
            return Err(Error::shape("sgd", "params/grads length"));
        }
        let mut out = Vec::new();
        for (p, g) in params.iter().zip(grads) {
            let dec = g_cpu::mul_scalar(g, self.lr)?;
            out.push(g_cpu::sub(p, &dec)?);
        }
        Ok(out)
    }
}

/// Decoupled AdamW as in the PC dossier (record the equations).
/// Decoupled AdamW. Call [`AdamW::step`] with the same param order as [`AdamW::new`].
pub struct AdamW {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub weight_decay: f64,
    pub step: i64,
    m: Vec<Tensor>,
    v: Vec<Tensor>,
}

impl AdamW {
    pub fn new(
        params: &[&Tensor],
        lr: f64,
        beta1: f64,
        beta2: f64,
        eps: f64,
        wd: f64,
    ) -> Result<Self> {
        let mut m = Vec::new();
        let mut v = Vec::new();
        for p in params {
            m.push(Tensor::zeros(p.shape(), p.dtype())?);
            v.push(Tensor::zeros(p.shape(), p.dtype())?);
        }
        Ok(Self {
            lr,
            beta1,
            beta2,
            eps,
            weight_decay: wd,
            step: 0,
            m,
            v,
        })
    }

    pub fn step(&mut self, params: &[&Tensor], grads: &[Tensor]) -> Result<Vec<Tensor>> {
        self.step += 1;
        let t = self.step as f64;
        let mut out = Vec::new();
        for (i, (p, g)) in params.iter().zip(grads).enumerate() {
            self.m[i] = g_cpu::add(
                &g_cpu::mul_scalar(&self.m[i], self.beta1)?,
                &g_cpu::mul_scalar(g, 1.0 - self.beta1)?,
            )?;
            let g2 = g_cpu::mul(g, g)?;
            self.v[i] = g_cpu::add(
                &g_cpu::mul_scalar(&self.v[i], self.beta2)?,
                &g_cpu::mul_scalar(&g2, 1.0 - self.beta2)?,
            )?;
            let mh = g_cpu::mul_scalar(&self.m[i], 1.0 / (1.0 - self.beta1.powf(t)))?;
            let vh = g_cpu::mul_scalar(&self.v[i], 1.0 / (1.0 - self.beta2.powf(t)))?;
            let denom = match p.dtype() {
                Dtype::F32 => {
                    let v: Vec<f32> = vh
                        .to_vec_f32()?
                        .into_iter()
                        .map(|x| x.sqrt() + self.eps as f32)
                        .collect();
                    Tensor::from_slice_f32(&v, p.shape())?
                }
                Dtype::F64 => {
                    let v: Vec<f64> = vh
                        .to_vec_f64()?
                        .into_iter()
                        .map(|x| x.sqrt() + self.eps)
                        .collect();
                    Tensor::from_slice_f64(&v, p.shape())?
                }
                Dtype::I64 => return Err(Error::dtype("adamw", "float")),
            };
            let upd = match p.dtype() {
                Dtype::F32 => {
                    let num = mh.to_vec_f32()?;
                    let den = denom.to_vec_f32()?;
                    let v: Vec<f32> = num.iter().zip(den.iter()).map(|(n, d)| n / d).collect();
                    Tensor::from_slice_f32(&v, p.shape())?
                }
                Dtype::F64 => {
                    let num = mh.to_vec_f64()?;
                    let den = denom.to_vec_f64()?;
                    let v: Vec<f64> = num.iter().zip(den.iter()).map(|(n, d)| n / d).collect();
                    Tensor::from_slice_f64(&v, p.shape())?
                }
                Dtype::I64 => unreachable!(),
            };
            let decayed = g_cpu::mul_scalar(p, 1.0 - self.lr * self.weight_decay)?;
            out.push(g_cpu::sub(&decayed, &g_cpu::mul_scalar(&upd, self.lr)?)?);
        }
        Ok(out)
    }
}

/// Negative log-likelihood of class indices given log-probs `[B, C]`.
pub fn nll_loss(log_probs: &Tensor, targets: &Tensor, reduction: Reduce) -> Result<Tensor> {
    if targets.dtype() != Dtype::I64 {
        return Err(Error::dtype(
            "nll_loss",
            "targets must be i64 class indices",
        ));
    }
    if log_probs.rank() != 2 || targets.rank() != 1 {
        return Err(Error::shape("nll_loss", "expected [B,C] and [B]"));
    }
    if log_probs.shape()[0] != targets.shape()[0] {
        return Err(Error::shape("nll_loss", "batch mismatch"));
    }
    let b = log_probs.shape()[0];
    let c = log_probs.shape()[1] as i64;
    let gathered = match log_probs.dtype() {
        Dtype::F32 => {
            let lp = log_probs.to_vec_f32()?;
            let tg = targets.to_vec_i64()?;
            let width = log_probs.shape()[1];
            let mut v = Vec::with_capacity(b);
            for i in 0..b {
                let cls = tg[i];
                if cls < 0 || cls >= c {
                    return Err(Error::index("nll_loss", "class oob"));
                }
                v.push(-lp[i * width + cls as usize]);
            }
            Tensor::from_vec_f32(v, &[b])?
        }
        Dtype::F64 => {
            let mut v = Vec::with_capacity(b);
            for i in 0..b {
                let cls = targets.read_i64_at(&[i])?;
                if cls < 0 || cls >= c {
                    return Err(Error::index("nll_loss", "class oob"));
                }
                v.push(-log_probs.read_f64_at(&[i, cls as usize])?);
            }
            Tensor::from_slice_f64(&v, &[b])?
        }
        Dtype::I64 => return Err(Error::dtype("nll_loss", "float logits")),
    };
    // Attach a custom backward through log_probs only.
    if log_probs.requires_grad() {
        let mut y = gathered;
        y.set_grad_fn(Arc::new(NllBw {
            parents: vec![log_probs.clone()],
            targets: targets.detach(),
            batch: b,
            classes: c as usize,
        }));
        match reduction {
            Reduce::None => Ok(y),
            Reduce::Sum => g_ad::sum(&y, None, false),
            Reduce::Mean => g_ad::mean(&y, None, false),
        }
    } else {
        match reduction {
            Reduce::None => Ok(gathered),
            Reduce::Sum => g_ad::sum(&gathered, None, false),
            Reduce::Mean => g_ad::mean(&gathered, None, false),
        }
    }
}

struct NllBw {
    parents: Vec<Tensor>,
    targets: Tensor,
    batch: usize,
    classes: usize,
}
impl Backward for NllBw {
    fn name(&self) -> &'static str {
        "nll_loss"
    }
    fn parents(&self) -> &[Tensor] {
        &self.parents
    }
    fn backward(&self, gy: &Tensor) -> Result<Vec<Tensor>> {
        // d(-log p[class])/d logp = -one_hot
        match self.parents[0].dtype() {
            Dtype::F32 => {
                let mut g = vec![0.0f32; self.batch * self.classes];
                let scale = if gy.numel() == 1 { gy.item_f32()? } else { 1.0 };
                for i in 0..self.batch {
                    let cls = self.targets.read_i64_at(&[i])? as usize;
                    let s = if gy.numel() == 1 {
                        scale
                    } else {
                        gy.read_f32_at(&[i])?
                    };
                    g[i * self.classes + cls] = -s;
                }
                Tensor::from_slice_f32(&g, &[self.batch, self.classes]).map(|t| vec![t])
            }
            Dtype::F64 => {
                let mut g = vec![0.0f64; self.batch * self.classes];
                for i in 0..self.batch {
                    let cls = self.targets.read_i64_at(&[i])? as usize;
                    let s = if gy.numel() == 1 {
                        gy.item_f32()? as f64
                    } else {
                        gy.read_f64_at(&[i])?
                    };
                    g[i * self.classes + cls] = -s;
                }
                Tensor::from_slice_f64(&g, &[self.batch, self.classes]).map(|t| vec![t])
            }
            Dtype::I64 => Err(Error::dtype("nll_loss", "float")),
        }
    }
}

/// `nll_loss(log_softmax(logits), targets)`.
pub fn cross_entropy(logits: &Tensor, targets: &Tensor, reduction: Reduce) -> Result<Tensor> {
    let lp = log_softmax(logits, -1)?;
    nll_loss(&lp, targets, reduction)
}

/// Last-axis layer norm without learned scale/bias.
pub fn layer_norm(x: &Tensor, eps: f64) -> Result<Tensor> {
    if x.rank() == 0 {
        return Err(Error::shape("layer_norm", "rank >= 1"));
    }
    let ax = (x.rank() - 1) as isize;
    let mean = g_ad::mean(x, Some(&[ax]), true)?;
    let xc = g_ad::sub(x, &mean)?;
    let var = g_ad::mean(&g_ad::mul(&xc, &xc)?, Some(&[ax]), true)?;
    let denom = match x.dtype() {
        Dtype::F32 => {
            let v: Vec<f32> = var
                .to_vec_f32()?
                .into_iter()
                .map(|t| (t + eps as f32).sqrt())
                .collect();
            Tensor::from_slice_f32(&v, var.shape())?
        }
        Dtype::F64 => {
            let v: Vec<f64> = var
                .to_vec_f64()?
                .into_iter()
                .map(|t| (t + eps).sqrt())
                .collect();
            Tensor::from_slice_f64(&v, var.shape())?
        }
        Dtype::I64 => return Err(Error::dtype("layer_norm", "float")),
    };
    g_ad::div(&xc, &denom)
}

/// Row lookup: `weight[indices]` → `[..., dim]`.
pub fn embedding(weight: &Tensor, indices: &Tensor) -> Result<Tensor> {
    if weight.rank() != 2 {
        return Err(Error::shape("embedding", "weight [num, dim]"));
    }
    if indices.dtype() != Dtype::I64 {
        return Err(Error::dtype("embedding", "indices i64"));
    }
    let y = embed_forward(weight, indices)?;
    if !weight.requires_grad() {
        return Ok(y);
    }
    struct EmbBw {
        parents: Vec<Tensor>,
        indices: Tensor,
        w_shape: Vec<usize>,
    }
    impl Backward for EmbBw {
        fn name(&self) -> &'static str {
            "embedding"
        }
        fn parents(&self) -> &[Tensor] {
            &self.parents
        }
        fn backward(&self, gy: &Tensor) -> Result<Vec<Tensor>> {
            let dim = self.w_shape[1];
            let mut g = vec![0.0f32; self.w_shape[0] * dim];
            let flat = self.indices.reshape(&[-1])?;
            let n = flat.numel();
            let gyf = gy.reshape(&[-1, dim as isize])?;
            for i in 0..n {
                let row = flat.read_i64_at(&[i])? as usize;
                for j in 0..dim {
                    g[row * dim + j] += gyf.read_f32_at(&[i, j])?;
                }
            }
            Tensor::from_slice_f32(&g, &self.w_shape).map(|t| vec![t])
        }
    }
    let mut y = y;
    y.set_grad_fn(Arc::new(EmbBw {
        parents: vec![weight.clone()],
        indices: indices.detach(),
        w_shape: weight.shape().to_vec(),
    }));
    Ok(y)
}

fn embed_forward(weight: &Tensor, indices: &Tensor) -> Result<Tensor> {
    let flat = indices.reshape(&[-1])?;
    let n = flat.numel();
    let dim = weight.shape()[1];
    let mut rows = Vec::with_capacity(n * dim);
    let n_rows = weight.shape()[0] as i64;
    for i in 0..n {
        let idx = flat.read_i64_at(&[i])?;
        if idx < 0 || idx >= n_rows {
            return Err(Error::index("embedding", "index oob"));
        }
        for j in 0..dim {
            match weight.dtype() {
                Dtype::F32 => rows.push(weight.read_f32_at(&[idx as usize, j])?),
                _ => return Err(Error::dtype("embedding", "f32 weight in v1 helper")),
            }
        }
    }
    let mut out_shape = indices.shape().to_vec();
    out_shape.push(dim);
    Tensor::from_slice_f32(&rows, &out_shape)
}

/// One-hot of `i64` indices with last dim `depth`.
pub fn one_hot(indices: &Tensor, depth: usize) -> Result<Tensor> {
    if indices.dtype() != Dtype::I64 {
        return Err(Error::dtype("one_hot", "i64 indices"));
    }
    let n = indices.numel();
    let flat = indices.reshape(&[-1])?;
    let mut v = vec![0.0f32; n * depth];
    for i in 0..n {
        let k = flat.read_i64_at(&[i])?;
        if k < 0 || k as usize >= depth {
            return Err(Error::index("one_hot", "class oob"));
        }
        v[i * depth + k as usize] = 1.0;
    }
    let mut shape = indices.shape().to_vec();
    shape.push(depth);
    Tensor::from_slice_f32(&v, &shape)
}

/// `log π(a | logits)` for a batch of class indices.
pub fn categorical_log_prob(logits: &Tensor, actions: &Tensor) -> Result<Tensor> {
    let lp = log_softmax(logits, -1)?;
    let g = g_ad::take(&lp, 1, actions)?;
    g.reshape(&[-1])
}

/// Entropy of `softmax(logits)` per row.
pub fn categorical_entropy(logits: &Tensor) -> Result<Tensor> {
    let p = softmax(logits, -1)?;
    let lp = log_softmax(logits, -1)?;
    g_ad::neg(&g_ad::sum(&g_ad::mul(&p, &lp)?, Some(&[-1]), false)?)
}

/// Subtract mean and divide by std (+ `1e-8`).
pub fn whiten(x: &Tensor) -> Result<Tensor> {
    let m = g_ad::mean(x, None, false)?;
    let s = g_ad::stddev(x, None, false)?;
    let eps = match x.dtype() {
        Dtype::F32 => Tensor::scalar_f32(1e-8)?,
        Dtype::F64 => Tensor::scalar_f64(1e-8)?,
        Dtype::I64 => return Err(Error::dtype("whiten", "float")),
    };
    g_ad::div(&g_ad::sub(x, &m)?, &g_ad::add(&s, &eps)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn categorical_log_prob_grad() {
        let logits = Tensor::from_slice_f32(&[0.1, 0.2, 0.3, 0.0, -0.1, 0.4], &[2, 3])
            .unwrap()
            .with_requires_grad();
        let a = Tensor::from_slice_i64(&[1, 2], &[2]).unwrap();
        let lp = categorical_log_prob(&logits, &a).unwrap();
        assert_eq!(lp.shape(), &[2]);
        let loss = g_ad::mean(&lp, None, false).unwrap();
        let g = g_ad::grad(&loss, &[&logits]).unwrap();
        assert_eq!(g[0].shape(), &[2, 3]);
    }

    #[test]
    fn linear_shape() {
        let x = Tensor::from_slice_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let w = Tensor::from_slice_f32(&[1.0, 0.0, 0.0, 1.0], &[2, 2]).unwrap();
        let y = linear(&x, &w, None).unwrap();
        assert_eq!(y.shape(), &[2, 2]);
        assert_eq!(y.to_vec_f32().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
    }
}
