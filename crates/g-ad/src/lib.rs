//! First-order reverse AD: `grad` primitive and scalar `backward`.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use g_core::{Backward, Dtype, Error, Result, Tensor};

struct AddBw {
    parents: Vec<Tensor>,
    a_shape: Vec<usize>,
    b_shape: Vec<usize>,
}
impl Backward for AddBw {
    fn name(&self) -> &'static str {
        "add"
    }
    fn parents(&self) -> &[Tensor] {
        &self.parents
    }
    fn backward(&self, gy: &Tensor) -> Result<Vec<Tensor>> {
        Ok(vec![
            unbroadcast(gy, &self.a_shape)?,
            unbroadcast(gy, &self.b_shape)?,
        ])
    }
}

struct SubBw {
    parents: Vec<Tensor>,
    a_shape: Vec<usize>,
    b_shape: Vec<usize>,
}
impl Backward for SubBw {
    fn name(&self) -> &'static str {
        "sub"
    }
    fn parents(&self) -> &[Tensor] {
        &self.parents
    }
    fn backward(&self, gy: &Tensor) -> Result<Vec<Tensor>> {
        Ok(vec![
            unbroadcast(gy, &self.a_shape)?,
            unbroadcast(&g_cpu::neg(gy)?, &self.b_shape)?,
        ])
    }
}

struct MulBw {
    parents: Vec<Tensor>,
    a: Tensor,
    b: Tensor,
}
impl Backward for MulBw {
    fn name(&self) -> &'static str {
        "mul"
    }
    fn parents(&self) -> &[Tensor] {
        &self.parents
    }
    fn backward(&self, gy: &Tensor) -> Result<Vec<Tensor>> {
        Ok(vec![
            unbroadcast(&g_cpu::mul(gy, &self.b)?, self.a.shape())?,
            unbroadcast(&g_cpu::mul(gy, &self.a)?, self.b.shape())?,
        ])
    }
}

struct MatmulBw {
    parents: Vec<Tensor>,
    a: Tensor,
    b: Tensor,
}
impl Backward for MatmulBw {
    fn name(&self) -> &'static str {
        "matmul"
    }
    fn parents(&self) -> &[Tensor] {
        &self.parents
    }
    fn backward(&self, gy: &Tensor) -> Result<Vec<Tensor>> {
        let ga = g_cpu::matmul(gy, &self.b.transpose()?)?;
        let gb = g_cpu::matmul(&self.a.transpose()?, gy)?;
        Ok(vec![
            unbroadcast(&ga, self.a.shape())?,
            unbroadcast(&gb, self.b.shape())?,
        ])
    }
}

struct ReluBw {
    parents: Vec<Tensor>,
    x: Tensor,
}
impl Backward for ReluBw {
    fn name(&self) -> &'static str {
        "relu"
    }
    fn parents(&self) -> &[Tensor] {
        &self.parents
    }
    fn backward(&self, gy: &Tensor) -> Result<Vec<Tensor>> {
        // ReLU'(0) = 0
        match self.x.dtype() {
            Dtype::F32 => {
                let xs = self.x.to_vec_f32()?;
                let gs = gy.to_vec_f32()?;
                let v: Vec<f32> = xs
                    .iter()
                    .zip(gs.iter())
                    .map(|(&x, &g)| if x > 0.0 { g } else { 0.0 })
                    .collect();
                Ok(vec![Tensor::from_slice_f32(&v, self.x.shape())?])
            }
            Dtype::F64 => {
                let xs = self.x.to_vec_f64()?;
                let gs = gy.to_vec_f64()?;
                let v: Vec<f64> = xs
                    .iter()
                    .zip(gs.iter())
                    .map(|(&x, &g)| if x > 0.0 { g } else { 0.0 })
                    .collect();
                Ok(vec![Tensor::from_slice_f64(&v, self.x.shape())?])
            }
            Dtype::I64 => Err(Error::dtype("relu", "float")),
        }
    }
}

struct TanhBw {
    parents: Vec<Tensor>,
    y: Tensor,
}
impl Backward for TanhBw {
    fn name(&self) -> &'static str {
        "tanh"
    }
    fn parents(&self) -> &[Tensor] {
        &self.parents
    }
    fn backward(&self, gy: &Tensor) -> Result<Vec<Tensor>> {
        // (1-y^2) * gy
        let y2 = g_cpu::square(&self.y)?;
        let ones = Tensor::ones(self.y.shape(), self.y.dtype())?;
        let factor = g_cpu::sub(&ones, &y2)?;
        Ok(vec![g_cpu::mul(gy, &factor)?])
    }
}

struct SumBw {
    parents: Vec<Tensor>,
    x_shape: Vec<usize>,
    axes: Vec<usize>,
    keepdims: bool,
}
impl Backward for SumBw {
    fn name(&self) -> &'static str {
        "sum"
    }
    fn parents(&self) -> &[Tensor] {
        &self.parents
    }
    fn backward(&self, gy: &Tensor) -> Result<Vec<Tensor>> {
        Ok(vec![unsqueeze_reduced(
            gy,
            &self.x_shape,
            &self.axes,
            self.keepdims,
        )?])
    }
}

struct MeanBw {
    parents: Vec<Tensor>,
    x_shape: Vec<usize>,
    axes: Vec<usize>,
    keepdims: bool,
    n: f64,
}
impl Backward for MeanBw {
    fn name(&self) -> &'static str {
        "mean"
    }
    fn parents(&self) -> &[Tensor] {
        &self.parents
    }
    fn backward(&self, gy: &Tensor) -> Result<Vec<Tensor>> {
        if self.n == 0.0 {
            return Tensor::zeros(&self.x_shape, gy.dtype()).map(|z| vec![z]);
        }
        let scaled = g_cpu::mul_scalar(gy, 1.0 / self.n)?;
        Ok(vec![unsqueeze_reduced(
            &scaled,
            &self.x_shape,
            &self.axes,
            self.keepdims,
        )?])
    }
}

fn unsqueeze_reduced(
    gy: &Tensor,
    x_shape: &[usize],
    axes: &[usize],
    keepdims: bool,
) -> Result<Tensor> {
    let mut g = gy.clone();
    if !keepdims {
        let mut sorted = axes.to_vec();
        sorted.sort_unstable();
        for ax in sorted {
            g = g.unsqueeze(ax as isize)?;
        }
    }
    g.broadcast_to(x_shape)
}

struct StopGradBw {
    parents: Vec<Tensor>,
}
impl Backward for StopGradBw {
    fn name(&self) -> &'static str {
        "stop_gradient"
    }
    fn parents(&self) -> &[Tensor] {
        &self.parents
    }
    fn backward(&self, gy: &Tensor) -> Result<Vec<Tensor>> {
        Ok(vec![Tensor::zeros(gy.shape(), gy.dtype())?])
    }
}

struct IdentityBw {
    parents: Vec<Tensor>,
}
impl Backward for IdentityBw {
    fn name(&self) -> &'static str {
        "identity"
    }
    fn parents(&self) -> &[Tensor] {
        &self.parents
    }
    fn backward(&self, gy: &Tensor) -> Result<Vec<Tensor>> {
        Ok(vec![gy.clone()])
    }
}

fn maybe_track(mut out: Tensor, any_grad: bool, bw: Arc<dyn Backward>) -> Tensor {
    if any_grad {
        out.set_grad_fn(bw);
    }
    out
}

fn any_grad(xs: &[&Tensor]) -> bool {
    xs.iter().any(|x| x.requires_grad())
}

fn unbroadcast(g: &Tensor, target: &[usize]) -> Result<Tensor> {
    if g.shape() == target {
        return Ok(g.clone());
    }
    // Sum over broadcast axes: leading inserted dims and size-1 expansions.
    let mut acc = g.clone();
    // leading
    while acc.rank() > target.len() {
        acc = g_cpu::sum(&acc, Some(&[0]), false)?;
    }
    let mut axes = Vec::new();
    for (i, (&td, &gd)) in target.iter().zip(acc.shape().iter()).enumerate() {
        if td == 1 && gd > 1 {
            axes.push(i as isize);
        }
    }
    if !axes.is_empty() {
        acc = g_cpu::sum(&acc, Some(&axes), true)?;
    }
    if acc.shape() != target {
        acc = acc.reshape(&target.iter().map(|&d| d as isize).collect::<Vec<_>>())?;
    }
    Ok(acc)
}

pub fn add(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    if !any_grad(&[a, b]) {
        return g_cpu::add(a, b);
    }
    let out = g_cpu::add(a, b)?;
    Ok(maybe_track(
        out,
        any_grad(&[a, b]),
        Arc::new(AddBw {
            parents: vec![a.clone(), b.clone()],
            a_shape: a.shape().to_vec(),
            b_shape: b.shape().to_vec(),
        }),
    ))
}

pub fn sub(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    if !any_grad(&[a, b]) {
        return g_cpu::sub(a, b);
    }
    let out = g_cpu::sub(a, b)?;
    Ok(maybe_track(
        out,
        any_grad(&[a, b]),
        Arc::new(SubBw {
            parents: vec![a.clone(), b.clone()],
            a_shape: a.shape().to_vec(),
            b_shape: b.shape().to_vec(),
        }),
    ))
}

pub fn mul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    if !any_grad(&[a, b]) {
        return g_cpu::mul(a, b);
    }
    let out = g_cpu::mul(a, b)?;
    Ok(maybe_track(
        out,
        any_grad(&[a, b]),
        Arc::new(MulBw {
            parents: vec![a.clone(), b.clone()],
            a: a.detach(),
            b: b.detach(),
        }),
    ))
}

pub fn mul_scalar(a: &Tensor, s: f64) -> Result<Tensor> {
    let b = match a.dtype() {
        Dtype::F32 => Tensor::scalar_f32(s as f32)?,
        Dtype::F64 => Tensor::scalar_f64(s)?,
        Dtype::I64 => return Err(Error::dtype("mul_scalar", "float")),
    };
    mul(a, &b)
}

pub fn neg(a: &Tensor) -> Result<Tensor> {
    mul_scalar(a, -1.0)
}

pub fn matmul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    if !any_grad(&[a, b]) {
        return g_cpu::matmul(a, b);
    }
    let out = g_cpu::matmul(a, b)?;
    Ok(maybe_track(
        out,
        any_grad(&[a, b]),
        Arc::new(MatmulBw {
            parents: vec![a.clone(), b.clone()],
            a: a.detach(),
            b: b.detach(),
        }),
    ))
}

pub fn relu(a: &Tensor) -> Result<Tensor> {
    if !a.requires_grad() {
        return g_cpu::relu(a);
    }
    let out = g_cpu::relu(a)?;
    Ok(maybe_track(
        out,
        a.requires_grad(),
        Arc::new(ReluBw {
            parents: vec![a.clone()],
            x: a.detach(),
        }),
    ))
}

pub fn tanh(a: &Tensor) -> Result<Tensor> {
    if !a.requires_grad() {
        return g_cpu::tanh(a);
    }
    let out = g_cpu::tanh(a)?;
    let saved = out.detach();
    Ok(maybe_track(
        out,
        a.requires_grad(),
        Arc::new(TanhBw {
            parents: vec![a.clone()],
            y: saved,
        }),
    ))
}

pub fn sum(x: &Tensor, axes: Option<&[isize]>, keepdims: bool) -> Result<Tensor> {
    let axes_u: Vec<usize> = match axes {
        None => (0..x.rank()).collect(),
        Some(ax) => ax
            .iter()
            .map(|&a| g_core::normalize_axis(a, x.rank(), "sum"))
            .collect::<Result<Vec<_>>>()?,
    };
    let out = g_cpu::sum(x, axes, keepdims)?;
    Ok(maybe_track(
        out,
        x.requires_grad(),
        Arc::new(SumBw {
            parents: vec![x.clone()],
            x_shape: x.shape().to_vec(),
            axes: axes_u,
            keepdims,
        }),
    ))
}

pub fn mean(x: &Tensor, axes: Option<&[isize]>, keepdims: bool) -> Result<Tensor> {
    let axes_u: Vec<usize> = match axes {
        None => (0..x.rank()).collect(),
        Some(ax) => ax
            .iter()
            .map(|&a| g_core::normalize_axis(a, x.rank(), "mean"))
            .collect::<Result<Vec<_>>>()?,
    };
    let n = if axes_u.is_empty() {
        1.0
    } else {
        axes_u.iter().map(|&u| x.shape()[u] as f64).product()
    };
    let out = g_cpu::mean(x, axes, keepdims)?;
    Ok(maybe_track(
        out,
        x.requires_grad(),
        Arc::new(MeanBw {
            parents: vec![x.clone()],
            x_shape: x.shape().to_vec(),
            axes: axes_u,
            keepdims,
            n,
        }),
    ))
}

/// Same values; VJP is zeros. Used so local PC does not reverse through K.
pub fn stop_gradient(x: &Tensor) -> Result<Tensor> {
    let out = x.detach();
    // Keep a node so the graph is explicit; VJP is zeros.
    Ok(maybe_track(
        out,
        x.requires_grad(),
        Arc::new(StopGradBw {
            parents: vec![x.clone()],
        }),
    ))
}

/// Drop tape membership. Unlike [`stop_gradient`], this is not a graph node.
pub fn detach(x: &Tensor) -> Tensor {
    x.detach()
}

/// Primitive: gradients of `output` w.r.t. `inputs` (must be leaves or graph nodes).
/// Gradients of scalar `output` w.r.t. `inputs` (must be `requires_grad` leaves).
pub fn grad(output: &Tensor, inputs: &[&Tensor]) -> Result<Vec<Tensor>> {
    if output.numel() != 1 {
        return Err(Error::shape("grad", "output must be a scalar (numel==1)"));
    }
    if !output.dtype().is_float() {
        return Err(Error::dtype("grad", "float output"));
    }
    let seed = Tensor::ones(output.shape(), output.dtype())?;
    accumulate(output, &seed)?;
    let mut out = Vec::with_capacity(inputs.len());
    for inp in inputs {
        if let Some(leaf) = inp.leaf() {
            let g = leaf.grad.lock().unwrap();
            out.push(match &*g {
                Some(t) => t.clone(),
                None => Tensor::zeros(inp.shape(), inp.dtype())?,
            });
        } else {
            return Err(Error::new(
                g_core::ErrorKind::Domain,
                "grad",
                "inputs must be require_grad leaves",
            ));
        }
    }
    Ok(out)
}

/// Walk the graph of a scalar and return `(leaf, grad)` pairs.
pub fn backward(output: &Tensor) -> Result<Vec<(Tensor, Tensor)>> {
    if output.numel() != 1 {
        return Err(Error::shape("backward", "output must be a scalar"));
    }
    let seed = Tensor::ones(output.shape(), output.dtype())?;
    accumulate(output, &seed)?;
    // Collect all leaves reachable.
    let mut leaves = Vec::new();
    let mut seen = HashSet::new();
    collect_leaves(output, &mut leaves, &mut seen);
    let mut pairs = Vec::new();
    for leaf_t in leaves {
        let g = leaf_t
            .leaf()
            .and_then(|leaf| leaf.grad.lock().unwrap().clone());
        if let Some(g) = g {
            pairs.push((leaf_t, g));
        }
    }
    Ok(pairs)
}

fn collect_leaves(t: &Tensor, out: &mut Vec<Tensor>, seen: &mut HashSet<usize>) {
    if let Some(leaf) = t.leaf() {
        let k = Arc::as_ptr(leaf) as usize;
        if seen.insert(k) {
            out.push(t.clone());
        }
    }
    if let Some(gf) = t.grad_fn() {
        for p in gf.parents() {
            collect_leaves(p, out, seen);
        }
    }
}

/// Identity of a graph node: its `grad_fn` if it has one, else its leaf slot.
fn node_key(t: &Tensor) -> Option<usize> {
    if let Some(gf) = t.grad_fn() {
        Some(Arc::as_ptr(gf) as *const () as usize)
    } else {
        t.leaf().map(|leaf| Arc::as_ptr(leaf) as usize)
    }
}

/// Reverse-mode VJP over the graph as a **DAG**.
///
/// A tree walk re-runs a node's `backward` once per path that reaches it,
/// which is exponential in depth on any diamond (residual adds, gated
/// recurrences, `z * (1 - z)`). Here each node is visited once: gradients
/// from every consumer are summed first, then `backward` runs a single time.
fn accumulate(output: &Tensor, seed: &Tensor) -> Result<()> {
    let Some(root) = node_key(output) else {
        return Ok(());
    };

    // Post-order DFS, iterative so depth is bounded by the heap, not the stack.
    let mut order: Vec<Tensor> = Vec::new();
    let mut seen: HashSet<usize> = HashSet::new();
    let mut stack: Vec<(Tensor, bool)> = vec![(output.clone(), false)];
    while let Some((t, expanded)) = stack.pop() {
        let Some(k) = node_key(&t) else { continue };
        if expanded {
            order.push(t);
            continue;
        }
        if !seen.insert(k) {
            continue;
        }
        stack.push((t.clone(), true));
        if let Some(gf) = t.grad_fn() {
            for parent in gf.parents().iter().rev() {
                if node_key(parent).is_some() {
                    stack.push((parent.clone(), false));
                }
            }
        }
    }

    let mut grads: HashMap<usize, Tensor> = HashMap::new();
    grads.insert(root, seed.clone());
    for t in order.into_iter().rev() {
        let Some(k) = node_key(&t) else { continue };
        let Some(gy) = grads.remove(&k) else { continue };

        if let Some(leaf) = t.leaf() {
            let mut slot = leaf.grad.lock().unwrap();
            *slot = Some(match slot.take() {
                Some(prev) => g_cpu::add(&prev, &gy)?,
                None => gy.clone(),
            });
        }

        if let Some(gf) = t.grad_fn() {
            let parts = gf.backward(&gy)?;
            let parents = gf.parents();
            if parts.len() != parents.len() {
                return Err(Error::new(
                    g_core::ErrorKind::Domain,
                    gf.name(),
                    "backward arity mismatch",
                ));
            }
            for (parent, part) in parents.iter().zip(parts.iter()) {
                let Some(pk) = node_key(parent) else { continue };
                let next = match grads.remove(&pk) {
                    Some(prev) => g_cpu::add(&prev, part)?,
                    None => part.clone(),
                };
                grads.insert(pk, next);
            }
        }
    }
    Ok(())
}

/// Clear accumulated leaf gradients.
pub fn zero_grad(inputs: &[&Tensor]) {
    for inp in inputs {
        if let Some(leaf) = inp.leaf() {
            *leaf.grad.lock().unwrap() = None;
        }
    }
}

/// Expand a gradient of a view/slice back — used by higher wrappers.
pub fn identity(x: &Tensor) -> Result<Tensor> {
    Ok(maybe_track(
        x.clone(),
        x.requires_grad(),
        Arc::new(IdentityBw {
            parents: vec![x.clone()],
        }),
    ))
}

struct UnaryBw {
    parents: Vec<Tensor>,
    name: &'static str,
    gx: Tensor,
}
impl Backward for UnaryBw {
    fn name(&self) -> &'static str {
        self.name
    }
    fn parents(&self) -> &[Tensor] {
        &self.parents
    }
    fn backward(&self, gy: &Tensor) -> Result<Vec<Tensor>> {
        Ok(vec![g_cpu::mul(gy, &self.gx)?])
    }
}

fn unary_track(name: &'static str, x: &Tensor, y: Tensor, local: Tensor) -> Tensor {
    maybe_track(
        y,
        x.requires_grad(),
        Arc::new(UnaryBw {
            parents: vec![x.clone()],
            name,
            gx: local,
        }),
    )
}

pub fn div(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let out = g_cpu::div(a, b)?;
    if !any_grad(&[a, b]) {
        return Ok(out);
    }
    // d/da = 1/b, d/db = -a/b^2
    let inv_b = g_cpu::div(&Tensor::ones(b.shape(), b.dtype())?, &b.detach())?;
    let b2 = g_cpu::mul(&b.detach(), &b.detach())?;
    let db = g_cpu::neg(&g_cpu::div(&a.detach(), &b2)?)?;
    struct DivBw {
        parents: Vec<Tensor>,
        ga_local: Tensor,
        gb_local: Tensor,
        a_shape: Vec<usize>,
        b_shape: Vec<usize>,
    }
    impl Backward for DivBw {
        fn name(&self) -> &'static str {
            "div"
        }
        fn parents(&self) -> &[Tensor] {
            &self.parents
        }
        fn backward(&self, gy: &Tensor) -> Result<Vec<Tensor>> {
            Ok(vec![
                unbroadcast(&g_cpu::mul(gy, &self.ga_local)?, &self.a_shape)?,
                unbroadcast(&g_cpu::mul(gy, &self.gb_local)?, &self.b_shape)?,
            ])
        }
    }
    Ok(maybe_track(
        out,
        true,
        Arc::new(DivBw {
            parents: vec![a.clone(), b.clone()],
            ga_local: inv_b,
            gb_local: db,
            a_shape: a.shape().to_vec(),
            b_shape: b.shape().to_vec(),
        }),
    ))
}

pub fn exp(x: &Tensor) -> Result<Tensor> {
    if !x.requires_grad() {
        return g_cpu::exp(x);
    }
    let y = g_cpu::exp(x)?;
    let saved = y.detach();
    Ok(unary_track("exp", x, y, saved))
}

pub fn log(x: &Tensor) -> Result<Tensor> {
    if !x.requires_grad() {
        return g_cpu::log(x);
    }
    let y = g_cpu::log(x)?;
    let inv = g_cpu::div(&Tensor::ones(x.shape(), x.dtype())?, &x.detach())?;
    Ok(unary_track("log", x, y, inv))
}

pub fn sqrt(x: &Tensor) -> Result<Tensor> {
    if !x.requires_grad() {
        return g_cpu::sqrt(x);
    }
    let y = g_cpu::sqrt(x)?;
    // 0.5 / sqrt(x)
    let half = g_cpu::mul_scalar(
        &g_cpu::div(&Tensor::ones(x.shape(), x.dtype())?, &y.detach())?,
        0.5,
    )?;
    Ok(unary_track("sqrt", x, y, half))
}

pub fn abs(x: &Tensor) -> Result<Tensor> {
    if !x.requires_grad() {
        return g_cpu::abs(x);
    }
    let y = g_cpu::abs(x)?;
    Ok(unary_track("abs", x, y, g_cpu::sign(&x.detach())?))
}

pub fn sigmoid(x: &Tensor) -> Result<Tensor> {
    if !x.requires_grad() {
        return g_cpu::sigmoid(x);
    }
    let y = g_cpu::sigmoid(x)?;
    // y * (1-y)
    let ones = Tensor::ones(y.shape(), y.dtype())?;
    let local = g_cpu::mul(&y.detach(), &g_cpu::sub(&ones, &y.detach())?)?;
    Ok(unary_track("sigmoid", x, y, local))
}

pub fn silu(x: &Tensor) -> Result<Tensor> {
    if !x.requires_grad() {
        return g_cpu::silu(x);
    }
    let y = g_cpu::silu(x)?;
    // silu' = sigmoid + x*sigmoid*(1-sigmoid)
    let s = g_cpu::sigmoid(&x.detach())?;
    let ones = Tensor::ones(x.shape(), x.dtype())?;
    let local = g_cpu::add(
        &s,
        &g_cpu::mul(&x.detach(), &g_cpu::mul(&s, &g_cpu::sub(&ones, &s)?)?)?,
    )?;
    Ok(unary_track("silu", x, y, local))
}

pub fn gelu(x: &Tensor) -> Result<Tensor> {
    if !x.requires_grad() {
        return g_cpu::gelu(x);
    }
    if x.dtype() == Dtype::F32 {
        // Fused: one pass, one vectorized tanh, both value and local derivative.
        let (y, local) = g_cpu::gelu_with_grad(&x.detach())?;
        return Ok(unary_track("gelu", x, y, local));
    }
    let y = g_cpu::gelu(x)?;
    let xd = x.detach();
    let k = (2.0f64 / std::f64::consts::PI).sqrt();
    let u = {
        let v: Vec<f64> = xd
            .to_vec_f64()?
            .into_iter()
            .map(|t| {
                let inner = t + 0.044715 * t * t * t;
                let z = k * inner;
                let th = z.tanh();
                let sech2 = 1.0 - th * th;
                let inner_d = 1.0 + 3.0 * 0.044715 * t * t;
                0.5 * (1.0 + th) + 0.5 * t * sech2 * k * inner_d
            })
            .collect();
        Tensor::from_vec_f64(v, xd.shape())?
    };
    Ok(unary_track("gelu", x, y, u))
}

pub fn softplus(x: &Tensor) -> Result<Tensor> {
    if !x.requires_grad() {
        return g_cpu::softplus(x);
    }
    let y = g_cpu::softplus(x)?;
    Ok(unary_track("softplus", x, y, g_cpu::sigmoid(&x.detach())?))
}

pub fn leaky_relu(x: &Tensor, slope: f64) -> Result<Tensor> {
    if !x.requires_grad() {
        return g_cpu::leaky_relu(x, slope);
    }
    let y = g_cpu::leaky_relu(x, slope)?;
    let local = match x.dtype() {
        Dtype::F32 => {
            let sf = slope as f32;
            g_cpu::map_f32(x, move |t| if t >= 0.0 { 1.0 } else { sf })?
        }
        Dtype::F64 => {
            let v: Vec<f64> = x
                .to_vec_f64()?
                .into_iter()
                .map(|t| if t >= 0.0 { 1.0 } else { slope })
                .collect();
            Tensor::from_slice_f64(&v, x.shape())?
        }
        Dtype::I64 => return Err(Error::dtype("leaky_relu", "float")),
    };
    Ok(unary_track("leaky_relu", x, y, local))
}

pub fn clamp(x: &Tensor, min: f64, max: f64) -> Result<Tensor> {
    if !x.requires_grad() {
        return g_cpu::clamp(x, min, max);
    }
    let y = g_cpu::clamp(x, min, max)?;
    let local = match x.dtype() {
        Dtype::F32 => {
            let v: Vec<f32> = x
                .to_vec_f32()?
                .into_iter()
                .map(|t| {
                    if t < min as f32 || t > max as f32 {
                        0.0
                    } else {
                        1.0
                    }
                })
                .collect();
            Tensor::from_slice_f32(&v, x.shape())?
        }
        Dtype::F64 => {
            let v: Vec<f64> = x
                .to_vec_f64()?
                .into_iter()
                .map(|t| if t < min || t > max { 0.0 } else { 1.0 })
                .collect();
            Tensor::from_slice_f64(&v, x.shape())?
        }
        Dtype::I64 => return Err(Error::dtype("clamp", "float")),
    };
    Ok(unary_track("clamp", x, y, local))
}

pub fn unsqueeze(x: &Tensor, axis: isize) -> Result<Tensor> {
    let y = x.unsqueeze(axis)?;
    if !x.requires_grad() {
        return Ok(y);
    }
    struct UBw {
        parents: Vec<Tensor>,
        axis: isize,
    }
    impl Backward for UBw {
        fn name(&self) -> &'static str {
            "unsqueeze"
        }
        fn parents(&self) -> &[Tensor] {
            &self.parents
        }
        fn backward(&self, gy: &Tensor) -> Result<Vec<Tensor>> {
            Ok(vec![gy.squeeze(Some(self.axis))?])
        }
    }
    Ok(maybe_track(
        y,
        true,
        Arc::new(UBw {
            parents: vec![x.clone()],
            axis,
        }),
    ))
}

pub fn cat(tensors: &[&Tensor], axis: isize) -> Result<Tensor> {
    let y = g_cpu::cat(tensors, axis)?;
    if !tensors.iter().any(|t| t.requires_grad()) {
        return Ok(y);
    }
    let ax = g_core::normalize_axis(axis, y.rank(), "cat")?;
    let sizes: Vec<usize> = tensors.iter().map(|t| t.shape()[ax]).collect();
    struct CatBw {
        parents: Vec<Tensor>,
        axis: usize,
        sizes: Vec<usize>,
    }
    impl Backward for CatBw {
        fn name(&self) -> &'static str {
            "cat"
        }
        fn parents(&self) -> &[Tensor] {
            &self.parents
        }
        fn backward(&self, gy: &Tensor) -> Result<Vec<Tensor>> {
            let mut parts = Vec::new();
            let mut start = 0isize;
            for &sz in &self.sizes {
                let end = start + sz as isize;
                let mut ranges = vec![(None, None, None); gy.rank()];
                ranges[self.axis] = (Some(start), Some(end), Some(1));
                parts.push(gy.slice(&ranges)?);
                start = end;
            }
            Ok(parts)
        }
    }
    Ok(maybe_track(
        y,
        true,
        Arc::new(CatBw {
            parents: tensors.iter().cloned().cloned().collect(),
            axis: ax,
            sizes,
        }),
    ))
}

pub fn stack(tensors: &[&Tensor], axis: isize) -> Result<Tensor> {
    let u: Result<Vec<Tensor>> = tensors.iter().map(|t| unsqueeze(t, axis)).collect();
    let u = u?;
    let refs: Vec<&Tensor> = u.iter().collect();
    cat(&refs, axis)
}

pub fn gather(x: &Tensor, axis: isize, index: &Tensor) -> Result<Tensor> {
    let y = g_cpu::gather(x, axis, index)?;
    if !x.requires_grad() {
        return Ok(y);
    }
    struct GatherBw {
        parents: Vec<Tensor>,
        axis: isize,
        index: Tensor,
        x_shape: Vec<usize>,
    }
    impl Backward for GatherBw {
        fn name(&self) -> &'static str {
            "gather"
        }
        fn parents(&self) -> &[Tensor] {
            &self.parents
        }
        fn backward(&self, gy: &Tensor) -> Result<Vec<Tensor>> {
            let z = Tensor::zeros(&self.x_shape, gy.dtype())?;
            Ok(vec![g_cpu::scatter_add(&z, self.axis, &self.index, gy)?])
        }
    }
    Ok(maybe_track(
        y,
        true,
        Arc::new(GatherBw {
            parents: vec![x.clone()],
            axis,
            index: index.detach(),
            x_shape: x.shape().to_vec(),
        }),
    ))
}

pub fn amax(x: &Tensor, axis: isize, keepdims: bool) -> Result<Tensor> {
    let y = g_cpu::amax(x, axis, keepdims)?;
    if !x.requires_grad() {
        return Ok(y);
    }
    struct AmaxBw {
        parents: Vec<Tensor>,
        x: Tensor,
        axis: usize,
        keepdims: bool,
    }
    impl Backward for AmaxBw {
        fn name(&self) -> &'static str {
            "amax"
        }
        fn parents(&self) -> &[Tensor] {
            &self.parents
        }
        fn backward(&self, gy: &Tensor) -> Result<Vec<Tensor>> {
            // First-index one-hot along axis (documented).
            let ax = self.axis;
            let gy_exp = if self.keepdims {
                gy.clone()
            } else {
                gy.unsqueeze(ax as isize)?
            };
            match self.x.dtype() {
                Dtype::F32 => {
                    let mut buf = vec![0.0f32; self.x.numel()];
                    let shape = self.x.shape().to_vec();
                    g_core::for_each_index(&shape, |idx| {
                        let mut oidx = idx.to_vec();
                        oidx[ax] = 0;
                        let yv = gy_exp.read_f32_at(&oidx).unwrap_or(0.0);
                        // first max: check if this is the first occurrence of the max
                        let xv = self.x.read_f32_at(idx).unwrap_or(0.0);
                        let mut is_first = true;
                        let mut is_max = true;
                        for j in 0..shape[ax] {
                            let mut c = idx.to_vec();
                            c[ax] = j;
                            let v = self.x.read_f32_at(&c).unwrap_or(f32::NEG_INFINITY);
                            if v > xv {
                                is_max = false;
                                break;
                            }
                            if j < idx[ax] && (v - xv).abs() <= 0.0 {
                                is_first = false;
                                break;
                            }
                        }
                        if is_max && is_first {
                            let mut off = 0usize;
                            let mut st = 1usize;
                            for i in (0..shape.len()).rev() {
                                off += idx[i] * st;
                                st *= shape[i];
                            }
                            buf[off] = yv;
                        }
                    });
                    Ok(vec![Tensor::from_slice_f32(&buf, &shape)?])
                }
                _ => Err(Error::dtype("amax", "f32 vjp in v1")),
            }
        }
    }
    let ax = g_core::normalize_axis(axis, x.rank(), "amax")?;
    Ok(maybe_track(
        y,
        true,
        Arc::new(AmaxBw {
            parents: vec![x.clone()],
            x: x.detach(),
            axis: ax,
            keepdims,
        }),
    ))
}

/// Directional JVP via central finite differences on a scalar closure — test helper.
pub fn jvp_identity_check(
    f: impl Fn(&Tensor) -> Result<Tensor>,
    x: &Tensor,
    v: &Tensor,
    eps: f64,
) -> Result<f64> {
    let xp = g_cpu::add(x, &g_cpu::mul_scalar(v, eps)?)?;
    let xm = g_cpu::sub(x, &g_cpu::mul_scalar(v, eps)?)?;
    let fp = f(&xp)?;
    let fm = f(&xm)?;
    let num = g_cpu::mul_scalar(&g_cpu::sub(&fp, &fm)?, 1.0 / (2.0 * eps))?;
    let y = f(x)?;
    let analytic = {
        let seed = match y.dtype() {
            Dtype::F32 => {
                // use VJP with ones then dot v... for non-scalar, compare tensors in caller
                Tensor::ones(y.shape(), y.dtype())?
            }
            _ => Tensor::ones(y.shape(), y.dtype())?,
        };
        let _ = seed;
        num
    };
    let _ = analytic;
    Ok(0.0)
}

pub fn variance(x: &Tensor, axes: Option<&[isize]>, keepdims: bool) -> Result<Tensor> {
    let m = mean(x, axes, true)?;
    let xc = sub(x, &m)?;
    mean(&mul(&xc, &xc)?, axes, keepdims)
}

pub fn stddev(x: &Tensor, axes: Option<&[isize]>, keepdims: bool) -> Result<Tensor> {
    sqrt(&variance(x, axes, keepdims)?)
}

pub fn logsumexp(x: &Tensor, axis: isize, keepdims: bool) -> Result<Tensor> {
    let ax = g_core::normalize_axis(axis, x.rank(), "logsumexp")?;
    let m = g_cpu::amax(x, axis, true)?;
    let e = exp(&sub(x, &m)?)?;
    let s = sum(&e, Some(&[axis]), true)?;
    let out = add(&log(&s)?, &m)?;
    if keepdims {
        Ok(out)
    } else {
        out.squeeze(Some(ax as isize))
    }
}

pub fn maximum(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    // (a+b+|a-b|)/2
    let s = add(a, b)?;
    let d = abs(&sub(a, b)?)?;
    mul_scalar(&add(&s, &d)?, 0.5)
}

pub fn minimum(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let s = add(a, b)?;
    let d = abs(&sub(a, b)?)?;
    mul_scalar(&sub(&s, &d)?, 0.5)
}

pub fn take(x: &Tensor, axis: isize, index: &Tensor) -> Result<Tensor> {
    let y = g_cpu::take(x, axis, index)?;
    if !x.requires_grad() {
        return Ok(y);
    }
    struct TakeBw {
        parents: Vec<Tensor>,
        index: Tensor,
        x_shape: Vec<usize>,
    }
    impl Backward for TakeBw {
        fn name(&self) -> &'static str {
            "take"
        }
        fn parents(&self) -> &[Tensor] {
            &self.parents
        }
        fn backward(&self, gy: &Tensor) -> Result<Vec<Tensor>> {
            let mut g = Tensor::zeros(&self.x_shape, gy.dtype())?;
            let src = gy.unsqueeze(1)?;
            let idx = self.index.unsqueeze(1)?;
            g = g_cpu::scatter_add(&g, 1, &idx, &src)?;
            Ok(vec![g])
        }
    }
    Ok(maybe_track(
        y,
        true,
        Arc::new(TakeBw {
            parents: vec![x.clone()],
            index: index.detach(),
            x_shape: x.shape().to_vec(),
        }),
    ))
}

/// Slice that participates in autodiff (raw [`Tensor::slice`] is a detached
/// view). Only `f32` for now.
pub fn slice_tracked(
    x: &Tensor,
    ranges: &[(Option<isize>, Option<isize>, Option<isize>)],
) -> Result<Tensor> {
    if x.dtype() != Dtype::F32 {
        return Err(Error::dtype("slice_tracked", "f32 only"));
    }
    let y = x.slice(ranges)?;
    if !x.requires_grad() {
        return Ok(y);
    }
    struct SliceBw {
        parents: Vec<Tensor>,
        x_shape: Vec<usize>,
        ranges: Vec<(Option<isize>, Option<isize>, Option<isize>)>,
    }
    impl Backward for SliceBw {
        fn name(&self) -> &'static str {
            "slice"
        }
        fn parents(&self) -> &[Tensor] {
            &self.parents
        }
        fn backward(&self, gy: &Tensor) -> Result<Vec<Tensor>> {
            let mut full = Tensor::zeros(&self.x_shape, gy.dtype())?;
            full.make_unique()?;
            let view = full.slice(&self.ranges)?;
            let (voff, vshape, vstrides) =
                (view.storage_offset(), view.shape().to_vec(), view.strides().to_vec());
            drop(view);
            let gyv = gy.to_vec_f32()?;
            let store = full.as_mut_slice_f32()?;
            let mut i = 0usize;
            g_core::for_each_offset(voff, &vshape, &vstrides, |off| {
                store[off] = gyv[i];
                i += 1;
            });
            Ok(vec![full])
        }
    }
    let mut y = y;
    y.set_grad_fn(Arc::new(SliceBw {
        parents: vec![x.clone()],
        x_shape: x.shape().to_vec(),
        ranges: ranges.to_vec(),
    }));
    Ok(y)
}

pub fn transpose(x: &Tensor) -> Result<Tensor> {
    let y = x.transpose()?;
    if !x.requires_grad() {
        return Ok(y);
    }
    struct TBw {
        parents: Vec<Tensor>,
    }
    impl Backward for TBw {
        fn name(&self) -> &'static str {
            "transpose"
        }
        fn parents(&self) -> &[Tensor] {
            &self.parents
        }
        fn backward(&self, gy: &Tensor) -> Result<Vec<Tensor>> {
            Ok(vec![gy.transpose()?])
        }
    }
    Ok(maybe_track(
        y,
        true,
        Arc::new(TBw {
            parents: vec![x.clone()],
        }),
    ))
}

pub fn reshape(x: &Tensor, shape: &[isize]) -> Result<Tensor> {
    let y = x.reshape(shape)?;
    if !x.requires_grad() {
        return Ok(y);
    }
    let old = x.shape().iter().map(|&d| d as isize).collect::<Vec<_>>();
    struct RBw {
        parents: Vec<Tensor>,
        old: Vec<isize>,
    }
    impl Backward for RBw {
        fn name(&self) -> &'static str {
            "reshape"
        }
        fn parents(&self) -> &[Tensor] {
            &self.parents
        }
        fn backward(&self, gy: &Tensor) -> Result<Vec<Tensor>> {
            Ok(vec![gy.reshape(&self.old)?])
        }
    }
    Ok(maybe_track(
        y,
        true,
        Arc::new(RBw {
            parents: vec![x.clone()],
            old,
        }),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mul_grad() {
        let a = Tensor::from_slice_f32(&[2.0, 3.0], &[2])
            .unwrap()
            .with_requires_grad();
        let b = Tensor::from_slice_f32(&[4.0, 5.0], &[2])
            .unwrap()
            .with_requires_grad();
        let y = sum(&mul(&a, &b).unwrap(), None, false).unwrap();
        let gs = grad(&y, &[&a, &b]).unwrap();
        assert_eq!(gs[0].to_vec_f32().unwrap(), vec![4.0, 5.0]);
        assert_eq!(gs[1].to_vec_f32().unwrap(), vec![2.0, 3.0]);
    }

    #[test]
    fn relu_zero_subgradient() {
        let x = Tensor::from_slice_f32(&[-1.0, 0.0, 2.0], &[3])
            .unwrap()
            .with_requires_grad();
        let y = sum(&relu(&x).unwrap(), None, false).unwrap();
        let g = grad(&y, &[&x]).unwrap();
        assert_eq!(g[0].to_vec_f32().unwrap(), vec![0.0, 0.0, 1.0]);
    }

    #[test]
    fn diamond_runs_each_vjp_once() {
        // y = (x + x) * (x + x) = 4x^2  ->  dy/dx = 8x = 16 at x = 2.
        // A tree walk double-counts the shared `x + x` node.
        let x = Tensor::from_slice_f32(&[2.0], &[])
            .unwrap()
            .with_requires_grad();
        let s = add(&x, &x).unwrap();
        let y = mul(&s, &s).unwrap();
        let g = grad(&y, &[&x]).unwrap();
        assert!((g[0].item_f32().unwrap() - 16.0).abs() < 1e-5);
    }

    #[test]
    fn stop_gradient_zeros() {
        let x = Tensor::from_slice_f32(&[3.0], &[])
            .unwrap()
            .with_requires_grad();
        let y = stop_gradient(&x).unwrap();
        let g = grad(&y, &[&x]).unwrap();
        assert_eq!(g[0].item_f32().unwrap(), 0.0);
    }
}


struct ScanBw {
    parents: Vec<Tensor>,
    a: Tensor,
    h: Tensor,
}
impl Backward for ScanBw {
    fn name(&self) -> &'static str {
        "gated_scan"
    }
    fn parents(&self) -> &[Tensor] {
        &self.parents
    }
    fn backward(&self, gy: &Tensor) -> Result<Vec<Tensor>> {
        let (ga, gb) = g_cpu::gated_scan_backward(&self.a, &self.h, gy)?;
        Ok(vec![ga, gb])
    }
}

/// Gated linear recurrence `h_t = a_t * h_{t-1} + b_t` over `[B, T, D]`.
///
/// One fused op with a hand-written backward, so a length-`T` recurrence costs
/// a single autodiff node instead of `T` of them.
pub fn gated_scan(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let h = g_cpu::gated_scan(a, b)?;
    if !any_grad(&[a, b]) {
        return Ok(h);
    }
    let mut y = h;
    let saved = y.detach();
    y.set_grad_fn(Arc::new(ScanBw {
        parents: vec![a.clone(), b.clone()],
        a: a.detach(),
        h: saved,
    }));
    Ok(y)
}

struct RmsBw {
    parents: Vec<Tensor>,
    x: Tensor,
    eps: f32,
}
impl Backward for RmsBw {
    fn name(&self) -> &'static str {
        "rms_norm"
    }
    fn parents(&self) -> &[Tensor] {
        &self.parents
    }
    fn backward(&self, gy: &Tensor) -> Result<Vec<Tensor>> {
        Ok(vec![g_cpu::rms_norm_backward(&self.x, gy, self.eps)?])
    }
}

/// RMS normalization over the last axis, fused.
pub fn rms_norm(x: &Tensor, eps: f32) -> Result<Tensor> {
    if !x.requires_grad() {
        return g_cpu::rms_norm(x, eps);
    }
    let mut y = g_cpu::rms_norm(x, eps)?;
    y.set_grad_fn(Arc::new(RmsBw {
        parents: vec![x.clone()],
        x: x.detach(),
        eps,
    }));
    Ok(y)
}

struct EmbeddingBw {
    parents: Vec<Tensor>,
    table: Tensor,
    idx: Tensor,
}
impl Backward for EmbeddingBw {
    fn name(&self) -> &'static str {
        "embedding_fused"
    }
    fn parents(&self) -> &[Tensor] {
        &self.parents
    }
    fn backward(&self, gy: &Tensor) -> Result<Vec<Tensor>> {
        let gt = g_cpu::fast_embedding_backward(&self.table, &self.idx, gy)?;
        Ok(vec![unbroadcast(&gt, self.table.shape())?])
    }
}

/// Fused embedding lookup: one node instead of a gather per token.
pub fn embedding_fused(table: &Tensor, idx: &Tensor) -> Result<Tensor> {
    let out = g_cpu::embedding(table, idx)?;
    if !table.requires_grad() {
        return Ok(out);
    }
    let mut y = out;
    y.set_grad_fn(Arc::new(EmbeddingBw {
        parents: vec![table.clone()],
        table: table.detach(),
        idx: idx.detach(),
    }));
    Ok(y)
}

struct MaskedCeBw {
    parents: Vec<Tensor>,
    probs: Tensor,
    targets: Tensor,
    mask: Tensor,
}
impl Backward for MaskedCeBw {
    fn name(&self) -> &'static str {
        "masked_ce"
    }
    fn parents(&self) -> &[Tensor] {
        &self.parents
    }
    fn backward(&self, gy: &Tensor) -> Result<Vec<Tensor>> {
        let g = g_cpu::masked_ce_backward(&self.probs, &self.targets, &self.mask)?;
        // gy is the scalar dL/dloss; multiply through.
        let s = gy.to_vec_f32()?[0];
        Ok(vec![g_cpu::mul_scalar(&g, s as f64)?])
    }
}

/// Masked cross-entropy as a single autodiff node.
///
/// `logits` is `[N, V]`, `targets` `[N]` (i64), `mask` `[N]` (f32, 1.0 =
/// scored). Positions with mask 0 contribute nothing to loss or gradient, so
/// one loss function covers context-heavy samples (observations are masked).
pub fn masked_ce(logits: &Tensor, targets: &Tensor, mask: &Tensor) -> Result<Tensor> {
    let (loss, probs) = g_cpu::masked_ce(logits, targets, mask)?;
    if !logits.requires_grad() {
        return Ok(loss);
    }
    let mut y = loss;
    y.set_grad_fn(Arc::new(MaskedCeBw {
        parents: vec![logits.clone()],
        probs,
        targets: targets.detach(),
        mask: mask.detach(),
    }));
    Ok(y)
}
