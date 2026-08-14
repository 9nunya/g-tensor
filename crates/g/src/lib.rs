//! CPU-first tensor and autodiff library for Apple Silicon Macs.
//!
//! `g` is **not** a PyTorch clone. Ops return [`Result`], in-place mutation
//! requires unique ownership, and small/medium work stays on the CPU because
//! that is what is fast on an M3-class machine.
//!
//! # Quick start
//! ```
//! use g::prelude::*;
//! # fn main() -> Result<()> {
//! let x = from_slice_f32(&[1.0, 2.0], &[1, 2])?;
//! let w = from_slice_f32(&[0.5, -0.25], &[2, 1])?.with_requires_grad();
//! let y = x.linear(&w, None)?.relu()?;
//! let loss = y.mse_loss(&from_slice_f32(&[0.0], &[1, 1])?, Reduce::Mean)?;
//! let gs = grad(&loss, &[&w])?;
//! assert_eq!(gs[0].shape(), w.shape());
//! # Ok(())
//! # }
//! ```
//!
//! Free functions (`add`, `matmul`, …) are the primitives. [`TensorExt`] is
//! method sugar for the same functions.
//!
//! Enable `cpu-accelerate` for Accelerate GEMM and `gpu` for large FP32
//! MPSGraph matmuls. Neither changes the default small-op placement.
//!
//! First-order reverse AD only. Unrolled / implicit PC is not in 0.1.

#[doc(inline)]
pub use g_core::{
    arange_f32, eye, from_slice_f32, from_slice_f64, from_slice_i64, full_f32, linspace_f32, ones,
    randn_f32, zeros, Device, Dtype, Error, ErrorKind, Result, Tensor,
};

#[doc(inline)]
pub use g_ad::{backward, detach, gated_scan, grad, masked_ce, rms_norm, stop_gradient, zero_grad};
#[doc(inline)]
pub use g_nn::{
    categorical_entropy, categorical_log_prob, cross_entropy, embedding, layer_norm, linear,
    log_softmax, mse_loss, nll_loss, one_hot, softmax, whiten, AdamW, Reduce, Sgd,
};

/// Elementwise `a + b` with right-aligned broadcasting.
pub fn add(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    g_ad::add(a, b)
}
/// Elementwise `a - b`.
pub fn sub(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    g_ad::sub(a, b)
}
/// Elementwise `a * b`.
pub fn mul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    g_ad::mul(a, b)
}
/// Elementwise `a / b`.
pub fn div(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    g_ad::div(a, b)
}
/// Scale every element by `s` (cast to the tensor dtype).
pub fn mul_scalar(a: &Tensor, s: f64) -> Result<Tensor> {
    g_ad::mul_scalar(a, s)
}
/// Elementwise negation.
pub fn neg(a: &Tensor) -> Result<Tensor> {
    g_ad::neg(a)
}
/// Dense matrix product. Rank-0 is rejected. 1-D operands follow the
/// `(k,)@(k,)→()` / matvec rules. Large FP32 GEMMs may run on MPSGraph
/// when the `gpu` feature is on; small/medium work stays on CPU.
pub fn matmul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    #[cfg(feature = "gpu")]
    {
        if g_apple::should_offload_matmul(a, b) {
            if let Ok(y) = g_apple::matmul(a, b) {
                return Ok(y);
            }
        }
    }
    g_ad::matmul(a, b)
}
/// `max(x, 0)`. Subgradient at 0 is **0**.
pub fn relu(a: &Tensor) -> Result<Tensor> {
    g_ad::relu(a)
}
/// Hyperbolic tangent.
pub fn tanh(a: &Tensor) -> Result<Tensor> {
    g_ad::tanh(a)
}
/// Elementwise exponential.
pub fn exp(a: &Tensor) -> Result<Tensor> {
    g_ad::exp(a)
}
/// Natural logarithm. Domain `≤ 0` follows IEEE (`-inf` / `NaN`).
pub fn log(a: &Tensor) -> Result<Tensor> {
    g_ad::log(a)
}
/// Elementwise square root.
pub fn sqrt(a: &Tensor) -> Result<Tensor> {
    g_ad::sqrt(a)
}
/// Elementwise absolute value.
pub fn abs(a: &Tensor) -> Result<Tensor> {
    g_ad::abs(a)
}
/// Logistic sigmoid `1 / (1 + e^{-x})`.
pub fn sigmoid(a: &Tensor) -> Result<Tensor> {
    g_ad::sigmoid(a)
}
/// SiLU / swish: `x · sigmoid(x)`.
pub fn silu(a: &Tensor) -> Result<Tensor> {
    g_ad::silu(a)
}
/// GELU (tanh approximation).
pub fn gelu(a: &Tensor) -> Result<Tensor> {
    g_ad::gelu(a)
}
/// Softplus `log(1 + e^x)` with a large-`x` identity.
pub fn softplus(a: &Tensor) -> Result<Tensor> {
    g_ad::softplus(a)
}
/// Leaky ReLU with negative slope `slope`.
pub fn leaky_relu(a: &Tensor, slope: f64) -> Result<Tensor> {
    g_ad::leaky_relu(a, slope)
}
/// Clamp every element into `[min, max]`.
pub fn clamp(a: &Tensor, min: f64, max: f64) -> Result<Tensor> {
    g_ad::clamp(a, min, max)
}
/// Sum. `axes = None` reduces all dims to a scalar. Empty slices are **0**.
pub fn sum(x: &Tensor, axes: Option<&[isize]>, keepdims: bool) -> Result<Tensor> {
    g_ad::sum(x, axes, keepdims)
}
/// Mean. Empty slices are **NaN**.
pub fn mean(x: &Tensor, axes: Option<&[isize]>, keepdims: bool) -> Result<Tensor> {
    g_ad::mean(x, axes, keepdims)
}
/// Concatenate along `axis`.
pub fn cat(tensors: &[&Tensor], axis: isize) -> Result<Tensor> {
    g_ad::cat(tensors, axis)
}
/// Stack along a new `axis`.
pub fn stack(tensors: &[&Tensor], axis: isize) -> Result<Tensor> {
    g_ad::stack(tensors, axis)
}
/// Maximum along `axis`. Empty reductions **error**. Ties use the first index in the VJP.
pub fn amax(x: &Tensor, axis: isize, keepdims: bool) -> Result<Tensor> {
    g_ad::amax(x, axis, keepdims)
}
/// Gather `index` along `axis`. `index` is `i64`, same rank as `x`. OOB is an error.
pub fn gather(x: &Tensor, axis: isize, index: &Tensor) -> Result<Tensor> {
    g_ad::gather(x, axis, index)
}
/// Insert a size-1 axis (tracked for AD).
pub fn unsqueeze(x: &Tensor, axis: isize) -> Result<Tensor> {
    g_ad::unsqueeze(x, axis)
}
/// `dst[index] += src` along `axis`. Duplicate indices **sum**.
pub fn scatter_add(dst: &Tensor, axis: isize, index: &Tensor, src: &Tensor) -> Result<Tensor> {
    g_cpu::scatter_add(dst, axis, index, src)
}

/// Population variance (`mean((x - mean)^2)`).
pub fn variance(x: &Tensor, axes: Option<&[isize]>, keepdims: bool) -> Result<Tensor> {
    g_ad::variance(x, axes, keepdims)
}
/// `sqrt(variance(x))`.
pub fn stddev(x: &Tensor, axes: Option<&[isize]>, keepdims: bool) -> Result<Tensor> {
    g_ad::stddev(x, axes, keepdims)
}
/// Numerically stable `log(sum(exp(x)))` along `axis`.
pub fn logsumexp(x: &Tensor, axis: isize, keepdims: bool) -> Result<Tensor> {
    g_ad::logsumexp(x, axis, keepdims)
}
/// Elementwise maximum.
pub fn maximum(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    g_ad::maximum(a, b)
}
/// Elementwise minimum.
pub fn minimum(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    g_ad::minimum(a, b)
}
/// Select `x[i, index[i]]` for a rank-2 `x` and rank-1 `i64` index (batch take).
pub fn take(x: &Tensor, axis: isize, index: &Tensor) -> Result<Tensor> {
    g_ad::take(x, axis, index)
}
/// Swap the last two axes. Participates in AD (unlike a raw view clone).
pub fn transpose(x: &Tensor) -> Result<Tensor> {
    g_ad::transpose(x)
}
/// Reshape with at most one `-1`. Participates in AD.
pub fn reshape(x: &Tensor, shape: &[isize]) -> Result<Tensor> {
    g_ad::reshape(x, shape)
}

/// Method sugar for the free functions above. Import via [`prelude`].
pub trait TensorExt {
    /// [`add`]
    fn add(&self, other: &Tensor) -> Result<Tensor>;
    /// [`sub`]
    fn sub(&self, other: &Tensor) -> Result<Tensor>;
    /// [`mul`]
    fn mul(&self, other: &Tensor) -> Result<Tensor>;
    /// [`div`]
    fn div(&self, other: &Tensor) -> Result<Tensor>;
    /// [`matmul`]
    fn matmul(&self, other: &Tensor) -> Result<Tensor>;
    /// [`relu`]
    fn relu(&self) -> Result<Tensor>;
    /// [`tanh`]
    fn tanh(&self) -> Result<Tensor>;
    /// [`gelu`]
    fn gelu(&self) -> Result<Tensor>;
    /// [`sigmoid`]
    fn sigmoid(&self) -> Result<Tensor>;
    /// [`exp`]
    fn exp(&self) -> Result<Tensor>;
    /// [`log`]
    fn log(&self) -> Result<Tensor>;
    /// [`sum`]
    fn sum(&self, axes: Option<&[isize]>, keepdims: bool) -> Result<Tensor>;
    /// [`mean`]
    fn mean(&self, axes: Option<&[isize]>, keepdims: bool) -> Result<Tensor>;
    /// [`linear`]: `self @ w + b` with `w` shaped `[in, out]`.
    fn linear(&self, w: &Tensor, b: Option<&Tensor>) -> Result<Tensor>;
    /// [`softmax`]
    fn softmax(&self, axis: isize) -> Result<Tensor>;
    /// [`log_softmax`]
    fn log_softmax(&self, axis: isize) -> Result<Tensor>;
    /// [`mse_loss`]
    fn mse_loss(&self, target: &Tensor, reduction: Reduce) -> Result<Tensor>;
    /// Reverse-mode AD if `self` is a scalar (`numel == 1`).
    fn backward(&self) -> Result<Vec<(Tensor, Tensor)>>;
    /// [`stop_gradient`]
    fn stop_gradient(&self) -> Result<Tensor>;
    /// [`transpose`]
    fn t(&self) -> Result<Tensor>;
}

impl TensorExt for Tensor {
    fn add(&self, other: &Tensor) -> Result<Tensor> {
        add(self, other)
    }
    fn sub(&self, other: &Tensor) -> Result<Tensor> {
        sub(self, other)
    }
    fn mul(&self, other: &Tensor) -> Result<Tensor> {
        mul(self, other)
    }
    fn div(&self, other: &Tensor) -> Result<Tensor> {
        div(self, other)
    }
    fn matmul(&self, other: &Tensor) -> Result<Tensor> {
        matmul(self, other)
    }
    fn relu(&self) -> Result<Tensor> {
        relu(self)
    }
    fn tanh(&self) -> Result<Tensor> {
        tanh(self)
    }
    fn gelu(&self) -> Result<Tensor> {
        gelu(self)
    }
    fn sigmoid(&self) -> Result<Tensor> {
        sigmoid(self)
    }
    fn exp(&self) -> Result<Tensor> {
        exp(self)
    }
    fn log(&self) -> Result<Tensor> {
        log(self)
    }
    fn sum(&self, axes: Option<&[isize]>, keepdims: bool) -> Result<Tensor> {
        sum(self, axes, keepdims)
    }
    fn mean(&self, axes: Option<&[isize]>, keepdims: bool) -> Result<Tensor> {
        mean(self, axes, keepdims)
    }
    fn linear(&self, w: &Tensor, b: Option<&Tensor>) -> Result<Tensor> {
        linear(self, w, b)
    }
    fn softmax(&self, axis: isize) -> Result<Tensor> {
        softmax(self, axis)
    }
    fn log_softmax(&self, axis: isize) -> Result<Tensor> {
        log_softmax(self, axis)
    }
    fn mse_loss(&self, target: &Tensor, reduction: Reduce) -> Result<Tensor> {
        mse_loss(self, target, reduction)
    }
    fn backward(&self) -> Result<Vec<(Tensor, Tensor)>> {
        backward(self)
    }
    fn stop_gradient(&self) -> Result<Tensor> {
        stop_gradient(self)
    }
    fn t(&self) -> Result<Tensor> {
        transpose(self)
    }
}

#[cfg(feature = "gpu")]
pub use g_apple;

/// Common imports: [`Tensor`], [`TensorExt`], constructors, and a few ops.
pub mod prelude {
    pub use crate::TensorExt;
    pub use crate::{
        add, arange_f32, backward, cat, cross_entropy, detach, from_slice_f32, gated_scan, gelu,
        grad, linear, matmul, mse_loss, randn_f32, relu, rms_norm, sigmoid, softmax,
        stop_gradient, sum, tanh, zeros, Device, Dtype, Reduce, Result, Tensor,
    };
}
