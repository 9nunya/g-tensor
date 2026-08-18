use crate::error::Result;
use crate::tensor::Tensor;

/// First-order VJP node. Parents are the tensors that receive gradient pieces.
///
/// A [`Backward`] implementation encodes the vector-Jacobian product for one
/// primitive op. During the reverse pass the engine calls
/// [`Backward::backward`] with the incoming cotangent `gy` and expects one
/// gradient contribution per parent, in [`Backward::parents`] order. Ops with
/// no differentiable parents return an empty vector.
pub trait Backward: Send + Sync {
    /// Stable name of the op (used in debug output and graph logs).
    fn name(&self) -> &'static str;
    /// Tensors that produced this node and receive gradient pieces.
    fn parents(&self) -> &[Tensor];
    /// Return `parents().len()` gradient contributions for upstream cotangent `gy`.
    fn backward(&self, gy: &Tensor) -> Result<Vec<Tensor>>;
}
