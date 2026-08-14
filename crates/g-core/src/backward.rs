use crate::error::Result;
use crate::tensor::Tensor;

/// First-order VJP node. Parents are the tensors that receive gradient pieces.
pub trait Backward: Send + Sync {
    fn name(&self) -> &'static str;
    fn parents(&self) -> &[Tensor];
    fn backward(&self, gy: &Tensor) -> Result<Vec<Tensor>>;
}
