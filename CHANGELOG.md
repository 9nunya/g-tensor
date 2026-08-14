# Changelog

## 0.1.0

- Gradcheck suite for unary/binary/reduce/matmul/take/gather/embed/cat/stack/transpose/amax
- VJPs for `cat`, `stack`, `gather`, `amax`, `embedding`, `unsqueeze`, `transpose`/`reshape`
- Intermediate tensors no longer share leaf accumulators (view AD bugfix)
- `sum`/`mean` VJP inserts reduced axes when `keepdims=false`


First shippable release of `g` on Apple Silicon.

- CPU-first tensors, ops, first-order AD, SGD/AdamW
- Local predictive coding via `stop_gradient`
- Optional Accelerate GEMM
- Optional MPSGraph GEMM for large FP32 matrices (`--features gpu`)
- Examples: `mlp`, `pc_local`
