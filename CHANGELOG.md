# Changelog

## 0.1.0

- `grad(output, inputs)` now clears all reachable leaf slots before its reverse pass, so repeated functional gradients are fresh; `backward` retains explicit accumulation semantics
- Gradcheck suite for unary/binary/reduce/matmul/take/gather/embed/cat/stack/transpose/amax
- VJPs for `cat`, `stack`, `gather`, `amax`, `embedding`, `unsqueeze`, `transpose`/`reshape`
- Intermediate tensors no longer share leaf accumulators (view AD bugfix)
- `sum`/`mean` VJP inserts reduced axes when `keepdims=false`


First shippable release of `g` on Apple Silicon.

- CPU-first tensors, ops, first-order AD, SGD/AdamW
- Local predictive coding via `stop_gradient`
- Optional Accelerate GEMM
- Optional MPSGraph GEMM for large FP32 matrices (`--features gpu`)
- GPU backend selects the best Metal device and splits large GEMMs across all
  Metal devices plus the CPU in parallel (Intel iGPU + AMD dGPU supported;
  iGPU opt-in via `set_include_integrated(true)`, dGPU + CPU on by default)
- Examples: `mlp`, `pc_local`
