# g

A Rust tensor and ML library for **Apple Silicon** and Intel Macs with Metal,
designed to feel as easy as PyTorch without copying its API.

v1 is **CPU-first**. On an M3 Air, representative PC/training MLP forwards are
faster on the CPU than on naive Metal. Large GEMMs can use MPSGraph when you
build with `--features gpu`; on a multi-GPU machine the backend splits one
large GEMM across every Metal device **and** the CPU in parallel.

```rust
use g::prelude::*;

fn main() -> Result<()> {
    let x = from_slice_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2])?;
    let w = from_slice_f32(&[0.5, 0.0, 0.0, 0.5], &[2, 2])?.with_requires_grad();
    let y = x.linear(&w, None)?.gelu()?;
    let loss = y.mse_loss(&g::zeros(&[2, 2], Dtype::F32)?, Reduce::Mean)?;
    let gs = grad(&loss, &[&w])?;
    println!("{}", gs[0].to_vec_f32()?[0]);
    Ok(())
}
```

## What’s in the box

- Runtime-rank tensors (`f32`, `f64`, `i64`) on CPU
- Construction: `from_slice`, `zeros`/`ones`/`full`, `arange`, `linspace`, `eye`, `randn`
- Views: `reshape`/`view`, `permute`, `slice`, `squeeze`/`unsqueeze`, `flatten`, `cast`
- Elementwise: add/sub/mul/div, exp/log/sqrt/abs, relu/leaky_relu/gelu/silu/sigmoid/tanh/softplus/clamp
- Reductions: `sum`, `mean`, `amax` (empty max errors)
- Linear algebra: `matmul`, `linear`
- Indexing: `gather`, `scatter_add`, `cat`/`stack`
- NN: softmax, log_softmax, mse, nll, cross-entropy, layer_norm, embedding
- AD: fresh functional `grad(&loss, &[&w])`, accumulating `.backward()` on scalars, `stop_gradient` / `detach`
- Optim: SGD, AdamW
- Local predictive coding without reversing through inference (`stop_gradient`)
- Optional Accelerate GEMM (`cpu-accelerate`)
- Optional MPSGraph GEMM for large FP32 mats (`gpu`): single device, all Metal
  devices in parallel, or all devices plus the CPU

## Non-goals

No Python. No PyTorch clone. No CUDA. No iOS. ANE is not a supported device.

## Docs

```bash
cargo doc -p g --no-deps --open
```

That is the API reference (every public fn/type). There is no docs.rs page because
the crate is not published.

## Build

```bash
cargo test --workspace
cargo run -p g --example mlp
cargo run -p g --example pc_local
cargo test -p g-cpu --features accelerate --lib
cargo test -p g --features gpu --test gpu_gemm   # large 1024 GEMM
```

MSRV: current stable. License: MIT OR Apache-2.0.

## Design

CPU is the default because that is what is fast on the M3 Air at PC/train
shapes. Do not call `.to(Gpu)` to “go faster” on small layers. Performance
regression gates are not enabled yet.

On a dual-GPU Intel MacBook the GPU backend uses the AMD dGPU plus the CPU in
parallel by default, and leaves the Intel iGPU out (it shares memory bandwidth
with the CPU and adds little once the CPU is busy). Call
`g::set_include_integrated(true)` to opt the iGPU back in; `gpu_device_names()`
reports the devices actually used for GEMMs.


## Examples

```bash
cargo run -p g --example mlp              # MLP regression
cargo run -p g --example classify_blobs   # 3-class CE
cargo run -p g --example autoencoder
cargo run -p g --example adamw_sine       # fit sin(x) with AdamW
cargo run -p g --example bandit           # softmax Bernoulli bandit
cargo run -p g --example reinforce_cartpole
cargo run -p g --example ppo_cartpole     # clipped PPO
cargo run -p g --example dqn_gridworld
cargo run -p g --example pc_local         # local PC energy
cargo run -p g --example pc_train         # PC inference + Hebbian update
cargo run -p g --example attention_toy    # 2-token attention
```


## 0.1 status

This is a **0.1** crate: first-order reverse AD, CPU-first, Apple Silicon.
The public API may still change. There is no blocking performance CI gate.
Conv/attention-as-a-module, JVP/HVP, and a full device runtime are not included.

What *is* covered: the ops used by the examples, plus a finite-difference
gradcheck file (`tests/gradcheck.rs`) for the differentiable surface.
