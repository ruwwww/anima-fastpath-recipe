# Anima DiT Fast-Path Recipe (ComfyUI)

A validated, clean, and highly-optimized recipe to run **Anima Base v1.0** (Cosmos 2 DiT architecture) at full resolution (**832x1216**, 30 Steps, Batched CFG $B=2$) in **~11.2 - 11.5 seconds (~2.59 - 2.67 it/s)** on modern Blackwell consumer GPUs (NVIDIA RTX 5060 Ti 16GB / RTX 50-series) and **~14.9 - 15.3 seconds (~2.05 it/s)** on BF16 base weights.

This setup leverages PyTorch Inductor non-attention kernel fusion, native ComfyUI compile nodes, hardware-accelerated CK-Attention (SageAttention backend), and modern **MXFP8 & ConvRot INT8 quantized checkpoints**—achieving maximum throughput with zero custom node registration.

## Benchmark & Fidelity Comparison (RTX 5060 Ti 16GB)

Tested on native 832x1216 resolution, 30 steps, `er_sde` / `simple` scheduler, CFG 4.0 - 5.0, Batched CFG ($B=2$, positive + negative conditioning concurrently):

| Model Format | Checkpoint | Step Latency ($B=2$) | Generation Speed | Total 30-Step Time | Visual / Semantic Fidelity vs BF16 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **MXFP8** | `anima-base-v1.0-mxfp8` | ~374 ms / step | **2.67 it/s** | **~11.2s** | Fast Blackwell microscaling; slight micro-texture variance due to activation outliers |
| **ConvRot INT8** *(Recommended)* | `anima-base-v1.0-convrot-int8` | ~386 ms / step | **2.59 it/s** | **~11.5s** | **~90% Winrate vs MXFP8**; nearly identical to BF16 ground truth via Regular Hadamard rotation |
| **BF16 Ground Truth** | `anima-base-v1.0` | ~487 ms / step | **2.05 it/s** | **~14.9s** | Reference baseline |

### Why is MXFP8 slightly faster than ConvRot INT8?
- **Native Blackwell Execution:** MXFP8 computes directly using 5th-gen Tensor Core hardware microscaling without requiring online activation transformation.
- **Online Activation Rotation Overhead:** ConvRot weights are pre-rotated offline ($W \cdot H$), but activations must undergo a group-wise Regular Hadamard Transform ($X \cdot H$) online at runtime. This adds ~0.4 ms per block (~12 ms total per step), which is a negligible price (~0.3s overall) for near-lossless BF16 fidelity.

---

## 1. Download Quantized Models

### Option A: ConvRot INT8 (Recommended for Daily Production)
* **Hugging Face Model:** [ruwwww/Anima-ConvRot](https://huggingface.co/ruwwww/Anima-ConvRot)
* **File:** `anima-base-v1.0-convrot-int8.safetensors` (2.41 GB, ~38% VRAM footprint reduction)
* **Features:** Regular Hadamard block rotation ($N_0=256$), zero outlier clipping, ~90% semantic match to BF16.

### Option B: Blackwell MXFP8
* **Hugging Face Model:** [Bedovyy/Anima-FP8](https://huggingface.co/Bedovyy/Anima-FP8/tree/main)
* **File:** `anima-base-v1.0-mxfp8.safetensors`
* **Features:** Microscaling FP8 format tuned for raw compute speed on sm_120 Blackwell Tensor Cores.

---

## 2. Layer Retention Policy (Zero Visual Degradation)

Both ConvRot INT8 and MXFP8 enforce a selective quantization rule that leaves critical boundary layers untouched:

- **Preserved in Full Precision (`BF16`):**
  - Initial blocks: `blocks.0.` and `blocks.1.` (input patch feature projection)
  - Final block: `blocks.27.` (high-frequency spatial reconstruction)
  - All conditioning routing: `adaln_modulation` (timestep & guidance gates)
  - Boundary layers: `x_embedder` and `final_layer`
- **Quantized (MXFP8 / ConvRot INT8):**
  - Intermediate blocks: `blocks.2.` through `blocks.26.`
  - Linear Attention Projections: `q_proj`, `k_proj`, `v_proj`, `output_proj`
  - Feed-Forward Networks (MLP): `mlp.layer1` and `mlp.layer2` (which dominate ~70% of total DiT FLOPs)

---

## 3. System Requirements & Startup Flags

### Requirements
- **GPU:** NVIDIA Ampere, Ada Lovelace, or Blackwell (RTX 3060/4060/4070/4090/5060 Ti)
- **PyTorch:** 2.5+ with CUDA 12.4+ / 12.8+ / 13.0+
- **ComfyUI:** Latest official upstream release with `comfy-kitchen` installed:
  ```bash
  pip install comfy-kitchen torch torchvision torchaudio triton ninja
  ```

### ComfyUI Startup Flags
Launch ComfyUI with the fast memory allocator and SageAttention backend:
```bash
python main.py \
  --listen 0.0.0.0 \
  --port 8188 \
  --enable-cors-header \
  --preview-method auto \
  --fast \
  --use-ck-attention
```

---

## 4. Workflow Architecture

Both ConvRot and MXFP8 models contain standard ComfyUI `comfy_quant` metadata and are loaded automatically by `comfy-kitchen` with **ZERO custom node registration required**.

```
[UNETLoader (anima-base-v1.0-convrot-int8 OR mxfp8)]
          │
          ▼ (MODEL)
[TorchCompileModel (Native)] OR [TorchCompileModelAdvanced]
          │
          ▼ (COMPILED MODEL)
[KSampler]  <--- (steps: 30, cfg: 4.0-5.0, sampler: "er_sde", scheduler: "simple")
          │
          ▼ (LATENT)
[VAEDecode (qwen_image_vae / WanVAE)]
          │
          ▼ (IMAGE)
[SaveImage]
```

### Node Configuration:
1. **UNETLoader:** Select `anima-base-v1.0-convrot-int8.safetensors` or `anima-base-v1.0-mxfp8.safetensors`.
2. **TorchCompileModel (Native ComfyUI Node):**
   - Built-in under `comfy_extras/nodes_torch_compile.py`.
   - `backend`: `inductor`
   - Automatically applies `disable_dynamic=True` and dictionary guard filtering.
3. **KSampler:**
   - Steps: `30`
   - CFG: `4.0` - `5.0`
   - Sampler: `er_sde`
   - Scheduler: `simple`
   - Denoise: `1.0`

---

## 5. Standalone Quantization Converter

You can quantize any custom Anima DiT checkpoint yourself using the included converter tool:

```bash
# Quantize to ConvRot INT8 (default, group_size=256)
python tools/quantize_anima_convrot.py \
  --input /path/to/anima-base-v1.0.safetensors \
  --output /path/to/anima-base-v1.0-convrot-int8.safetensors \
  --mode int8

# Quantize to ConvRot W4A4
python tools/quantize_anima_convrot.py \
  --input /path/to/anima-base-v1.0.safetensors \
  --output /path/to/anima-base-v1.0-convrot-w4a4.safetensors \
  --mode w4a4
```

The script automatically embeds standard `comfy_quant` metadata into the output safetensors file, allowing ComfyUI to dispatch `ck.int8_linear` and `ck.convrot_w4a4_linear` kernels out of the box.
