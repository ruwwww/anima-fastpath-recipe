# Anima DiT Fast-Path Recipe (ComfyUI)

A validated, clean, and highly-optimized recipe to run **Anima Base v1.0** (Cosmos 2 DiT architecture) at full resolution (**832x1216**, 30 Steps, Batched CFG $B=2$) in **~14.9 - 15.3 seconds (~2.08 it/s)** on consumer GPUs (NVIDIA RTX 5060 Ti / RTX 40-series).

This setup uses pure PyTorch Inductor non-attention kernel fusion paired with `comfy-kitchen` SageAttention / CK-Attention and official startup flags—achieving a true ~2x end-to-end generation speedup without visual degradation or Turbo LoRAs.

## Key Performance Highlights
- **Resolution:** Native 832x1216 ($3,952$ spatial patch tokens)
- **Batching:** Batched CFG ($B=2$, Positive + Negative conditioning simultaneously)
- **Sampling:** 30 steps (`er_sde` / `simple` scheduler, CFG: 4.0)
- **Total Generation Time:** **~14.93s** (Sampling latency: ~0.48s per CFG step / ~2.08 it/s)
- **VRAM Usage:** ~5.0 GB peak active allocation during sampling (Fits comfortably on 8GB/16GB VRAM GPUs)

---

## 1. System Requirements & Dependencies

### Hardware & Environment
- **GPU:** NVIDIA Ampere, Ada Lovelace, or Blackwell (e.g. RTX 3060/4060/4070/4090/5060 Ti)
- **Python:** 3.10 or 3.11
- **PyTorch:** PyTorch 2.5+ or 2.6+ with CUDA 12.4+ / 12.8+ / 13.0+
- **ComfyUI:** Latest official upstream release

### Required Dependencies
```bash
# Core acceleration and custom kernels
pip install comfy-kitchen torch torchvision torchaudio triton ninja
```

Optional (Recommended Custom Node for Compile Graph):
- [ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes) (provides `TorchCompileModelAdvanced` node)

---

## 2. ComfyUI Startup Flags

Launch your ComfyUI backend with the exact high-performance flags below:

```bash
python main.py \
  --listen 0.0.0.0 \
  --port 8188 \
  --enable-cors-header \
  --preview-method auto \
  --fast \
  --use-ck-attention
```

### Explanation of Flags:
- `--fast`: Enables internal fast memory allocation optimizations and aggressive Tensor Core dispatch.
- `--use-ck-attention`: Activates hardware-accelerated CK-Attention (SageAttention backend) via `comfy-kitchen` for DiT cross-attention and self-attention operations.

---

## 3. Workflow Architecture (How to Assemble)

```
[UNETLoader (Anima Base)]
          │
          ▼ (MODEL)
[TorchCompileModelAdvanced]  <--- (backend: "inductor", mode: "default", fullgraph: False)
          │
          ▼ (MODEL)
[KSampler]  <--- (steps: 30, cfg: 4.0, sampler: "er_sde", scheduler: "simple")
          │
          ▼ (LATENT)
[VAEDecode (WanVAE)]
          │
          ▼ (IMAGE)
[SaveImage]
```

### Recommended Node Settings:
1. **Model Loader:**
   - Model: `anima-base-v1.0.safetensors`
2. **TorchCompileModelAdvanced:**
   - `backend`: `inductor`
   - `mode`: `default` (Fused AdaLN + Gated Residual elementwise kernels)
   - `fullgraph`: `False`
   - `dynamic`: `False`
3. **Text Encoder (CLIPLoader / DualCLIPLoader):**
   - Model: Qwen 3 0.6B / Anima Text Encoder (`fp16` / `bf16`)
4. **VAE:**
   - Model: `WanVAE` / `Wan2.1 VAE`
5. **KSampler:**
   - Steps: `30`
   - CFG: `4.0`
   - Sampler: `er_sde`
   - Scheduler: `simple`
   - Denoise: `1.0`

---

## 4. Workflows Included in this Repo

- `workflows/anima_fastpath_workflow.json`: Complete ComfyUI Canvas workflow (Drag-and-drop onto ComfyUI Web GUI).
- `workflows/anima_fastpath_api_prompt.json`: Direct ComfyUI API prompt payload for headless automation.

---

## 5. Profiling Breakdown (RTX 5060 Ti)

| Sub-system / Kernel Stage | Implementation | Latency per CFG Step ($B=2$) | % GPU Time |
| :--- | :--- | :--- | :--- |
| **Linear Layers (GEMM)** | CUTLASS TensorOp BF16 | ~295 ms | 61.5% |
| **Self & Cross Attention** | CK-Attention (SageAttention) | ~80 ms | 16.7% |
| **Elementwise & Modulation** | Inductor Fused AdaLN + Residual | ~65 ms | 13.5% |
| **Normalization & RoPE** | Fused RMSNorm + Half RoPE | ~40 ms | 8.3% |
| **Total Iteration Time** | **Fast-Path Pipeline** | **~480 ms / it (~2.08 it/s)** | **100%** |
