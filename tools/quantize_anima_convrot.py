#!/usr/bin/env python3
"""
Quantize Anima DiT (Cosmos 2) using ConvRot (Regular Hadamard Rotation) via comfy-kitchen & ComfyUI ops.

This script applies group-wise regular Hadamard rotation and quantization to Anima DiT
following the proven layer-retention policy (preserving visual quality at 100% BF16 fidelity):
- Preserved in BF16:
    * blocks.0 and blocks.1 (sensitive input representation)
    * blocks.27 (final latent reconstruction)
    * all adaln_modulation modules (timestep and guidance routing)
    * x_embedder and final_layer
- Quantized with ConvRot (group_size=256):
    * blocks.2 through blocks.26
    * Attention projections: q_proj, k_proj, v_proj, output_proj
    * MLP layers: mlp.layer1, mlp.layer2

Output safetensors file contains standard ComfyUI `comfy_quant` metadata, allowing
ComfyUI's model loader and comfy-kitchen to dispatch hardware-accelerated kernels
automatically with ZERO custom node registration required.
"""

import os
import sys
import json
import time
import argparse
import torch
from safetensors import safe_open
from safetensors.torch import save_file

# Add ComfyUI to path for native quantization layouts
COMFY_DIR = "/home/kuroko/ComfyUI"
if COMFY_DIR not in sys.path:
    sys.path.append(COMFY_DIR)

from comfy.quant_ops import QuantizedTensor


def should_quantize_layer(key: str) -> bool:
    """Determine whether a layer should be quantized according to Anima DiT quality policy."""
    # Must be a weight of a linear layer
    if not key.endswith(".weight"):
        return False

    # Never quantize sensitive boundary layers
    if "blocks.0." in key or "blocks.1." in key or "blocks.27." in key:
        return False
    if "adaln_modulation" in key:
        return False
    if "x_embedder" in key or "final_layer" in key:
        return False
    if "norm" in key:
        return False

    # Quantize attention projections and MLP layers in intermediate blocks
    target_matches = [
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.output_proj",
        "cross_attn.q_proj",
        "cross_attn.k_proj",
        "cross_attn.v_proj",
        "cross_attn.output_proj",
        "mlp.layer1",
        "mlp.layer2",
    ]
    return any(match in key for match in target_matches)


def quantize_anima_model(
    input_path: str,
    output_path: str,
    mode: str = "int8",
    convrot_groupsize: int = 256,
):
    print(f"[*] Loading source model: {input_path}")
    print(f"[*] Target mode: ConvRot {mode.upper()} (groupsize={convrot_groupsize})")
    print(f"[*] Target output: {output_path}")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    start_time = time.time()
    new_state_dict = {}
    quantized_count = 0
    preserved_count = 0

    with safe_open(input_path, framework="pt", device="cpu") as f:
        keys = list(f.keys())
        total_keys = len(keys)
        print(f"[*] Total tensors in checkpoint: {total_keys}")

        for idx, key in enumerate(keys, 1):
            tensor = f.get_tensor(key)

            if should_quantize_layer(key):
                # Ensure in_features is divisible by convrot_groupsize
                if tensor.shape[-1] % convrot_groupsize != 0:
                    print(f"[-] Skipping {key} (in_features {tensor.shape[-1]} not divisible by {convrot_groupsize})")
                    new_state_dict[key] = tensor
                    preserved_count += 1
                    continue

                prefix = key[:-len("weight")]  # e.g. "net.blocks.2.mlp.layer1."

                if mode == "int8":
                    q_weight = QuantizedTensor.from_float(
                        tensor,
                        "TensorWiseINT8Layout",
                        per_channel=True,
                        convrot=True,
                        convrot_groupsize=convrot_groupsize,
                    )
                    new_state_dict[key] = q_weight._qdata
                    new_state_dict[f"{prefix}weight_scale"] = q_weight._params.scale
                    quant_meta = {
                        "format": "int8_tensorwise",
                        "convrot": True,
                        "convrot_groupsize": convrot_groupsize,
                    }
                elif mode == "w4a4":
                    q_weight = QuantizedTensor.from_float(
                        tensor,
                        "TensorCoreConvRotW4A4Layout",
                        convrot_groupsize=convrot_groupsize,
                        quant_group_size=64,
                    )
                    new_state_dict[key] = q_weight._qdata
                    new_state_dict[f"{prefix}weight_scale"] = q_weight._params.scale
                    quant_meta = {
                        "format": "convrot_w4a4",
                        "convrot_groupsize": convrot_groupsize,
                        "quant_group_size": 64,
                        "linear_dtype": "int4",
                    }
                else:
                    raise ValueError(f"Unknown mode: {mode}")

                new_state_dict[f"{prefix}comfy_quant"] = torch.tensor(
                    list(json.dumps(quant_meta).encode("utf-8")),
                    dtype=torch.uint8,
                )
                quantized_count += 1
                if quantized_count % 10 == 0:
                    print(f"  [{idx}/{total_keys}] Quantized {quantized_count} layers with ConvRot...")
            else:
                new_state_dict[key] = tensor
                preserved_count += 1

    print(f"[*] Quantization complete in {time.time() - start_time:.2f}s!")
    print(f"[*] Layers quantized with ConvRot: {quantized_count}")
    print(f"[*] Layers preserved in full precision: {preserved_count}")

    print(f"[*] Saving quantized safetensors model to {output_path}...")
    save_start = time.time()
    save_file(new_state_dict, output_path)
    print(f"[*] Successfully saved in {time.time() - save_start:.2f}s!")

    orig_size_gb = os.path.getsize(input_path) / (1024**3)
    new_size_gb = os.path.getsize(output_path) / (1024**3)
    print(f"[*] Size comparison: {orig_size_gb:.2f} GB -> {new_size_gb:.2f} GB ({new_size_gb / orig_size_gb * 100:.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantize Anima DiT with ConvRot")
    parser.add_argument(
        "--input",
        type=str,
        default="/home/kuroko/ComfyUI/models/diffusion_models/anima-base-v1.0.safetensors",
        help="Path to source Anima base safetensors model",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/mnt/data/models/anima-base-v1.0-convrot-int8.safetensors",
        help="Path to save the quantized safetensors model",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["int8", "w4a4"],
        default="int8",
        help="Quantization precision (int8 or w4a4)",
    )
    parser.add_argument(
        "--groupsize",
        type=int,
        default=256,
        help="Hadamard block rotation group size (power of 4, default 256)",
    )
    args = parser.parse_args()

    quantize_anima_model(
        input_path=args.input,
        output_path=args.output,
        mode=args.mode,
        convrot_groupsize=args.groupsize,
    )
