"""
FP8 Dequant → BF16 → Requant FP8 script.

Usage:
    python fp8_dequant_requant.py --input-fp8-hf-path /path/to/fp8_model --output-fp8-hf-path /path/to/output
"""

import json
import os
from argparse import ArgumentParser
from glob import glob

import torch
import triton
import triton.language as tl
from safetensors.torch import load_file, save_file
from tqdm import tqdm

from miles.utils.fp8_kernel import blockwise_cast_to_fp8_triton

TARGET_SHAPE = (10, 112)


@triton.jit
def weight_dequant_kernel(x_ptr, s_ptr, y_ptr, M, N, BLOCK_SIZE: tl.constexpr):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    n = tl.cdiv(N, BLOCK_SIZE)
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs = offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    s = tl.load(s_ptr + pid_m * n + pid_n)
    y = x * s
    tl.store(y_ptr + offs, y, mask=mask)


def weight_dequant(x: torch.Tensor, s: torch.Tensor, block_size: int = 128) -> torch.Tensor:
    assert x.is_contiguous() and s.is_contiguous()
    assert x.dim() == 2 and s.dim() == 2
    M, N = x.size()
    y = torch.empty_like(x, dtype=torch.bfloat16)

    def grid(meta):
        return (triton.cdiv(M, meta["BLOCK_SIZE"]), triton.cdiv(N, meta["BLOCK_SIZE"]))

    weight_dequant_kernel[grid](x, s, y, M, N, BLOCK_SIZE=block_size)
    return y


def main(fp8_path, output_path):
    torch.set_default_dtype(torch.bfloat16)
    os.makedirs(output_path, exist_ok=True)

    # Copy non-weight files
    for pattern in ["config.json", "*.py", "tokenizer*", "chat_template*"]:
        os.system(f"cp -rf {fp8_path}/{pattern} {output_path}/ 2>/dev/null")

    model_index_file = os.path.join(fp8_path, "model.safetensors.index.json")
    with open(model_index_file) as f:
        model_index = json.load(f)
    weight_map = model_index["weight_map"]

    loaded_files = {}

    def get_tensor(tensor_name):
        file_name = weight_map[tensor_name]
        if file_name not in loaded_files:
            loaded_files[file_name] = load_file(os.path.join(fp8_path, file_name), device="cuda")
        return loaded_files[file_name][tensor_name]

    new_weight_map = {}
    safetensor_files = sorted(glob(os.path.join(fp8_path, "*.safetensors")))

    for safetensor_file in tqdm(safetensor_files, desc="Processing shards"):
        file_name = os.path.basename(safetensor_file)
        print(f"Processing: {file_name}")
        current_state_dict = load_file(safetensor_file, device="cuda")
        loaded_files[file_name] = current_state_dict

        new_state_dict = {}
        for name, weight in current_state_dict.items():
            if name.endswith("_scale_inv") or name.endswith("_scale"):
                continue

            if weight.element_size() == 1:  # FP8 weight
                scale_inv_name = f"{name}_scale_inv"
                try:
                    scale_inv = get_tensor(scale_inv_name)
                except KeyError:
                    print(f"Warning: Missing scale_inv for {name}, keeping original")
                    new_state_dict[name] = weight
                    new_weight_map[name] = file_name
                    continue

                # Step 1: Dequant FP8 → BF16
                bf16_weight = weight_dequant(weight, scale_inv)

                # Step 2: Requant BF16 → FP8 with blockwise quantization
                qweight, scale = blockwise_cast_to_fp8_triton(bf16_weight, [128, 128])

                # Convert scale from 128x128 block layout to 64x64 block layout
                scale = scale.repeat_interleave(2, dim=0).repeat_interleave(2, dim=1)

                # Handle special shape trimming (from _quantize_param)
                if tuple(scale.shape) == TARGET_SHAPE:
                    scale = scale[:9, :]

                new_state_dict[name] = qweight
                new_weight_map[name] = file_name

                new_scale_inv_name = f"{name}_scale_inv"
                new_state_dict[new_scale_inv_name] = scale
                new_weight_map[new_scale_inv_name] = file_name

                print(f"  {name}: FP8→BF16→FP8, scale shape={scale.shape}")
            else:
                # Non-FP8 weight, keep as-is
                new_state_dict[name] = weight
                new_weight_map[name] = file_name

        save_file(new_state_dict, os.path.join(output_path, file_name))

        # Memory management: keep only 2 most recent files
        if len(loaded_files) > 2:
            oldest_file = next(iter(loaded_files))
            del loaded_files[oldest_file]
            torch.cuda.empty_cache()

    # Save updated model index
    with open(os.path.join(output_path, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {}, "weight_map": new_weight_map}, f, indent=2)

    print(f"Done. Output saved to {output_path}")

# python fp8_dequant_requant.py \
#    --input-fp8-hf-path /root/models/Kimi-K2-Instruct \
#    --output-fp8-hf-path /root/models/Kimi-K2-Instruct-deq-req
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input-fp8-hf-path", type=str, required=True, help="Path to input FP8 model")
    parser.add_argument("--output-fp8-hf-path", type=str, required=True, help="Path to output requantized FP8 model")
    args = parser.parse_args()
    main(args.input_fp8_hf_path, args.output_fp8_hf_path)
