"""
Benchmark for RDMA batch_register_memory with real model tensor shapes.

Creates tensors matching the exact model replica layout from new_tensorlog.log
(849 params, ~13.94 GB per GPU, bf16) as logged by prepare_memory_region.

Runs on ALL 8 GPUs of a node concurrently (one process per GPU) to simulate
the real multi-GPU registration scenario.

Compares two approaches:
  v3: per-tensor batch_register_memory (single call, internally parallel)
  v4: merged-block batch_register_memory (single call, internally parallel)

Usage:
    python miles/tests/test_batch_register_memory_benchmark.py \
        [--num-gpus 8]
"""

import argparse
import time

import torch
import torch.multiprocessing as mp
from mooncake.engine import TransferEngine
import ray


# ---------------------------------------------------------------------------
#  Tensor specs from new_tensorlog.log (849 params, ~13.94 GB per GPU)
#
#  Model: 94 layers (0-93), all MoE, hidden_size=4096
#  128 total experts, 4 local experts per GPU (EP=32)
#  No MLA, no shared experts
# ---------------------------------------------------------------------------

# Per-layer tensors (same for all 94 layers)
LAYER_TENSORS = [
    # Attention
    ((512, 4096), torch.bfloat16),     # self_attn.qkv_proj.weight
    ((4096, 256), torch.bfloat16),     # self_attn.o_proj.weight
    ((128,), torch.bfloat16),          # self_attn.q_norm.weight
    ((128,), torch.bfloat16),          # self_attn.k_norm.weight
    # MoE MLP
    ((4, 3072, 4096), torch.bfloat16), # mlp.experts.w13_weight
    ((4, 4096, 1536), torch.bfloat16), # mlp.experts.w2_weight
    ((128, 4096), torch.bfloat16),     # mlp.gate.weight
    # LayerNorms
    ((4096,), torch.bfloat16),         # input_layernorm.weight
    ((4096,), torch.bfloat16),         # post_attention_layernorm.weight
]

NUM_LAYERS = 94


def create_model_tensors():
    """Create 849 tensors matching the exact model replica layout.

    Layout: embed(1) + 94 layers * 9 tensors + norm(1) + lm_head(1) = 849
    Total: ~13.94 GB (14,962,875,392 bytes)
    """
    tensors = []
    total_bytes = 0

    def alloc(shape, dtype):
        return torch.empty(shape, dtype=dtype, device="cuda")

    # model.embed_tokens.weight
    tensors.append(alloc((4748, 4096), torch.bfloat16))

    # Layers 0-93: 9 tensors each
    for _ in range(NUM_LAYERS):
        for shape, dtype in LAYER_TENSORS:
            tensors.append(alloc(shape, dtype))

    # model.norm.weight
    tensors.append(alloc((4096,), torch.bfloat16))

    # lm_head.weight
    tensors.append(alloc((4748, 4096), torch.bfloat16))

    for t in tensors:
        total_bytes += t.numel() * t.element_size()

    return tensors, total_bytes


def get_merged_blocks(tensors: list[torch.Tensor]) -> list[tuple[int, int]]:
    """Merge contiguous memory blocks for a set of tensors."""
    addr_set = {t.data_ptr() for t in tensors}

    memory_snapshot = torch.cuda.memory.memory_snapshot()
    merged_blocks: list[tuple[int, int]] = []
    for segment in memory_snapshot:
        current_block = None
        for block in segment.get("blocks", []):
            address = block.get("address", -1)
            size = block.get("size", -1)
            state = block.get("state", "")
            if address < 0 or size < 0 or state == "":
                continue
            if state == "active_allocated" and address in addr_set:
                if current_block is None:
                    current_block = (address, size)
                elif current_block[0] + current_block[1] == address:
                    current_block = (current_block[0], current_block[1] + size)
                else:
                    merged_blocks.append(current_block)
                    current_block = (address, size)
        if current_block is not None:
            merged_blocks.append(current_block)

    return merged_blocks


# ---------------------------------------------------------------------------
#  Registration helpers
# ---------------------------------------------------------------------------

def register_per_tensor(tensors: list[torch.Tensor], engine: TransferEngine):
    addrs = [t.data_ptr() for t in tensors]
    sizes = [t.numel() * t.element_size() for t in tensors]
    ret = engine.batch_register_memory(addrs, sizes)
    if ret != 0:
        raise RuntimeError(f"batch_register_memory failed, error: {ret}")


def register_merged(blocks: list[tuple[int, int]], engine: TransferEngine):
    addrs = [addr for addr, _ in blocks]
    sizes = [size for _, size in blocks]
    ret = engine.batch_register_memory(addrs, sizes)
    if ret != 0:
        raise RuntimeError(f"batch_register_memory failed, error: {ret}")


def unregister_tensors(tensors: list[torch.Tensor], engine: TransferEngine):
    addrs = [t.data_ptr() for t in tensors]
    engine.batch_unregister_memory(addrs)


def unregister_blocks(blocks: list[tuple[int, int]], engine: TransferEngine):
    addrs = [addr for addr, _ in blocks]
    engine.batch_unregister_memory(addrs)


# ---------------------------------------------------------------------------
#  Per-GPU worker
# ---------------------------------------------------------------------------

def gpu_worker(gpu_id: int, barrier: mp.Barrier,
               results_dict: dict, local_ip: str):
    """Worker function that runs on a single GPU."""
    torch.cuda.set_device(gpu_id)
    tag = f"[GPU {gpu_id}]"

    # Allocate tensors
    tensors, total_bytes = create_model_tensors()
    print(f"{tag} Allocated {len(tensors)} tensors, "
          f"{total_bytes:,} bytes ({total_bytes / 1024**3:.2f} GB)")

    # Init TransferEngine (each GPU needs its own engine instance)
    engine = TransferEngine()
    engine.initialize(local_ip, "P2PHANDSHAKE", "rdma", "")
    print(f"{tag} TransferEngine ready, rpc_port={engine.get_rpc_port()}")

    # Compute merged blocks
    merged_blocks = get_merged_blocks(tensors)
    print(f"{tag} {len(tensors)} tensors -> {len(merged_blocks)} merged blocks")

    # Warmup
    register_per_tensor(tensors, engine)
    unregister_tensors(tensors, engine)
    print(f"{tag} Warmup done")

    # ---- Benchmark: per-tensor ----
    barrier.wait()  # sync all GPUs before timing
    t0 = time.perf_counter()
    register_per_tensor(tensors, engine)
    t1 = time.perf_counter()
    per_tensor_time = t1 - t0
    print(f"{tag} per-tensor ({len(tensors)} buffers): {per_tensor_time:.4f}s")
    unregister_tensors(tensors, engine)

    # ---- Benchmark: merged ----
    barrier.wait()  # sync all GPUs before timing
    t0 = time.perf_counter()
    register_merged(merged_blocks, engine)
    t1 = time.perf_counter()
    merged_time = t1 - t0
    print(f"{tag} merged ({len(merged_blocks)} buffers): {merged_time:.4f}s")
    unregister_blocks(merged_blocks, engine)

    # Store results
    results_dict[gpu_id] = {
        "per_tensor_time": per_tensor_time,
        "merged_time": merged_time,
        "num_tensors": len(tensors),
        "num_merged_blocks": len(merged_blocks),
        "total_bytes": total_bytes,
    }

    # Cleanup
    del tensors
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark batch_register_memory with real model tensor shapes (multi-GPU)"
    )
    parser.add_argument("--num-gpus", type=int, default=8,
                        help="Number of GPUs to use (default: 8)")
    args = parser.parse_args()

    num_gpus = min(args.num_gpus, torch.cuda.device_count())
    print(f"Running benchmark on {num_gpus} GPUs "
          f"(~13.94 GB per GPU, ~{13.94 * num_gpus:.1f} GB total)")

    local_ip = ray._private.services.get_node_ip_address()
    print(f"Local IP: {local_ip}")

    mp.set_start_method("spawn", force=True)
    barrier = mp.Barrier(num_gpus)
    manager = mp.Manager()
    results_dict = manager.dict()

    # Spawn one process per GPU
    processes = []
    for gpu_id in range(num_gpus):
        p = mp.Process(
            target=gpu_worker,
            args=(gpu_id, barrier, results_dict, local_ip),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # ---- Summary ----
    print("\n" + "=" * 80)
    print(f"{'GPU':<6}  {'Tensors':>8}  {'Merged':>8}  "
          f"{'Per-tensor (s)':>15}  {'Merged (s)':>12}  {'Speedup':>8}")
    print("-" * 80)

    total_per_tensor = 0.0
    total_merged = 0.0
    for gpu_id in sorted(results_dict.keys()):
        r = results_dict[gpu_id]
        speedup = r["per_tensor_time"] / r["merged_time"] if r["merged_time"] > 0 else float("inf")
        total_per_tensor = max(total_per_tensor, r["per_tensor_time"])
        total_merged = max(total_merged, r["merged_time"])
        print(f"  {gpu_id:<4}  {r['num_tensors']:>8}  {r['num_merged_blocks']:>8}  "
              f"{r['per_tensor_time']:>15.4f}  {r['merged_time']:>12.4f}  {speedup:>7.2f}x")

    print("-" * 80)
    overall_speedup = total_per_tensor / total_merged if total_merged > 0 else float("inf")
    total_gb = sum(results_dict[g]["total_bytes"] for g in results_dict) / 1024**3
    print(f"  Max across {num_gpus} GPUs:          "
          f"{total_per_tensor:>15.4f}  {total_merged:>12.4f}  {overall_speedup:>7.2f}x")
    print(f"  Total memory registered: {total_gb:.2f} GB")
    print("=" * 80)


if __name__ == "__main__":
    main()
