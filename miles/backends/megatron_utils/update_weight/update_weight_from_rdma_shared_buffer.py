import dataclasses
import logging
from argparse import Namespace
from collections.abc import Callable, Mapping, Sequence

import ray
import torch
from ray.actor import ActorHandle
from sglang.srt.distributed.parallel_state import ParallelismContext, RankParallelismConfig
from sglang.srt.model_loader.parameter_mapper import ParameterMapper
from tqdm import tqdm

from miles.utils.memory_utils import print_memory
from miles.utils.timer import timer

from .update_weight_from_rdma import (
    RDMATransferManager,
    RemoteWeightInfo,
    create_cpu_replica,
    create_transfer_engine,
    query_remote_weight_infos,
    register_cpu_memory_region,
)
from .update_weight_from_remote import UpdateWeightFromRemote

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class EngineRankInfo:
    """Per-engine-rank metadata: unique model replica (weight_loaders), shared CPU pinned buffers."""

    engine_rank: int
    model_replica: torch.nn.Module  # shares CPU pinned buffers, has unique weight_loaders
    remote_weight_infos: list[RemoteWeightInfo]

    def add_remote_session(self, remote_info: RemoteWeightInfo) -> None:
        self.remote_weight_infos.append(remote_info)


class UpdateWeightFromRDMASharedBuffer(UpdateWeightFromRemote):
    """RDMA weight transfer using a single set of shared CPU pinned buffers.

    Architecture:
    - One set of CPU pinned buffers shared across all engine ranks (saves ~N× CPU memory)
    - One TransferEngine for all engine ranks
    - Lightweight replicas per engine rank share the pinned storage but have unique
      weight_loaders (so load_weights slices the correct shard for each rank)
    - Transfers are serialized across engine ranks because the shared buffer is
      overwritten for each rank's shard, EXCEPT the last engine rank's write which
      runs in a background thread (overlaps with the next bucket's all-gather)

    Per-bucket flow:
      1. compute transfer_ready_params once (same for all engine ranks)
      2. for each engine rank:
           load_weights(shared buffer) → RDMA write
         where the last rank's write is submitted to a background thread
      3. wait_transfers() at finish to collect all background writes
    """

    def __init__(
        self,
        args: Namespace,
        model: Sequence[torch.nn.Module],
        weights_getter: Callable[[], Mapping[str, torch.Tensor]],
        *,
        model_name: str,
        quantization_config: dict[str, int | str | list[str]] | None,
    ) -> None:
        super().__init__(
            args,
            model,
            weights_getter,
            model_name=model_name,
            quantization_config=quantization_config,
            weight_update_mode="rdma",
        )

        self._registered = False
        self._update_pending: dict[str, int] = {}
        num_workers = getattr(args, "rdma_transfer_workers", 4)
        self.transfer_manager = RDMATransferManager(num_workers=num_workers)

    def connect_rollout_engines(
        self, rollout_engines: Sequence[ActorHandle], rollout_engine_lock: ActorHandle
    ) -> None:
        self.rollout_engines = rollout_engines
        self.rollout_engine_lock = rollout_engine_lock

        if self._is_source:
            targets = self.transfer_plan.plan_p2p()
            (
                self.remote_weight_infos_by_session_id,
                targets_to_session_id,
                self.session_id_to_server_args,
            ) = query_remote_weight_infos(rollout_engines, targets)

            print_memory("[RDMA-Shared] After obtaining remote weight info")

            # Group targets by engine rank
            engine_rank_targets: dict[int, list] = {}
            for target in targets:
                engine_rank_targets.setdefault(target.engine_rank, []).append(target)

            # Create ONE transfer engine for all engine ranks
            self._engine = create_transfer_engine()
            self._engine_rank_infos: dict[int, EngineRankInfo] = {}
            self._shared_params_dict: dict[str, torch.Tensor] = {}
            self._shared_param_mapper: ParameterMapper | None = None

            first_engine_rank = True
            for engine_rank, rank_targets in engine_rank_targets.items():
                # Get parallelism config from first target of this engine rank
                first_target = rank_targets[0]
                session_id = targets_to_session_id[(first_target.engine_ind, first_target.engine_rank)]
                parallelism_config = RankParallelismConfig.from_dict(
                    self.remote_weight_infos_by_session_id[session_id][1]
                )
                server_args = self.session_id_to_server_args[session_id]

                if first_engine_rank:
                    # First engine rank: create full CPU pinned replica → shared buffers
                    logger.info(f"[RDMA-Shared] Creating shared CPU pinned replica from engine rank {engine_rank}")
                    model_replica = create_cpu_replica(
                        parallelism_config, self.args.hf_checkpoint, server_args
                    )
                    self._shared_params_dict = dict(model_replica.named_parameters())
                    self._shared_param_mapper = ParameterMapper.from_model(model_replica)
                    print_memory(f"[RDMA-Shared] After shared CPU pinned replica for engine rank {engine_rank}")
                    first_engine_rank = False
                else:
                    # Subsequent engine ranks: create lightweight replica sharing pinned buffers
                    logger.info(f"[RDMA-Shared] Creating lightweight replica for engine rank {engine_rank}")
                    model_replica = self._create_lightweight_replica(
                        parallelism_config, self.args.hf_checkpoint, server_args
                    )
                    print_memory(f"[RDMA-Shared] After lightweight replica for engine rank {engine_rank}")

                # Collect remote sessions for this engine rank
                remote_infos = []
                for target in rank_targets:
                    sid = targets_to_session_id[(target.engine_ind, target.engine_rank)]
                    remote_infos.append(
                        RemoteWeightInfo(sid, self.remote_weight_infos_by_session_id[sid][0])
                    )

                self._engine_rank_infos[engine_rank] = EngineRankInfo(
                    engine_rank=engine_rank,
                    model_replica=model_replica,
                    remote_weight_infos=remote_infos,
                )

            # Materialize as list for indexed access (last-rank detection)
            self._engine_rank_list = list(self._engine_rank_infos.values())

            print_memory("[RDMA-Shared] After all replicas and engine creation")

    def _create_lightweight_replica(
        self,
        parallelism_config: RankParallelismConfig,
        model_path: str,
        server_args,
    ) -> torch.nn.Module:
        """Create model on GPU (different weight_loaders), then point params to shared CPU pinned buffers.

        The model object (layers, weight_loaders) stays alive but shares the underlying
        storage with the first replica's CPU pinned buffers. No new CPU allocation.
        """
        from sglang.srt import server_args as server_args_module
        from sglang.srt.configs.device_config import DeviceConfig
        from sglang.srt.configs.load_config import LoadConfig
        from sglang.srt.configs.model_config import ModelConfig
        from sglang.srt.model_loader import get_model

        load_config = LoadConfig(
            load_format="auto",
            model_loader_extra_config=server_args.model_loader_extra_config,
            rl_quant_profile=server_args.rl_quant_profile,
        )
        server_args_module._global_server_args = server_args
        with ParallelismContext(parallelism_config):
            model = get_model(
                model_config=ModelConfig(model_path),
                load_config=load_config,
                device_config=DeviceConfig(),
            )

        gpu_params = sum(p.numel() for p in model.parameters())
        logger.info(f"[RDMA-Shared] Lightweight replica GPU model created: {gpu_params} params")

        # Point all params to shared pinned buffers (no new CPU allocation)
        for name, param in model.named_parameters():
            if name not in self._shared_params_dict:
                logger.warning(
                    f"[RDMA-Shared] Parameter {name} not found in shared buffers, skipping"
                )
                continue
            param.data = self._shared_params_dict[name]

        torch.cuda.empty_cache()
        logger.info(
            f"[RDMA-Shared] Lightweight replica: {gpu_params} params, "
            f"sharing {len(self._shared_params_dict)} CPU pinned buffers"
        )
        print_memory("[RDMA-Shared] After lightweight replica (shared buffers, GPU freed)")
        return model

    def leader_post_update(self) -> None:
        ray.get([engine.continue_generation.remote() for engine in self.rollout_engines])
        ray.get(
            [
                engine.update_weight_version.remote(weight_version=str(self.weight_version))
                for engine in self.rollout_engines
            ]
        )

    def on_transfer_start(self) -> None:
        """Register shared CPU pinned memory with RDMA on first call."""
        if not self._is_source:
            return

        if not self._registered:
            with timer("rdma_cpu_registration"):
                self._weight_memory_registry = register_cpu_memory_region(
                    self._shared_params_dict, self._engine
                )
            self._registered = True

    def _update_bucket_weights_from_remote(
        self, converted_named_tensors: list[tuple[str, torch.Tensor]], pbar: tqdm | None = None
    ) -> None:
        """Load weights into shared buffer, RDMA write per engine rank.

        For ranks 0..N-2: load_weights → synchronous RDMA write (must complete
        before the next load_weights overwrites the shared buffer).
        For the last rank: load_weights → submit to background thread (safe because
        nothing touches the buffer again until the next bucket, and the RDMA write
        reads from pinned memory that won't be overwritten until then).
        """
        if not self._is_source or not converted_named_tensors:
            return

        # Compute transfer-ready params once (same mapper, same bucket, same result)
        with timer("get_transfer_ready_params_shared", log_info=False):
            transfer_ready_params = self._get_transfer_ready_params(converted_named_tensors)

        # For each engine rank: load_weights → RDMA write
        last_idx = len(self._engine_rank_list) - 1
        for i, info in enumerate(self._engine_rank_list):
            with timer("load_weights_to_shared_buffer", log_info=False):
                info.model_replica.load_weights(converted_named_tensors)

            if transfer_ready_params:
                is_last = i == last_idx
                if is_last:
                    # Last engine rank: submit to background thread.
                    with timer("rdma_async_write_shared", log_info=False):
                        self.transfer_manager.submit(
                            self._do_rdma_write, info, transfer_ready_params
                        )
                else:
                    # Not the last rank: synchronous write.
                    # Must complete before the next load_weights() overwrites the buffer.
                    with timer("rdma_sync_write_shared", log_info=False):
                        self._do_rdma_write(info, transfer_ready_params)

        # Clear AFTER all engine ranks are processed
        converted_named_tensors.clear()

    def _get_transfer_ready_params(
        self, converted_named_tensors: list[tuple[str, torch.Tensor]]
    ) -> list[str]:
        """Determine which sglang params have all shards present so far.

        Uses the shared ParameterMapper — the result is engine-rank-independent
        since the mapping is a property of model architecture, not parallelism layout.

        Uses self._update_pending (persistent across bucket calls) to correctly
        track multi-shard parameters whose shards span multiple buckets (e.g.,
        MoE expert weights with EP > 1).
        """
        transfer_ready_params = []
        params_dict = self._shared_params_dict

        for name, _ in converted_named_tensors:
            mapped_result = self._shared_param_mapper.map(name)
            mapped, num_shards, num_experts = (
                mapped_result.sglang_name,
                mapped_result.num_shards,
                mapped_result.num_local_experts,
            )
            if mapped not in params_dict:
                logger.warning(f"Parameter {mapped} not found in shared model replica.")
                continue

            if num_experts is not None and num_experts > 0:
                total_expected = num_experts * num_shards
            else:
                total_expected = num_shards

            if total_expected == 1:
                transfer_ready_params.append(mapped)
            else:
                if mapped not in self._update_pending:
                    self._update_pending[mapped] = total_expected - 1
                else:
                    self._update_pending[mapped] -= 1
                if self._update_pending[mapped] == 0:
                    transfer_ready_params.append(mapped)

        return transfer_ready_params

    def _do_rdma_write(self, info: EngineRankInfo, names: list[str]) -> None:
        """RDMA write from shared CPU pinned buffers to remote GPUs.

        Called synchronously for all engine ranks except the last,
        and in a background thread for the last engine rank.
        """
        source_ptrs, source_lens = [], []
        valid_names = []

        for name in names:
            cpu_reg = self._weight_memory_registry.get(name)
            if cpu_reg is None:
                logger.warning(f"[RDMA-Shared] Parameter {name} not in shared weight registry")
                continue

            data_ptr, numel, ele_size = cpu_reg
            source_ptrs.append(data_ptr)
            source_lens.append(numel * ele_size)
            valid_names.append(name)

        if not source_ptrs:
            return

        for remote_session in info.remote_weight_infos:
            session_id = remote_session.session_id
            remote_weights_info = remote_session.weights_info

            target_ptrs = []
            for name in valid_names:
                if name in remote_weights_info:
                    target_ptrs.append(remote_weights_info[name][0])

            if len(target_ptrs) != len(source_ptrs):
                logger.warning(f"[RDMA-Shared] Pointer count mismatch for session {session_id}")
                continue

            ret = self._engine.batch_transfer_sync_write(session_id, source_ptrs, target_ptrs, source_lens)
            if ret < 0:
                logger.error(f"[RDMA-Shared] Transfer failed for session {session_id}, error: {ret}")

    def finish_transfer_task(self) -> None:
        """Wait for all background RDMA writes to complete."""
        if not self._is_source:
            return
        self.transfer_manager.wait_transfers()
        self._update_pending = {}
        logger.info("[RDMA-Shared] All transfers complete")
