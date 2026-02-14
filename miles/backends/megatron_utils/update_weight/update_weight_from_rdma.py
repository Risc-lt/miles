import dataclasses
import logging
import time
from argparse import Namespace
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor

import ray
import torch
from mooncake.engine import TransferEngine
from ray.actor import ActorHandle
from sglang.srt import server_args as server_args_module
from sglang.srt.configs.device_config import DeviceConfig
from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.distributed.parallel_state import ParallelismContext, RankParallelismConfig
from sglang.srt.model_loader import get_model
from sglang.srt.model_loader.parameter_mapper import ParameterMapper
from sglang.srt.server_args import ServerArgs
from tqdm import tqdm

from miles.utils.memory_utils import print_memory
from miles.utils.timer import timer

from .update_weight_from_remote import UpdateWeightFromRemote

logger = logging.getLogger(__name__)


def create_server_args_from_dict(data_dict: dict) -> ServerArgs:
    valid_fields = {f.name for f in dataclasses.fields(ServerArgs)}
    filtered_data = {k: v for k, v in data_dict.items() if k in valid_fields}
    return ServerArgs(**filtered_data)


def register_cpu_memory_region(cpu_params_dict: dict, transfer_engine: TransferEngine) -> dict:
    """Register CPU pinned memory with the transfer engine.

    Returns:
        dict: name -> (data_ptr, numel, element_size) for each registered tensor.
    """
    start_tic = time.time()
    weight_mr_dict = {}

    for name, cpu_tensor in cpu_params_dict.items():
        addr = cpu_tensor.data_ptr()
        size = cpu_tensor.numel() * cpu_tensor.element_size()
        ret = transfer_engine.register_memory(addr, size)
        if ret != 0:
            raise RuntimeError(f"register CPU memory failed for weight {name}, error: {ret}")
        weight_mr_dict[name] = (addr, cpu_tensor.numel(), cpu_tensor.element_size())

    elapsed = time.time() - start_tic
    logger.info(f"[RDMA] Registered {len(weight_mr_dict)} CPU tensors in {elapsed:.2f}s")
    return weight_mr_dict


@dataclasses.dataclass
class RemoteWeightInfo:
    session_id: str
    weights_info: dict[str, tuple[int, int, int]]  # name -> (remote_address, numel, element_size)


class StreamingTransferManager:
    """Manages pipelined RDMA transfers through CPU pinned memory.

    Pipeline per bucket:
      Main thread:  all-gather → load_weights(GPU) → cuda.sync → submit ──→ next bucket
      Background:                                                   ↘ D2H copy → RDMA write

    CPU registration is done synchronously on first iteration (fast, ~0.5-1s for host memory).
    Registration persists across iterations.
    """

    def __init__(self, num_workers: int = 4):
        self.num_workers = num_workers
        self.executor: ThreadPoolExecutor | None = None
        self.transfer_futures: list[Future] = []

    def ensure_started(self) -> None:
        """Lazily create the transfer threadpool."""
        if self.executor is None:
            self.executor = ThreadPoolExecutor(max_workers=self.num_workers)

    def submit_for_transfer(self, bundle: "TransferBundle", names: list[str]) -> None:
        """Submit a batch of ready tensors for D2H + RDMA transfer."""
        if not names:
            return
        self.ensure_started()
        future = self.executor.submit(self._do_transfer, bundle, names)
        self.transfer_futures.append(future)

    def _do_transfer(self, bundle: "TransferBundle", names: list[str]) -> None:
        """GPU→CPU copy then RDMA write from CPU pinned memory. Runs in threadpool."""
        gpu_params = bundle.params_dict
        cpu_params = bundle.cpu_params_dict
        source_ptrs, source_lens = [], []
        valid_names = []

        for name in names:
            if name not in cpu_params or name not in gpu_params:
                logger.warning(f"[RDMA] Parameter {name} not found in CPU/GPU params")
                continue

            # D2H copy: GPU replica → CPU pinned replica
            cpu_params[name].copy_(gpu_params[name])

            cpu_reg = bundle.weight_memory_registry.get(name)
            if cpu_reg is None:
                logger.warning(f"[RDMA] Parameter {name} not in CPU weight registry")
                continue

            data_ptr, numel, ele_size = cpu_reg
            source_ptrs.append(data_ptr)
            source_lens.append(numel * ele_size)
            valid_names.append(name)

        if not source_ptrs:
            return

        for remote_session in bundle.remote_weight_infos:
            session_id = remote_session.session_id
            remote_weights_info = remote_session.weights_info

            target_ptrs = []
            for name in valid_names:
                if name in remote_weights_info:
                    target_ptrs.append(remote_weights_info[name][0])

            if len(target_ptrs) != len(source_ptrs):
                logger.warning(f"[RDMA] Pointer count mismatch for session {session_id}")
                continue

            ret = bundle.engine.batch_transfer_sync_write(session_id, source_ptrs, target_ptrs, source_lens)
            if ret < 0:
                logger.error(f"[RDMA] Transfer failed for session {session_id}, error: {ret}")

    def wait_transfers(self) -> None:
        """Wait for all transfers to complete. CPU memory and registration persist."""
        for future in self.transfer_futures:
            try:
                future.result(timeout=30.0)
            except Exception as e:
                logger.error(f"[RDMA] Transfer future failed: {e}")

        self.transfer_futures.clear()


@dataclasses.dataclass
class TransferBundle:
    model_replica: torch.nn.Module
    engine: TransferEngine
    remote_weight_infos: list[RemoteWeightInfo]
    param_mapper: ParameterMapper
    # CPU pinned mirror of GPU replica parameters — RDMA source
    cpu_params_dict: dict = dataclasses.field(default_factory=dict)
    # CPU weight memory registry: name -> (data_ptr, numel, element_size)
    weight_memory_registry: dict = dataclasses.field(default_factory=dict)
    _offloaded: bool = False
    _cached_params_dict: dict = dataclasses.field(default_factory=dict)
    _update_pending: dict[str, int] = dataclasses.field(default_factory=dict)

    @property
    def params_dict(self):
        if not self._cached_params_dict:
            self._cached_params_dict = dict(self.model_replica.named_parameters())
        return self._cached_params_dict

    def reset(self):
        self._update_pending = {}

    def add_remote_session(self, remote_info: RemoteWeightInfo) -> None:
        self.remote_weight_infos.append(remote_info)

    def allocate_cpu_replica(self) -> None:
        """Allocate CPU pinned memory mirroring the GPU model replica."""
        self.cpu_params_dict = {}
        for name, param in self.model_replica.named_parameters():
            self.cpu_params_dict[name] = torch.empty_like(param, device="cpu").pin_memory()
        total_bytes = sum(t.numel() * t.element_size() for t in self.cpu_params_dict.values())
        logger.info(
            f"[RDMA] Allocated CPU pinned replica: {len(self.cpu_params_dict)} params, "
            f"{total_bytes / (1024**3):.2f} GB"
        )

    def get_transfer_ready_params(self, converted_named_tensors: list[tuple[str, torch.Tensor]]) -> list[str]:
        """Track which parameters are ready (all shards loaded)."""
        transfer_ready_params = []
        for name, _ in converted_named_tensors:
            mapped_result = self.param_mapper.map(name)
            mapped, num_shards, num_experts = (
                mapped_result.sglang_name,
                mapped_result.num_shards,
                mapped_result.num_local_experts,
            )
            if mapped not in self.params_dict:
                logger.warning(f"Parameter {mapped} not found in model replica.")
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


class UpdateWeightFromRDMA(UpdateWeightFromRemote):
    """RDMA weight transfer using CPU pinned memory as the transfer source.

    Architecture:
    - GPU model replica: used for load_weights (preserves all sglang model logic)
    - CPU pinned replica: registered with RDMA, used as transfer source
    - GPU replica is re-onloaded each iteration, offloaded after transfer
    - CPU replica + host MR registration persist across iterations (cheap)

    Per-iteration pipeline:
      on_transfer_start:  re-onload GPU replica (+ first-time CPU registration, ~0.5-1s)
      per bucket:         all-gather → load_weights(GPU) → sync → submit(D2H + RDMA)
      finish:             wait all transfers → offload GPU replica → training resumes
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
        num_workers = getattr(args, "rdma_transfer_workers", 4)
        self.transfer_manager = StreamingTransferManager(num_workers=num_workers)

    def connect_rollout_engines(
        self, rollout_engines: Sequence[ActorHandle], rollout_engine_lock: ActorHandle
    ) -> None:
        self.rollout_engines = rollout_engines
        self.rollout_engine_lock = rollout_engine_lock

        if self._is_source:
            self.remote_weight_infos_by_session_id = {}
            targets = self.transfer_plan.plan_p2p()
            targets_to_query = set((target.engine_ind, target.engine_rank) for target in targets)
            targets_to_session_id, self.session_id_to_engine_rank = {}, {}
            self.session_id_to_server_args = {}
            for engine_ind, engine_rank in targets_to_query:
                session_id, weights_info = ray.get(
                    self.rollout_engines[engine_ind].get_remote_instance_transfer_engine_info.remote(rank=engine_rank)
                )
                parallelism_info = ray.get(
                    self.rollout_engines[engine_ind].get_parallelism_info.remote(rank=engine_rank)
                )

                self.session_id_to_engine_rank[session_id] = engine_rank
                self.session_id_to_server_args[session_id] = create_server_args_from_dict(
                    ray.get(self.rollout_engines[engine_ind].get_server_info.remote())
                )
                assert session_id is not None, (
                    f"Failed to get session id from rollout engine {engine_ind} rank {engine_rank}"
                )
                logger.info(
                    f"[RDMA] Obtained remote {session_id} info from rollout engine {engine_ind} rank {engine_rank}"
                )
                logger.info(f"[RDMA] Remote weight info has {len(weights_info)} tensors.")
                self.remote_weight_infos_by_session_id[session_id] = (weights_info, parallelism_info)
                targets_to_session_id[(engine_ind, engine_rank)] = session_id

            print_memory("[RDMA] After obtaining remote weight info")

            self.engines = {}
            for target in targets:
                session_id = targets_to_session_id[(target.engine_ind, target.engine_rank)]
                remote_info = RemoteWeightInfo(session_id, self.remote_weight_infos_by_session_id[session_id][0])
                parallelism_config = RankParallelismConfig.from_dict(
                    self.remote_weight_infos_by_session_id[session_id][1]
                )
                if target.engine_rank not in self.engines:
                    transfer_engine = self._create_transfer_engine()
                    logger.info(f"[RDMA] Creating model replica for engine rank {target.engine_rank}")
                    model_replica = self._create_inference_replica(
                        parallelism_config, self.args.hf_checkpoint, self.session_id_to_server_args[session_id]
                    )
                    param_mapper = ParameterMapper.from_model(model_replica)
                    print_memory(f"[RDMA] After model replica at {target.engine_rank}")
                    bundle = TransferBundle(
                        model_replica=model_replica,
                        engine=transfer_engine,
                        remote_weight_infos=[remote_info],
                        param_mapper=param_mapper,
                    )
                    bundle.allocate_cpu_replica()
                    self.engines[target.engine_rank] = bundle
                else:
                    self.engines[target.engine_rank].add_remote_session(remote_info)

            print_memory("[RDMA] After Local Engine Replicas and engine Creation")

    def _create_transfer_engine(self) -> TransferEngine:
        transfer_engine = TransferEngine()
        local_ip = ray._private.services.get_node_ip_address()
        transfer_engine.initialize(local_ip, "P2PHANDSHAKE", "rdma", "")
        logger.info(f"[RDMA] Transfer Engine initialized at port {transfer_engine.get_rpc_port()}")
        return transfer_engine

    def _create_inference_replica(
        self,
        parallelism_config: RankParallelismConfig,
        model_path: str,
        server_args: ServerArgs,
    ):
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
        device = next(model.parameters()).device
        logger.info(f" Model {device}, params: {sum(p.numel() for p in model.parameters())} ")
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
        """Re-onload GPU replica for load_weights. First call also registers CPU memory."""
        if not self._is_source:
            return

        # Re-onload GPU replicas (offloaded at end of previous iteration)
        for transfer_bundle in self.engines.values():
            if transfer_bundle._offloaded:
                for weight in transfer_bundle.model_replica.parameters():
                    weight.untyped_storage().resize_(weight.numel() * weight.element_size())
                transfer_bundle._offloaded = False
                logger.info("[RDMA] Re-onloaded GPU replica for load_weights")

        if not self._registered:
            with timer("rdma_cpu_registration"):
                for bundle in self.engines.values():
                    bundle.weight_memory_registry = register_cpu_memory_region(
                        bundle.cpu_params_dict, bundle.engine
                    )
            self._registered = True

    def _update_bucket_weights_from_remote(
        self, converted_named_tensors: list[tuple[str, torch.Tensor]], pbar: tqdm | None = None
    ) -> None:
        """Load weights into GPU replica, submit D2H + RDMA to background threads.

        Critical path: all-gather → load_weights(GPU) → cuda.sync → submit → next bucket
        Background:                                                    ↘ D2H copy → RDMA write
        """
        if not self._is_source or not converted_named_tensors:
            return

        for transfer_bundle in self.engines.values():
            transfer_ready_params = transfer_bundle.get_transfer_ready_params(converted_named_tensors)
            transfer_bundle.model_replica.load_weights(converted_named_tensors)
            torch.cuda.synchronize()
            self.transfer_manager.submit_for_transfer(transfer_bundle, transfer_ready_params)

        converted_named_tensors.clear()

    def finish_transfer_task(self) -> None:
        """Wait for all D2H + RDMA transfers, then offload GPU replica."""
        if not self._is_source:
            return

        logger.info("[RDMA] Waiting for all D2H + RDMA transfers to complete...")
        self.transfer_manager.wait_transfers()
        logger.info("[RDMA] All transfers complete")

        for transfer_bundle in self.engines.values():
            transfer_bundle.reset()

        # Offload GPU replica — CPU replica + registration persist
        print_memory("[RDMA] Before offloading GPU replica")
        for transfer_bundle in self.engines.values():
            if not transfer_bundle._offloaded:
                for weight in transfer_bundle.model_replica.parameters():
                    weight.untyped_storage().resize_(0)
                transfer_bundle._offloaded = True
        torch.cuda.empty_cache()
        print_memory("[RDMA] After offloading GPU replica")
