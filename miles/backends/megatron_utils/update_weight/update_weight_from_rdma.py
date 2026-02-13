import dataclasses
import logging
import threading
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
from sglang.srt.model_loader.remote_instance_weight_loader_utils import register_memory_region_v2
from sglang.srt.server_args import ServerArgs
from tqdm import tqdm

from miles.utils.memory_utils import print_memory
from miles.utils.timer import timer

from .update_weight_from_remote import UpdateWeightFromRemote

logger = logging.getLogger(__name__)


def create_server_args_from_dict(data_dict: dict) -> ServerArgs:
    # Reconstruct Sglang ServerArgs from sglang Http query.
    valid_fields = {f.name for f in dataclasses.fields(ServerArgs)}
    filtered_data = {k: v for k, v in data_dict.items() if k in valid_fields}
    return ServerArgs(**filtered_data)


@dataclasses.dataclass
class RemoteWeightInfo:
    # Remote session and weight registration info.
    session_id: str
    weights_info: dict[str, tuple[int, int, int]]  # name -> (remote_address, numel, element_size)


@dataclasses.dataclass
class TransferTask:
    """Represents a transfer task for the threadpool."""

    bundle: "TransferBundle"
    names: list[str]


class StreamingTransferManager:
    """
    Manages streaming RDMA transfers with parallel registration.

    - Registration runs in background thread parallel to main thread's all-gather
    - Transfers stream to threadpool as tensors become ready
    - Uses blocking sync transfers (simpler, threadpool provides parallelism)
    - Batch deregistration at end for efficiency
    """

    def __init__(self, num_workers: int = 4):
        self.num_workers = num_workers
        self.executor: ThreadPoolExecutor | None = None
        self.registration_complete = threading.Event()
        self.pending_queue: list[TransferTask] = []
        self.queue_lock = threading.Lock()
        self.transfer_futures: list[Future] = []
        self.reg_thread: threading.Thread | None = None
        self._bundles: list[TransferBundle] = []

    def start_registration(self, bundles: list["TransferBundle"]) -> None:
        """
        Start batch registration in background thread - call at start of update cycle.
        Runs parallel to main thread's all-gather operations.
        """
        self._bundles = bundles
        self.registration_complete.clear()
        self.pending_queue.clear()
        self.transfer_futures.clear()
        self.executor = ThreadPoolExecutor(max_workers=self.num_workers)

        def do_registration():
            try:
                with timer("rdma_batch_registration"):
                    with ThreadPoolExecutor(max_workers=len(bundles)) as reg_pool:
                        futures = [reg_pool.submit(register_memory_region_v2, b.model_replica, b.engine) for b in bundles]
                        for bundle, future in zip(bundles, futures):
                            bundle.weight_memory_registry, bundle.registered_blocks = future.result()
                        logger.info(f"[RDMA] Registered {len(bundle.weight_memory_registry)} tensors for engine rank")
            except Exception as e:
                logger.error(f"[RDMA] Registration failed: {e}")
                raise
            finally:
                with self.queue_lock:
                    self.registration_complete.set()
                    for task in self.pending_queue:
                        future = self.executor.submit(self._do_transfer, task.bundle, task.names)
                        self.transfer_futures.append(future)
                    pending_count = len(self.pending_queue)
                    self.pending_queue.clear()
                    logger.info(f"[RDMA] Registration complete, drained {pending_count} pending tasks")

        self.reg_thread = threading.Thread(target=do_registration, daemon=True)
        self.reg_thread.start()
        logger.info("[RDMA] Started background registration thread")

    def submit_for_transfer(self, bundle: "TransferBundle", names: list[str]) -> None:
        """
        Called after load_weights + cuda sync for a batch of tensors.
        Streams to threadpool immediately if registration done, else queues.
        """
        if not names:
            return

        with self.queue_lock:
            if self.registration_complete.is_set():
                # Registration done - submit to threadpool immediately
                future = self.executor.submit(self._do_transfer, bundle, names)
                self.transfer_futures.append(future)
            else:
                # Registration still running - queue for later
                self.pending_queue.append(TransferTask(bundle=bundle, names=names))

    def _do_transfer(self, bundle: "TransferBundle", names: list[str]) -> None:
        """Blocking transfer - runs in threadpool worker."""
        # Build source pointers and lengths
        source_ptrs, source_lens = [], []
        for name in names:
            tensor_register = bundle.weight_memory_registry.get(name)
            if tensor_register is None:
                logger.warning(f"[RDMA] Parameter {name} not found in weight registry")
                continue
            data_ptr, numel, ele_size = tensor_register
            source_ptrs.append(data_ptr)
            source_lens.append(numel * ele_size)

        if not source_ptrs:
            return

        # Transfer to each remote session
        for remote_session in bundle.remote_weight_infos:
            session_id = remote_session.session_id
            remote_weights_info = remote_session.weights_info

            target_ptrs = []
            for name in names:
                if name in remote_weights_info:
                    target_ptrs.append(remote_weights_info[name][0])  # remote address

            if len(target_ptrs) != len(source_ptrs):
                logger.warning(f"[RDMA] Pointer count mismatch for session {session_id}")
                continue

            # Use blocking sync write
            ret = bundle.engine.batch_transfer_sync_write(session_id, source_ptrs, target_ptrs, source_lens)
            if ret < 0:
                logger.error(f"[RDMA] Transfer failed for session {session_id}, error: {ret}")

    def wait_and_cleanup(self) -> None:
        """Wait for all transfers to complete, then batch deregister."""
        # Ensure registration thread finished
        if self.reg_thread is not None:
            self.reg_thread.join(timeout=60.0)
            if self.reg_thread.is_alive():
                logger.error("[RDMA] Registration thread did not complete in time")

        # Wait for all transfer futures
        for future in self.transfer_futures:
            try:
                future.result(timeout=30.0)
            except Exception as e:
                logger.error(f"[RDMA] Transfer future failed: {e}")

        self.transfer_futures.clear()

        # Batch deregister all bundles
        with timer("rdma_batch_deregistration"):
            for bundle in self._bundles:
                if bundle.registered_blocks:
                    ptrs = [addr for addr, _ in bundle.registered_blocks]
                    bundle.engine.batch_unregister_memory(ptrs)
                    logger.info(f"[RDMA] Batch unregistered {len(ptrs)} memory blocks")
                    bundle.registered_blocks = []

        # Shutdown executor
        if self.executor is not None:
            self.executor.shutdown(wait=False)
            self.executor = None

        # Reset state
        self.reg_thread = None
        self._bundles = []

    def reset(self) -> None:
        """Reset for next update cycle."""
        self.registration_complete.clear()
        self.pending_queue.clear()
        self.transfer_futures.clear()


@dataclasses.dataclass
class TransferBundle:
    model_replica: Sequence[torch.nn.Module]
    engine: TransferEngine
    remote_weight_infos: list[RemoteWeightInfo]
    param_mapper: ParameterMapper
    # Weight memory registry: name -> (data_ptr, numel, element_size)
    weight_memory_registry: dict = dataclasses.field(default_factory=dict)
    # Registered address after merge (list of (address, size) tuples)
    registered_blocks: list = dataclasses.field(default_factory=list)
    _offloaded: bool = False
    _cached_params_dict: dict = dataclasses.field(default_factory=dict)
    # Local buffer to check for parameter readiness before transfer
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

    def get_transfer_ready_params(self, converted_named_tensors: list[tuple[str, torch.Tensor]]) -> list[str]:
        """
        Track which parameters are ready for transfer.
        A parameter is ready when all its shards have been loaded.
        """
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

    def execute_all(self) -> None:
        """Execute transfer for all parameters - used in non-pipelined mode."""
        for remote_session in self.remote_weight_infos:
            session_id, remote_weights_info = remote_session.session_id, remote_session.weights_info
            source_ptrs, target_ptrs, source_lens = [], [], []
            for name, tensor_register in self.weight_memory_registry.items():
                data_ptr, numel, ele_size = tensor_register
                source_ptrs.append(data_ptr)
                target_ptrs.append(remote_weights_info[name][0])  # remote address
                source_lens.append(numel * ele_size)

            # Batch transfer weights through RDMA
            ret = self.engine.batch_transfer_sync_write(session_id, source_ptrs, target_ptrs, source_lens)
            if ret < 0:
                raise RuntimeError(f"Batch transfer weights via RDMA failed with error code {ret}.")


class UpdateWeightFromRDMA(UpdateWeightFromRemote):
    """
    Update weights from RDMA using Transfer Engine.

    Similar to UpdateWeightFromNCCL but uses P2P RDMA transfer engine for the underlying weight transfer. Workflow
    consists of following steps:
    1. Based off the transfer plan, query the target rollout engines for remote session and weight info during connect_rollout_engines.
    2. Construct local model replica according to the plan and attach target session id and weight memory registry
    2. Do TP-EP all-gather for bucketed weights on parameters needing transfer from local just as in NCCL case.
    3. Convert the gathered HF tensor into target shape and register them with Engine.
    4. Call engine to batch transfer weights for each transfer task.
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

        self._offloaded = False
        self.pipelined_transfer = args.rdma_pipelined_transfer
        self.persistent_registration = getattr(args, "rdma_persistent_registration", False)
        self._registered = False  # Track whether persistent registration has been done

        # Initialize streaming transfer manager for pipelined transfers
        num_workers = getattr(args, "rdma_transfer_workers", 4)
        self.transfer_manager = StreamingTransferManager(num_workers=num_workers)

    def connect_rollout_engines(
        self, rollout_engines: Sequence[ActorHandle], rollout_engine_lock: ActorHandle
    ) -> None:
        """
        Initialize P2PTrainingTransferEngine if serves as a source.
        """
        # Store rollout engines and lock
        self.rollout_engines = rollout_engines
        self.rollout_engine_lock = rollout_engine_lock

        if self._is_source:
            # Query Engine session and weight info from rollout instances according to the transfer plan
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
                assert (
                    session_id is not None
                ), f"Failed to get session id from rollout engine {engine_ind} rank {engine_rank}"
                logger.info(
                    f"[RDMA] Obtained remote {session_id} info from rollout engine {engine_ind} rank {engine_rank}"
                )
                logger.info(f"[RDMA] Remote weight info has {len(weights_info)} tensors.")
                # logger.info(list(weights_info.keys()))
                self.remote_weight_infos_by_session_id[session_id] = (weights_info, parallelism_info)
                targets_to_session_id[(engine_ind, engine_rank)] = session_id

            print_memory("[RDMA] After obtaining remote weight info")

            # Create local model replicas and transfer engines for each target rollout shard
            self.engines = {}
            # Associate transfer tasks based on obtained session and weight info
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
                    # Note: Registration is deferred to on_transfer_start() for pipelining with all-gather
                    self.engines[target.engine_rank] = TransferBundle(
                        model_replica=model_replica,
                        engine=transfer_engine,
                        remote_weight_infos=[remote_info],
                        param_mapper=param_mapper,
                    )
                else:
                    self.engines[target.engine_rank].add_remote_session(remote_info)

            print_memory("[RDMA] After Local Engine Replicas and engine Creation")

    def _register_replica_memory(self, model_replica, remote_weight_info, transfer_engine):
        # Verify the 1-to-1 mapping between local replica and remote weights expected.
        for name, tensor in model_replica.named_parameters():
            if name not in remote_weight_info:
                raise RuntimeError(f"Local replica parameter {name} not found in remote replica.")
            remote_numel, remote_ele_size = remote_weight_info[name][1], remote_weight_info[name][2]
            if tensor.numel() != remote_numel or tensor.element_size() != remote_ele_size:
                raise RuntimeError(
                    f"Local replica parameter {name} numel {tensor.numel()} size {tensor.element_size()} does not match remote numel {remote_numel} size {remote_ele_size}."
                )
            if tensor.device.type != "cuda":
                raise RuntimeError(f"Local replica parameter {name} is not on CUDA device.")
        weight_memory_registry, registered_blocks = register_memory_region_v2(model_replica, transfer_engine)

        logger.info(
            f"[RDMA] Registered {len(list(model_replica.named_parameters()))} tensors from replica with transfer engine."
        )
        return weight_memory_registry, registered_blocks

    def _unregister_replica_memory(self, registered_blocks, transfer_engine):
        weight_blocks = []
        for address, _ in registered_blocks:
            weight_blocks.append(address)

        transfer_engine.batch_unregister_memory(weight_blocks)
        logger.info("[RDMA] Unregistered tensors from replica with transfer engine.")
        return

    def _create_transfer_engine(self) -> TransferEngine:
        transfer_engine = TransferEngine()
        local_ip = ray._private.services.get_node_ip_address()
        transfer_engine.initialize(local_ip, "P2PHANDSHAKE", "rdma", "")

        logger.info(f"[RDMA] Local replica Transfer Engine initialized at port {transfer_engine.get_rpc_port()}")
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
        # Update weight version as we were write-only.
        ray.get(
            [
                engine.update_weight_version.remote(weight_version=str(self.weight_version))
                for engine in self.rollout_engines
            ]
        )
        return

    def on_transfer_start(self) -> None:
        """
        Hook called at start of weight transfer cycle.
        Re-onloads model replicas if offloaded and starts background registration.
        Registration runs parallel to main thread's all-gather operations.

        With persistent_registration=True, registration is done once and kept across
        iterations — skipping re-onload, re-registration, and later deregistration/offload.
        """
        if not self._is_source:
            return

        # Persistent registration: skip if already registered from a previous iteration
        if self.persistent_registration and self._registered:
            logger.info("[RDMA] Persistent registration: skipping re-registration (already registered)")
            return

        bundles_to_register = []
        for transfer_bundle in self.engines.values():
            if transfer_bundle._offloaded:
                # Re-onload model replica - resize storage back to original size
                for weight in transfer_bundle.model_replica.parameters():
                    weight.untyped_storage().resize_(weight.numel() * weight.element_size())
                transfer_bundle._offloaded = False
                logger.info("[RDMA] Re-onloaded model replica from offloaded state")

            bundles_to_register.append(transfer_bundle)

        if self.pipelined_transfer and bundles_to_register:
            # Start registration in background thread - runs parallel to all-gather
            self.transfer_manager.start_registration(bundles_to_register)
        else:
            # Non-pipelined: register synchronously
            for bundle in bundles_to_register:
                bundle.weight_memory_registry, bundle.registered_blocks = register_memory_region_v2(
                    bundle.model_replica, bundle.engine
                )
                logger.info(f"[RDMA] Registered {len(bundle.weight_memory_registry)} tensors synchronously")

        if self.persistent_registration:
            self._registered = True

    def _update_bucket_weights_from_remote(
        self, converted_named_tensors: list[tuple[str, torch.Tensor]], pbar: tqdm | None = None
    ) -> None:
        """
        The RDMA P2P weight update is implemented as a single side write,
        meaning the trainer writes its weights directly to the rollout engines' memory.

        In pipelined mode:
        - Registration runs in background thread (started by on_transfer_start)
        - Transfers stream to threadpool as tensors become ready
        - If registration not done yet, transfers queue and drain when ready
        """
        if not self._is_source or not converted_named_tensors:
            return

        for transfer_bundle in self.engines.values():
            # Get list of parameters ready for transfer (all shards loaded)
            transfer_ready_params = transfer_bundle.get_transfer_ready_params(converted_named_tensors)

            # Load weights into model replica
            transfer_bundle.model_replica.load_weights(converted_named_tensors)

            if self.pipelined_transfer:
                # Ensure load_weights async CUDA copies are flushed to GPU memory before RDMA reads
                torch.cuda.synchronize()
                # Submit to streaming transfer manager (queues if registration not done)
                self.transfer_manager.submit_for_transfer(transfer_bundle, transfer_ready_params)

        converted_named_tensors.clear()

    def finish_transfer_task(self) -> None:
        """
        Complete all pending transfers and clean up.

        In pipelined mode:
        - Wait for registration thread to complete
        - Wait for all transfer futures in threadpool
        - Batch deregister all memory regions (unless persistent)
        - Offload model replicas (unless persistent)

        With persistent_registration=True, memory stays registered and replicas
        stay on GPU for the next iteration.
        """
        if not self._is_source:
            return

        if not self.pipelined_transfer:
            # Non-pipelined: execute all transfers synchronously
            for transfer_bundle in self.engines.values():
                transfer_bundle.execute_all()
            if not self.persistent_registration:
                # Deregister and offload
                for transfer_bundle in self.engines.values():
                    if transfer_bundle.registered_blocks:
                        self._unregister_replica_memory(transfer_bundle.registered_blocks, transfer_bundle.engine)
                        transfer_bundle.registered_blocks = []
        else:
            # Pipelined: wait for streaming transfers to complete and cleanup
            logger.info("[RDMA] Waiting for all streaming transfers to complete...")
            if self.persistent_registration:
                # Wait for transfers but skip deregistration
                self.transfer_manager.wait_transfers_only()
                logger.info("[RDMA] All transfers complete (persistent: keeping registration)")
            else:
                self.transfer_manager.wait_and_cleanup()
                logger.info("[RDMA] All transfers complete and memory deregistered")

            # Reset bundle state for next cycle
            for transfer_bundle in self.engines.values():
                transfer_bundle.reset()

        if not self.persistent_registration:
            # Offload model replicas from memory after transfer
            print_memory("[RDMA] Before offloading model replica")
            for transfer_bundle in self.engines.values():
                if not transfer_bundle._offloaded:
                    # Release GPU memory
                    for weight in transfer_bundle.model_replica.parameters():
                        weight.untyped_storage().resize_(0)
                    transfer_bundle._offloaded = True

            torch.cuda.empty_cache()
            print_memory("[RDMA] After offloading model replica")
        else:
            logger.info("[RDMA] Persistent registration: skipping offload (replicas stay on GPU)")

        # Reset transfer manager state for next cycle
        if self.pipelined_transfer and not self.persistent_registration:
            self.transfer_manager.reset()

        return
