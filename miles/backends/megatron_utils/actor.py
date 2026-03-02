import logging
import os
import random
import socket
from argparse import Namespace
from contextlib import nullcontext

import ray
import torch
import torch.distributed as dist
from megatron.core import mpu
from ray.actor import ActorHandle
from torch_memory_saver import torch_memory_saver
from transformers import AutoConfig, AutoTokenizer

from miles.ray.train_actor import TrainRayActor
from miles.utils import train_dump_utils
from miles.utils.context_utils import with_defer
from miles.utils.distributed_utils import get_gloo_group, init_process_group
from miles.utils.memory_utils import clear_memory, print_memory
from miles.utils.ray_utils import Box
from miles.utils.reloadable_process_group import destroy_process_groups, monkey_patch_torch_dist, reload_process_groups
from miles.utils.routing_replay import RoutingReplay
from miles.utils.timer import Timer, inverse_timer, timer
from miles.utils.tracking_utils import init_tracking
from miles.utils.types import RolloutBatch

from ...utils.profile_utils import TrainProfiler
from ...utils.tensor_backper import TensorBackuper
from ..training_utils.cp_utils import slice_with_cp
from ..training_utils.data import DataIterator, get_data_iterator, get_rollout_data, sync_actor_critic_data
from ..training_utils.log_utils import log_perf_data, log_rollout_data
from ..training_utils.loss import compute_advantages_and_returns, get_log_probs_and_entropy, get_values
from .checkpoint import load_checkpoint
from .initialize import init, is_megatron_main_rank
from .model import forward_only, initialize_model_and_optimizer, save, train
from .parallel import create_megatron_parallel_state
from .update_weight.common import named_params_and_buffers
from .update_weight.update_weight_from_distributed import UpdateWeightFromDistributed
from .update_weight.update_weight_from_rdma import UpdateWeightFromRDMA
from .update_weight.update_weight_from_rdma_shared_buffer import UpdateWeightFromRDMASharedBuffer
from .update_weight.update_weight_from_tensor import UpdateWeightFromTensor

logging.getLogger("megatron").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


class MegatronTrainRayActor(TrainRayActor):
    @with_defer(lambda: Timer().start("train_wait"))
    def init(
        self,
        args: Namespace,
        role: str,
        with_ref: bool = False,
    ) -> int | None:
        monkey_patch_torch_dist()

        super().init(args, role, with_ref)

        init(args)

        if is_megatron_main_rank():
            init_tracking(args, primary=False)

        self.prof = TrainProfiler(args)

        # read config and tokenizer serialized to prevent concurrent writing bug.
        for i in range(dist.get_world_size()):
            if i == dist.get_rank():
                self.hf_config = AutoConfig.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
                self.tokenizer = AutoTokenizer.from_pretrained(self.args.hf_checkpoint, trust_remote_code=True)
            dist.barrier(group=get_gloo_group())

        self.train_parallel_config = {
            "dp_size": mpu.get_data_parallel_world_size(with_context_parallel=False),
        }
        dist.barrier(group=get_gloo_group())

        if args.offload_train:
            if (x := args.train_memory_margin_bytes) > 0:
                logger.info(f"Set torch_memory_saver.memory_margin_bytes to {x}")
                torch_memory_saver.memory_margin_bytes = x

        if self.args.debug_rollout_only:
            return 0

        if role == "critic":
            self.args.load = self.args.critic_load
            self.args.save = self.args.critic_save
            self.args.lr = self.args.critic_lr
            self.args.lr_warmup_iters = self.args.critic_lr_warmup_iters

        (self.model, self.optimizer, self.opt_param_scheduler, loaded_rollout_id) = initialize_model_and_optimizer(
            args, role
        )

        self.parallel_state = create_megatron_parallel_state(model=self.model)

        if role == "critic":
            if self.args.offload_train:
                self.sleep()
            return

        start_rollout_id = loaded_rollout_id + 1

        self.weights_backuper = TensorBackuper.create(
            source_getter=lambda: named_params_and_buffers(
                self.args,
                self.model,
                convert_to_global_name=args.megatron_to_hf_mode == "raw",
                translate_gpu_to_cpu=not self.args.enable_weights_backuper,
            ),
            single_tag=None if args.enable_weights_backuper else "actor",
        )
        self._active_model_tag: str | None = "actor"
        self.weights_backuper.backup("actor")

        if with_ref:
            self.load_other_checkpoint("ref", args.ref_load)

        if self.args.keep_old_actor:
            # Load old_actor checkpoint
            self.load_other_checkpoint("old_actor", args.load)
            # Create rollout_actor as a copy of current actor
            if args.update_weights_interval == 1:
                self.weights_backuper.backup("rollout_actor")

        if self.args.vocab_size is None:
            self.args.vocab_size = self.tokenizer.vocab_size
        if self.args.colocate:
            update_weight_cls = UpdateWeightFromTensor
        else:
            if self.args.update_weight_transfer_mode == "nccl":
                update_weight_cls = UpdateWeightFromDistributed
            elif getattr(self.args, "rdma_shared_buffer", False):
                update_weight_cls = UpdateWeightFromRDMASharedBuffer
            else:
                update_weight_cls = UpdateWeightFromRDMA
        self.weight_updater = update_weight_cls(
            self.args,
            self.model,
            weights_getter=lambda: self.weights_backuper.get("actor"),
            model_name=type(self.hf_config).__name__.lower() if self.args.model_name is None else self.args.model_name,
            quantization_config=getattr(self.hf_config, "quantization_config", None),
        )

        # empty cache after initialization
        clear_memory()

        # Warmup NCCL communicators to avoid lazy init timeout during first forward pass
        self._warmup_nccl_communicators()

        # Pre-initialize DeepEP buffer to avoid lazy nvshmem init blocking during pipeline forward
        self._warmup_deepep_buffer()

        if self.args.offload_train:
            # recover to actor in the end.
            self._switch_model("actor")
            self.sleep()

        self.rollout_engines = None

        self.rollout_data_postprocess = None
        if self.args.rollout_data_postprocess_path is not None:
            from miles.utils.misc import load_function

            self.rollout_data_postprocess = load_function(self.args.rollout_data_postprocess_path)

        self.prof.on_init_end()

        return start_rollout_id

    def _warmup_nccl_communicators(self):
        """Force eager NCCL communicator creation for PP P2P to avoid lazy init timeout.

        Megatron's pipeline P2P uses three communication patterns that each create
        separate NCCL sub-communicators under lazy initialization:
        1. dist.send/recv on pp_group (unbatched P2P)
        2. batch_isend_irecv on pp_group (batched P2P, used by _communicate_shapes)
        3. dist.isend/irecv on WORLD group (used by _p2p_ops when pp_group.size()==2)

        All three must be warmed up to avoid 600s timeout during the first forward pass.
        """
        pp_size = mpu.get_pipeline_model_parallel_world_size()
        if pp_size <= 1:
            return

        pp_rank = mpu.get_pipeline_model_parallel_rank()
        pp_group = mpu.get_pipeline_model_parallel_group()
        prev_rank = mpu.get_pipeline_model_parallel_prev_rank()
        next_rank = mpu.get_pipeline_model_parallel_next_rank()

        dummy = torch.zeros(1, device=torch.cuda.current_device())

        # --- Pattern 1: unbatched dist.send/recv on pp_group ---
        # Forward direction
        if pp_rank > 0:
            dist.recv(dummy, src=prev_rank, group=pp_group)
        if pp_rank < pp_size - 1:
            dist.send(dummy, dst=next_rank, group=pp_group)
        # Backward direction
        if pp_rank < pp_size - 1:
            dist.recv(dummy, src=next_rank, group=pp_group)
        if pp_rank > 0:
            dist.send(dummy, dst=prev_rank, group=pp_group)

        # --- Pattern 2: batch_isend_irecv on pp_group ---
        # This is what _communicate_shapes uses (variable_seq_lengths=True path).
        # batch_isend_irecv creates different NCCL sub-communicators than send/recv.
        dummy_shape = torch.zeros(3, device=torch.cuda.current_device(), dtype=torch.int64)

        # Forward: recv from prev, send to next
        ops = []
        if pp_rank > 0:
            ops.append(dist.P2POp(dist.irecv, dummy_shape.clone(), prev_rank, pp_group))
        if pp_rank < pp_size - 1:
            ops.append(dist.P2POp(dist.isend, dummy_shape.clone(), next_rank, pp_group))
        if ops:
            reqs = dist.batch_isend_irecv(ops)
            for req in reqs:
                req.wait()
        torch.cuda.synchronize()

        # Backward: recv from next, send to prev
        ops = []
        if pp_rank < pp_size - 1:
            ops.append(dist.P2POp(dist.irecv, dummy_shape.clone(), next_rank, pp_group))
        if pp_rank > 0:
            ops.append(dist.P2POp(dist.isend, dummy_shape.clone(), prev_rank, pp_group))
        if ops:
            reqs = dist.batch_isend_irecv(ops)
            for req in reqs:
                req.wait()
        torch.cuda.synchronize()

        # --- Pattern 3: isend/irecv on WORLD group ---
        # _p2p_ops uses WORLD group when pp_group.size() == 2 for overlap.
        # Even when pp_group.size() > 2, warm up WORLD group P2P as a safety measure
        # since some code paths (e.g. _batched_p2p_ops -> recv on default_pg) may use it.
        prev_global = dist.get_global_rank(pp_group, (pp_group.rank() - 1) % pp_group.size())
        next_global = dist.get_global_rank(pp_group, (pp_group.rank() + 1) % pp_group.size())
        dummy_world = torch.zeros(1, device=torch.cuda.current_device())

        # Forward on WORLD
        ops = []
        if pp_rank > 0:
            ops.append(dist.P2POp(dist.irecv, dummy_world.clone(), prev_global))
        if pp_rank < pp_size - 1:
            ops.append(dist.P2POp(dist.isend, dummy_world.clone(), next_global))
        if ops:
            reqs = dist.batch_isend_irecv(ops)
            for req in reqs:
                req.wait()
        torch.cuda.synchronize()

        # Backward on WORLD
        ops = []
        if pp_rank < pp_size - 1:
            ops.append(dist.P2POp(dist.irecv, dummy_world.clone(), next_global))
        if pp_rank > 0:
            ops.append(dist.P2POp(dist.isend, dummy_world.clone(), prev_global))
        if ops:
            reqs = dist.batch_isend_irecv(ops)
            for req in reqs:
                req.wait()
        torch.cuda.synchronize()

        # --- CP group warmup ---
        cp_size = mpu.get_context_parallel_world_size()
        if cp_size > 1:
            cp_group = mpu.get_context_parallel_group()
            dummy_cp = torch.zeros(1, device=torch.cuda.current_device())
            dist.all_reduce(dummy_cp, group=cp_group)

        dist.barrier()
        logger.info(f"[NCCL Warmup] PP={pp_size}, CP={cp_size} communicators warmed up "
                     f"(unbatched + batched P2P on pp_group + WORLD group)")

    def _warmup_deepep_buffer(self):
        """Pre-initialize DeepEP Buffer (nvshmem) to avoid lazy init during pipeline forward.

        DeepEP's Buffer() constructor calls nvshmem_init/nvshmem_team_split internally,
        which are collective operations requiring all ranks in the EP group to participate.
        In the pipeline schedule, different PP stages enter forward_step at different times.
        If Buffer() is lazily initialized inside the first forward_step, the nvshmem collective
        can block until all EP group members arrive — but in a pipeline, some ranks may be
        waiting on P2P recv from an earlier stage that is itself blocked on nvshmem init,
        causing a deadlock-like timeout.

        This method pre-initializes the buffer before the pipeline schedule begins,
        ensuring all ranks have completed nvshmem setup.
        """
        if not getattr(self.args, 'moe_enable_deepep', False):
            return

        try:
            from megatron.core.transformer.moe.fused_a2a import get_buffer, HAVE_DEEP_EP
            if not HAVE_DEEP_EP:
                return
        except ImportError:
            return

        # Get the TP_EP group that flex dispatcher actually uses for fused_dispatch.
        # FlexMoETokenDispatcher passes pg_collection.tp_ep to _DeepepManager,
        # which is get_expert_tensor_and_model_parallel_group().
        # We MUST use the same group object here, otherwise get_buffer() will see
        # _buffer.group != group and re-create the Buffer during forward.
        ep_group = None
        try:
            ep_group = mpu.get_expert_tensor_and_model_parallel_group()
        except Exception:
            try:
                ep_group = mpu.get_expert_model_parallel_group()
            except Exception:
                pass

        if ep_group is None:
            return

        # Compute hidden_bytes matching what forward_step will use:
        # hidden_size * max(element_size_bf16, 2) = hidden_size * 2
        hidden_bytes = self.args.hidden_size * 2  # bf16

        logger.info(f"[DeepEP Warmup] Pre-initializing DeepEP Buffer "
                    f"(ep_size={ep_group.size()}, hidden_bytes={hidden_bytes})")

        # This call triggers Buffer(group, nvl_bytes, rdma_bytes) which internally
        # does nvshmem init — a collective across all EP group members.
        # Since we call this outside the pipeline schedule, all ranks are synchronized.
        get_buffer(ep_group, hidden_bytes)

        dist.barrier()
        logger.info(f"[DeepEP Warmup] DeepEP Buffer initialized successfully")

    @timer
    def sleep(self) -> None:
        assert self.args.offload_train

        clear_memory(clear_host_memory=True)
        print_memory("before offload model")
        destroy_process_groups()

        torch_memory_saver.pause()

        print_memory("after offload model")

    @timer
    def wake_up(self) -> None:
        assert self.args.offload_train
        print_memory("before wake_up model")

        torch_memory_saver.resume()

        clear_memory()
        reload_process_groups()
        print_memory("after wake_up model")

    def _switch_model(self, target_tag: str) -> None:
        if target_tag not in self.weights_backuper.backup_tags:
            raise ValueError(f"Cannot switch to unknown model tag: {target_tag}")
        self.weights_backuper.restore(target_tag)
        self._active_model_tag = target_tag

    def fill_routing_replay(self, data_iterator, num_microbatches, rollout_data):
        if "rollout_routed_experts" not in rollout_data:
            raise ValueError(
                "rollout_routed_experts is required in rollout_data when use_rollout_routing_replay is set."
            )

        from megatron.core.transformer.transformer_block import get_num_layers_to_build
        from megatron.core.transformer.transformer_layer import get_transformer_layer_offset

        from miles.utils.routing_replay import RoutingReplay

        for iterator in data_iterator:
            iterator.reset()

        tp_rank = self.parallel_state.tp_rank
        tp_size = self.parallel_state.tp_size

        def pad_func(experts, pad):
            _, num_layers, topk = experts.shape
            pad = (
                torch.arange(
                    pad * num_layers * topk,
                    device=experts.device,
                    dtype=experts.dtype,
                ).reshape((pad, num_layers, topk))
                % self.args.num_experts
            )
            return torch.cat([experts, pad], dim=0)

        for _ in range(sum(num_microbatches)):
            batch = data_iterator[0].get_next(["rollout_routed_experts", "tokens"])
            rollout_routed_experts = batch["rollout_routed_experts"]
            tokens = batch["tokens"]
            assert len(rollout_routed_experts) == len(tokens)
            for a, b in zip(rollout_routed_experts, tokens, strict=False):
                assert a.shape[0] == b.shape[0] - 1, f"{a.shape}, {b.shape}"

            # We need to pad the experts to the last token. We won't calculate loss on this token so this should be fine.
            # TODO: fuse this padding with the following slice_with_cp to reduce memory copy.
            rollout_routed_experts = [pad_func(r, 1) for r in rollout_routed_experts]
            # TODO: maybe extract a common process function for here and get_batch?
            rollout_routed_experts = [slice_with_cp(r, pad_func, self.parallel_state) for r in rollout_routed_experts]
            rollout_routed_experts = torch.cat(rollout_routed_experts, dim=0)
            pad_size = self.parallel_state.dp_size * self.args.data_pad_size_multiplier
            pad = (pad_size - rollout_routed_experts.size(0) % pad_size) % pad_size
            if pad != 0:
                rollout_routed_experts = pad_func(rollout_routed_experts, pad)

            if self.args.sequence_parallel:
                seqlen = rollout_routed_experts.size(0)
                assert seqlen % tp_size == 0
                start, end = seqlen // tp_size * tp_rank, seqlen // tp_size * (tp_rank + 1)
                rollout_routed_experts = rollout_routed_experts[start:end]

            routing_replay_offset = 0
            for vp_stage, model in enumerate(self.model):
                config = model.module.config
                num_layers_to_build = get_num_layers_to_build(config, vp_stage=vp_stage)
                offset = get_transformer_layer_offset(config, vp_stage=vp_stage)
                for layer_id in range(offset, offset + num_layers_to_build):
                    # skip dense layer
                    if isinstance(config.moe_layer_freq, int):
                        if layer_id % config.moe_layer_freq != 0:
                            continue
                    elif isinstance(config.moe_layer_freq, list):
                        assert len(config.moe_layer_freq) == config.num_layers
                        if config.moe_layer_freq[layer_id] == 0:
                            continue
                    layer_routed_experts = rollout_routed_experts[:, layer_id]
                    RoutingReplay.all_routing_replays[routing_replay_offset].record(layer_routed_experts)
                    routing_replay_offset += 1
            assert routing_replay_offset == len(RoutingReplay.all_routing_replays)

        del rollout_data["rollout_routed_experts"]

        for iterator in data_iterator:
            iterator.reset()

    def compute_log_prob(
        self,
        data_iterator: list[DataIterator],
        num_microbatches: list[int],
        store_prefix: str = "",
    ) -> dict[str, list[torch.Tensor]]:

        with timer(f"{store_prefix}log_probs"):
            return forward_only(
                get_log_probs_and_entropy,
                self.args,
                self.model,
                data_iterator,
                num_microbatches,
                self.parallel_state,
                store_prefix=store_prefix,
            )

    def train(self, rollout_id: int, rollout_data_ref: Box) -> None:
        if self.args.offload_train:
            self.wake_up()

        with timer("data_preprocess"):
            rollout_data = get_rollout_data(self.args, rollout_data_ref, self.parallel_state)
            if self.args.debug_rollout_only:
                log_rollout_data(rollout_id, self.args, rollout_data, self.parallel_state)
                return

        if self.role == "critic":
            return self.train_critic(rollout_id, rollout_data)
        else:
            return self.train_actor(rollout_id, rollout_data)

    def train_critic(self, rollout_id: int, rollout_data: RolloutBatch) -> None:
        # Create data iterator for log_probs and train.
        data_iterator, num_microbatches = get_data_iterator(self.args, self.model, self.parallel_state, rollout_data)
        rollout_data.update(
            forward_only(
                get_values,
                self.args,
                self.model,
                data_iterator,
                num_microbatches,
                self.parallel_state,
            )
        )

        if rollout_id >= self.args.num_critic_only_steps:
            sync_actor_critic_data(self.args, rollout_data, self._actor_critic_groups)

        compute_advantages_and_returns(self.args, self.parallel_state, rollout_data)

        self.args.loss_type = "value_loss"
        train(
            rollout_id,
            self.model,
            self.optimizer,
            self.opt_param_scheduler,
            data_iterator,
            num_microbatches,
            self.parallel_state,
        )

    def train_actor(self, rollout_id: int, rollout_data: RolloutBatch) -> None:
        # Create data iterator for log_probs and train.
        data_iterator, num_microbatches = get_data_iterator(self.args, self.model, self.parallel_state, rollout_data)

        if self.args.use_rollout_routing_replay:
            self.fill_routing_replay(data_iterator, num_microbatches, rollout_data)

        with inverse_timer("train_wait"), timer("train"):
            if self.args.compute_advantages_and_returns:
                if "ref" in self.weights_backuper.backup_tags:
                    if self.args.use_routing_replay:
                        os.environ["ROUTING_REPLAY_STAGE"] = "fallthrough"
                    self._switch_model("ref")
                    rollout_data.update(
                        self.compute_log_prob(
                            data_iterator,
                            num_microbatches,
                            store_prefix="ref_",
                        )
                    )
                self._switch_model("old_actor" if self.args.keep_old_actor else "actor")
                if not self.args.use_rollout_logprobs or self.args.get_mismatch_metrics:
                    if self.args.use_routing_replay:
                        if self.args.use_rollout_routing_replay:
                            os.environ["ROUTING_REPLAY_STAGE"] = "replay_forward"
                        else:
                            os.environ["ROUTING_REPLAY_STAGE"] = "record"
                    rollout_data.update(
                        self.compute_log_prob(
                            data_iterator,
                            num_microbatches,
                            store_prefix="",
                        )
                    )
                    if self.args.use_rollout_routing_replay:
                        RoutingReplay.clear_all_forward()

                if self.args.use_critic:
                    sync_actor_critic_data(
                        self.args,
                        rollout_data,
                        self._actor_critic_groups,
                    )
                if self._active_model_tag != "actor":
                    self._switch_model("actor")

                # Calculate adv and returns. Need to performed before training (instead of on the fly),
                # because we may need normalize the whole rollout.
                compute_advantages_and_returns(self.args, self.parallel_state, rollout_data)

            if self.rollout_data_postprocess is not None:
                self.rollout_data_postprocess(self.args)

            log_rollout_data(rollout_id, self.args, rollout_data, self.parallel_state)

            # Train
            if self.args.use_routing_replay:
                os.environ["ROUTING_REPLAY_STAGE"] = "replay_backward"
            with timer("actor_train"):
                train(
                    rollout_id,
                    self.model,
                    self.optimizer,
                    self.opt_param_scheduler,
                    data_iterator,
                    num_microbatches,
                    self.parallel_state,
                )

            self.prof.step(rollout_id=rollout_id)

        train_dump_utils.save_debug_train_data(self.args, rollout_id=rollout_id, rollout_data=rollout_data)

        if self.args.use_routing_replay:
            RoutingReplay.clear_all()

        # update the cpu actor weight to the latest model
        self.weights_backuper.backup("actor")

        # Update ref model if needed
        if (
            self.args.ref_update_interval is not None
            and (rollout_id + 1) % self.args.ref_update_interval == 0
            and "ref" in self.weights_backuper.backup_tags
        ):
            with timer("ref_model_update"):
                if is_megatron_main_rank():
                    logger.info(f"Updating ref model at rollout_id {rollout_id}")
                self.weights_backuper.backup("ref")

        log_perf_data(rollout_id, self.args, self.parallel_state)

    @timer
    def save_model(self, rollout_id: int, force_sync: bool = False) -> None:
        if self.args.debug_rollout_only:
            return

        # torch dist may trigger nccl communication during saving.
        if self.args.offload_train:
            reload_process_groups()

        if self.args.async_save:
            from megatron.training.async_utils import maybe_finalize_async_save

            maybe_finalize_async_save(blocking=True)

        save(rollout_id, self.model, self.optimizer, self.opt_param_scheduler)

        if force_sync and self.args.async_save:
            maybe_finalize_async_save(blocking=True)

        if self.args.save_hf is not None and self.role == "actor":
            from miles.backends.megatron_utils.model import save_hf_model

            save_hf_model(self.args, rollout_id, self.model)

        if self.args.offload_train:
            destroy_process_groups()

    @timer
    def update_weights(self) -> None:
        if self.args.debug_train_only or self.args.debug_rollout_only:
            return

        if self.args.use_fault_tolerance:
            if dist.get_rank() == 0:
                ray.get(self.rollout_manager.recover_rollout_engines.remote())
            dist.barrier(group=get_gloo_group())

        rollout_engines, rollout_engine_lock, num_new_engines = ray.get(
            self.rollout_manager.get_rollout_engines_and_lock.remote()
        )

        if self.args.offload_train:
            reload_process_groups()

        if num_new_engines > 0:
            self.weight_updater.connect_rollout_engines(rollout_engines, rollout_engine_lock)
            dist.barrier(group=get_gloo_group())
            if dist.get_rank() == 0:
                ray.get(self.rollout_manager.clear_num_new_engines.remote())

        with torch_memory_saver.disable() if self.args.offload_train else nullcontext():
            print_memory("before update_weights")
            self.weight_updater.update_weights()
            print_memory("after update_weights")

            if self.args.ci_test and len(rollout_engines) > 0:
                engine = random.choice(rollout_engines)
                engine_version = ray.get(engine.get_weight_version.remote())
                if str(engine_version) != str(self.weight_updater.weight_version):
                    raise RuntimeError(
                        f"Weight version mismatch! Engine: {engine_version}, Updater: {self.weight_updater.weight_version}"
                    )

            if getattr(self.args, "keep_old_actor", False):
                if self.args.update_weights_interval == 1:
                    logger.info("updating model queue: rollout_actor -> old_actor, actor -> rollout_actor")
                    # Queue-style update: rollout_actor params -> old_actor, actor params -> rollout_actor
                    # First copy rollout_actor to old_actor
                    self.weights_backuper.copy(src_tag="rollout_actor", dst_tag="old_actor")
                    # Then copy current actor to rollout_actor
                    self.weights_backuper.backup("rollout_actor")
                else:
                    self.weights_backuper.backup("old_actor")

        if self.args.offload_train:
            destroy_process_groups()

    def load_other_checkpoint(self, model_tag: str, path: str) -> None:
        old_args = self.args.load, self.args.no_load_optim, self.args.no_load_rng, self.args.finetune
        self.args.load = path
        self.args.no_load_optim = True
        self.args.no_load_rng = True
        self.args.finetune = True

        if model_tag == "ref" and self.args.ref_ckpt_step is not None:
            old_ckpt_step = self.args.ckpt_step
            self.args.ckpt_step = self.args.ref_ckpt_step

        _, _ = load_checkpoint(
            self.model,
            None,
            None,
            checkpointing_context={},
            skip_load_to_model_and_opt=False,
        )
        self.args.load, self.args.no_load_optim, self.args.no_load_rng, self.args.finetune = old_args

        if model_tag == "ref" and self.args.ref_ckpt_step is not None:
            self.args.ckpt_step = old_ckpt_step

        self.weights_backuper.backup(model_tag)
        self._active_model_tag = model_tag

    def connect_actor_critic(
        self,
        actor_handle: ActorHandle | None = None,
        master_address: str | None = None,
        master_port: int | None = None,
    ) -> None:
        if self.role == "actor":
            master_address = ray.util.get_node_ip_address()
            with socket.socket() as sock:
                sock.bind(("", 0))
                master_port = sock.getsockname()[1]
            actor_handle.connect_actor_critic.remote(master_address=master_address, master_port=master_port)

        group_name = "actor_critic"
        world_size = 2
        self._actor_critic_groups = init_process_group(
            backend="nccl",
            init_method=f"tcp://{master_address}:{master_port}",
            world_size=world_size,
            rank=0 if self.role == "actor" else 1,
            group_name=group_name,
        )
