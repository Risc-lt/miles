import dataclasses
import gc
import logging
import math
import os
from argparse import Namespace
from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path

import torch
from megatron.core import mpu
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.distributed import finalize_model_grads
from megatron.core.enums import ModelType
from megatron.core.models.gpt import GPTModel
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.optimizer.optimizer import MegatronOptimizer
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.core.utils import get_model_config
from megatron.training.global_vars import get_args
from megatron.training.training import get_model

from miles.utils.memory_utils import clear_memory

from ..training_utils.ci_utils import check_grad_norm, check_kl
from ..training_utils.data import DataIterator, get_batch
from ..training_utils.log_utils import aggregate_forward_results, aggregate_train_losses, log_train_step
from ..training_utils.loss import loss_function
from ..training_utils.parallel import ParallelState
from .checkpoint import load_checkpoint, save_checkpoint
from .model_provider import get_model_provider_func
from .parallel import get_packed_seq_params

logger = logging.getLogger(__name__)


def get_optimizer_param_scheduler(args: Namespace, optimizer: MegatronOptimizer) -> OptimizerParamScheduler:
    """Create and configure the optimizer learning-rate/weight-decay scheduler.

    This configures iteration-based schedules derived from the global batch size
    and run-time arguments.

    Args:
        args (Namespace): Training/runtime arguments (argparse namespace).
        optimizer (MegatronOptimizer): Megatron optimizer bound to the model.

    Returns:
        OptimizerParamScheduler: Initialized scheduler bound to ``optimizer``.
    """
    # Iteration-based training.
    args.train_iters = args.num_rollout * args.rollout_batch_size * args.n_samples_per_prompt // args.global_batch_size
    if args.lr_decay_iters is None:
        args.lr_decay_iters = args.train_iters
    lr_decay_steps = args.lr_decay_iters * args.global_batch_size
    wd_incr_steps = args.train_iters * args.global_batch_size
    wsd_decay_steps = None
    if args.lr_wsd_decay_iters is not None:
        wsd_decay_steps = args.lr_wsd_decay_iters * args.global_batch_size
    if args.lr_warmup_fraction is not None:
        lr_warmup_steps = args.lr_warmup_fraction * lr_decay_steps
    else:
        lr_warmup_steps = args.lr_warmup_iters * args.global_batch_size

    opt_param_scheduler = OptimizerParamScheduler(
        optimizer,
        init_lr=args.lr_warmup_init,
        max_lr=args.lr,
        min_lr=args.min_lr,
        lr_warmup_steps=lr_warmup_steps,
        lr_decay_steps=lr_decay_steps,
        lr_decay_style=args.lr_decay_style,
        start_wd=args.start_weight_decay,
        end_wd=args.end_weight_decay,
        wd_incr_steps=wd_incr_steps,
        wd_incr_style=args.weight_decay_incr_style,
        use_checkpoint_opt_param_scheduler=args.use_checkpoint_opt_param_scheduler,
        override_opt_param_scheduler=args.override_opt_param_scheduler,
        wsd_decay_steps=wsd_decay_steps,
        lr_wsd_decay_style=args.lr_wsd_decay_style,
    )

    return opt_param_scheduler


def setup_model_and_optimizer(
    args: Namespace,
    role: str = "actor",
) -> tuple[list[DDP], MegatronOptimizer, OptimizerParamScheduler]:
    """Build model(s), wrap with DDP, and construct optimizer and scheduler.

    Args:
        args (Namespace): Training/runtime arguments (argparse namespace).
        role (str): Logical role of the model (e.g., "actor", "critic").
        no_wd_decay_cond (Callable[..., bool] | None): Predicate to exclude
            parameters from weight decay.
        scale_lr_cond (Callable[..., bool] | None): Predicate to scale LR for
            selected parameter groups.
        lr_mult (float): Global learning-rate multiplier for the optimizer.

    Returns:
        tuple[list[DDP], MegatronOptimizer, OptimizerParamScheduler]:
            - List of model chunks wrapped by ``DDP``.
            - The constructed ``MegatronOptimizer`` instance.
            - The learning-rate/weight-decay scheduler tied to the optimizer.
    """
    assert not args.moe_use_upcycling
    assert args.load is not None or args.pretrained_checkpoint is not None

    model = get_model(get_model_provider_func(args, role), ModelType.encoder_or_decoder)

    # Optimizer
    kwargs = {}
    for f in dataclasses.fields(OptimizerConfig):
        if hasattr(args, f.name):
            kwargs[f.name] = getattr(args, f.name)
    config = OptimizerConfig(**kwargs)
    config.timers = None

    optimizer = get_megatron_optimizer(
        config=config,
        model_chunks=model,
        use_gloo_process_groups=args.enable_gloo_process_groups,
    )
    opt_param_scheduler = get_optimizer_param_scheduler(args, optimizer)
    return model, optimizer, opt_param_scheduler


def enable_forward_pre_hook(model_chunks: Sequence[DDP]) -> None:
    """Enable forward pre-hooks for provided DDP-wrapped model chunks.

    Args:
        model_chunks (Sequence[DDP]): Sequence of DDP modules to enable hooks on.
    """
    for model_chunk in model_chunks:
        assert isinstance(model_chunk, DDP)
        model_chunk.enable_forward_pre_hook()


def disable_forward_pre_hook(model_chunks: Sequence[DDP], param_sync: bool = True) -> None:
    """Disable forward pre-hooks for provided DDP-wrapped model chunks.

    Args:
        model_chunks (Sequence[DDP]): Sequence of DDP modules to disable hooks on.
        param_sync (bool): Whether to synchronize parameters when disabling.
    """
    for model_chunk in model_chunks:
        assert isinstance(model_chunk, DDP)
        model_chunk.disable_forward_pre_hook(param_sync=param_sync)


@torch.no_grad()
def forward_only(
    f: Callable[..., dict[str, list[torch.Tensor]]],
    args: Namespace,
    model: Sequence[DDP],
    data_iterator: Sequence[DataIterator],
    num_microbatches: Sequence[int],
    parallel_state: ParallelState,
    store_prefix: str = "",
) -> dict[str, list[torch.Tensor]]:
    """Run forward passes only and collect non-loss outputs (e.g., logprobs).

    The model is put into evaluation mode, a forward-only pipeline pass is
    executed, and relevant outputs are aggregated and returned.

    Args:
        f (Callable[..., dict[str, list[torch.Tensor]]]): Post-forward callback used to
            compute and package outputs to collect. This should accept a logits
            tensor as its first positional argument and additional keyword-only
            arguments; see ``get_log_probs_and_entropy``/``get_values`` in
            ``megatron_utils.loss`` for examples. It will be partially applied
            so that the callable returned from the internal forward step only
            requires the logits tensor.
        args (Namespace): Runtime arguments.
        model (Sequence[DDP]): Sequence of DDP-wrapped model chunks.
        data_iterator (Sequence[DataIterator]): Iterable(s) yielding batches for inference.
        num_microbatches (Sequence[int]): Number of microbatches per rollout step.
        store_prefix (str): Prefix to prepend to stored output keys.

    Returns:
        dict[str, list[torch.Tensor]]: Aggregated outputs keyed by ``store_prefix + key``.
    """

    # reset data iterator
    for iterator in data_iterator:
        iterator.reset()

    config = get_model_config(model[0])

    def forward_step(
        data_iterator: DataIterator, model: GPTModel, return_schedule_plan: bool = False
    ) -> tuple[torch.Tensor, Callable[[torch.Tensor], dict[str, list[torch.Tensor]]]]:
        """Forward step used by Megatron's pipeline engine.

        Args:
            data_iterator (DataIterator): Input data iterator.
            model (GPTModel): The GPT model chunk to execute.

        Returns:
            tuple[torch.Tensor, Callable[[torch.Tensor], dict[str, list[torch.Tensor]]]]:
            Output tensor(s) and a callable that computes and packages results
            to be collected by the engine.
        """

        assert not return_schedule_plan, "forward_only step should never return schedule plan"

        # Get the batch.
        batch = get_batch(
            data_iterator,
            [
                "tokens",
                "loss_masks",
                "multimodal_train_inputs",
                "total_lengths",
                "response_lengths",
                "max_seq_lens",
            ],
            parallel_state,
            args.data_pad_size_multiplier,
            args.qkv_format,
        )
        unconcat_tokens = batch["unconcat_tokens"]
        tokens = batch["tokens"]
        packed_seq_params = get_packed_seq_params(batch, args)
        total_lengths = batch["total_lengths"]
        response_lengths = batch["response_lengths"]
        output_tensor = model(
            input_ids=tokens,
            position_ids=None,
            attention_mask=None,
            labels=None,
            packed_seq_params=packed_seq_params,
            loss_mask=batch["full_loss_masks"],
            **(batch["multimodal_train_inputs"] if batch["multimodal_train_inputs"] is not None else {}),
        )

        return output_tensor, partial(
            f,
            args=args,
            parallel_state=parallel_state,
            unconcat_tokens=unconcat_tokens,
            total_lengths=total_lengths,
            response_lengths=response_lengths,
            with_entropy=args.use_rollout_entropy,
            max_seq_lens=batch.get("max_seq_lens", None),
        )

    # Turn on evaluation mode which disables dropout.
    for model_module in model:
        model_module.eval()

    if args.custom_megatron_before_log_prob_hook_path:
        from miles.utils.misc import load_function

        custom_before_log_prob_hook = load_function(args.custom_megatron_before_log_prob_hook_path)
        custom_before_log_prob_hook(args, model, store_prefix)

    forward_backward_func = get_forward_backward_func()
    # Don't care about timing during evaluation
    config.timers = None
    forward_data_store = []
    num_steps_per_rollout = len(num_microbatches)
    for step_id in range(num_steps_per_rollout):
        # collect_non_loss_data
        forward_data_store += forward_backward_func(
            forward_step_func=forward_step,
            data_iterator=data_iterator,
            model=model,
            num_microbatches=num_microbatches[step_id],
            seq_length=args.seq_length,
            micro_batch_size=args.micro_batch_size,
            forward_only=True,
            collect_non_loss_data=True,
        )

    # Move model back to the train mode.
    for model_module in model:
        model_module.train()

    rollout_data = {}
    # Store the results on the last stage
    if mpu.is_pipeline_last_stage():
        aggregated = aggregate_forward_results(forward_data_store, data_iterator[0], args, store_prefix="")
        for key, value in aggregated.items():
            rollout_data[f"{store_prefix}{key}"] = value
    return rollout_data


def train_one_step(
    args: Namespace,
    rollout_id: int,
    step_id: int,
    data_iterator: Sequence[DataIterator],
    model: Sequence[DDP],
    optimizer: MegatronOptimizer,
    opt_param_scheduler: OptimizerParamScheduler,
    num_microbatches: int,
    parallel_state: ParallelState,
) -> tuple[dict[str, float], float]:
    """Execute a single pipeline-parallel training step.

    Runs forward/backward over ``num_microbatches``, applies optimizer step and
    one scheduler step when gradients are valid.

    Args:
        args (Namespace): Runtime arguments.
        rollout_id (int): Rollout identifier.
        step_id (int): Step index within the current rollout.
        data_iterator (Sequence[DataIterator]): Iterable(s) yielding training batches.
        model (Sequence[DDP]): Sequence of DDP-wrapped model chunks.
        optimizer (MegatronOptimizer): Optimizer instance.
        opt_param_scheduler (OptimizerParamScheduler): LR/WD scheduler.
        num_microbatches (int): Number of microbatches to process.

    Returns:
        tuple[dict[str, float], float]: Reduced loss dictionary (last stage only)
        and gradient norm for logging.
    """
    args = get_args()

    # Set grad to zero.
    for model_chunk in model:
        model_chunk.zero_grad_buffer()
    optimizer.zero_grad()

    if args.custom_megatron_before_train_step_hook_path:
        from miles.utils.misc import load_function

        custom_before_train_step_hook = load_function(args.custom_megatron_before_train_step_hook_path)
        custom_before_train_step_hook(args, rollout_id, step_id, model, optimizer, opt_param_scheduler)

    def forward_step(data_iterator: DataIterator, model: GPTModel, return_schedule_plan: bool = False) -> tuple[
        torch.Tensor,
        Callable[[torch.Tensor], tuple[torch.Tensor, int, dict[str, torch.Tensor | list[str]]]],
    ]:
        """Forward step used by Megatron's pipeline engine during training.

        Args:
            data_iterator (DataIterator): Input data iterator.
            model (GPTModel): The GPT model chunk to execute.

        Returns:
            tuple[torch.Tensor, Callable[[torch.Tensor], tuple[torch.Tensor, int, dict[str, torch.Tensor | list[str]]]]]:
            Output tensor(s) and the loss function, which returns
            (loss, num_elems, {"keys": list[str], "values": torch.Tensor}).
        """

        # Get the batch.
        batch = get_batch(
            data_iterator,
            [
                "tokens",
                "multimodal_train_inputs",
                "packed_seq_params",
                "total_lengths",
                "response_lengths",
                "loss_masks",
                "log_probs",
                "ref_log_probs",
                "values",
                "advantages",
                "returns",
                "rollout_log_probs",
                "max_seq_lens",
            ],
            parallel_state,
            args.data_pad_size_multiplier,
            args.qkv_format,
        )

        if os.environ.get("ENABLE_ROUTING_REPLAY", "0") == "1":
            old_stage = os.environ["ROUTING_REPLAY_STAGE"]
            os.environ["ROUTING_REPLAY_STAGE"] = "replay_forward"

        if return_schedule_plan:
            assert not args.enable_mtp_training, "MTP training should not be enabled when using combined 1f1b"
            output_tensor = model.build_schedule_plan(
                input_ids=batch["tokens"],
                position_ids=None,
                attention_mask=None,
                labels=None,
                packed_seq_params=get_packed_seq_params(batch, args),
                loss_mask=batch["full_loss_masks"],
            )
        else:
            forward_kwargs = {
                "input_ids": batch["tokens"],
                "position_ids": None,
                "attention_mask": None,
                "labels": None,
                "packed_seq_params": get_packed_seq_params(batch, args),
                "loss_mask": batch["full_loss_masks"],
            }

            if args.enable_mtp_training:
                forward_kwargs["mtp_kwargs"] = {"mtp_labels": batch["tokens"]}

            if batch["multimodal_train_inputs"] is not None:
                forward_kwargs.update(batch["multimodal_train_inputs"])

            output_tensor = model(**forward_kwargs)

        if os.environ.get("ENABLE_ROUTING_REPLAY", "0") == "1":
            os.environ["ROUTING_REPLAY_STAGE"] = old_stage

        return output_tensor, partial(
            loss_function, args, parallel_state, batch, num_microbatches, apply_megatron_loss_scaling=True
        )

    # Forward pass.
    forward_backward_func = get_forward_backward_func()
    losses_reduced = forward_backward_func(
        forward_step_func=forward_step,
        data_iterator=data_iterator,
        model=model,
        num_microbatches=num_microbatches,
        seq_length=args.seq_length,
        micro_batch_size=args.micro_batch_size,
        decoder_seq_length=args.decoder_seq_length,
        forward_only=False,
    )

    valid_step = True
    if not getattr(args, "check_for_nan_in_loss_and_grad", True):
        found_inf_flag = optimizer.prepare_grads()
        if found_inf_flag:
            valid_step = False
        else:
            grad_norm = optimizer.get_grad_norm()
            if isinstance(grad_norm, torch.Tensor):
                valid_step = not (torch.isnan(grad_norm) or torch.isinf(grad_norm))
            else:
                valid_step = not (math.isnan(grad_norm) or math.isinf(grad_norm))

    # CI check: verify only MTP parameters have non-zero gradients when truncation happens
    # This check must happen before optimizer.step() as gradients may be modified during step
    if args.ci_test and args.enable_mtp_training:
        from miles.backends.megatron_utils.ci_utils import check_mtp_only_grad

        check_mtp_only_grad(model, step_id)

    if valid_step:
        # Update parameters.
        update_successful, grad_norm, num_zeros_in_grad = optimizer.step()

        # Update learning rate.
        assert update_successful
        opt_param_scheduler.step(increment=args.global_batch_size)

    # release grad
    for model_chunk in model:
        model_chunk.zero_grad_buffer()
    optimizer.zero_grad()

    if mpu.is_pipeline_last_stage(ignore_virtual=True):
        loss_reduced = aggregate_train_losses(losses_reduced, parallel_state)
        return loss_reduced, grad_norm
    return {}, grad_norm


def should_disable_forward_pre_hook(args: Namespace) -> bool:
    """Block forward pre-hook for certain configurations."""
    return args.use_distributed_optimizer and args.overlap_param_gather


def finalize_model_grads_with_empty_cache(*args, **kwargs):
    # trigger empty cache when there are less than 10% free memory before the final reduce scatter.
    # TODO: this is an ad-hoc method and we should figure out why the oom happens in the first place.
    device = torch.cuda.current_device()
    free, total = torch.cuda.mem_get_info(device)
    if free / total < 0.1:
        clear_memory()
    return finalize_model_grads(*args, **kwargs)


def train(
    rollout_id: int,
    model: Sequence[DDP],
    optimizer: MegatronOptimizer,
    opt_param_scheduler: OptimizerParamScheduler,
    data_iterator: Sequence[DataIterator],
    num_microbatches: Sequence[int],
    parallel_state: ParallelState,
) -> None:
    """Run training over a rollout consisting of multiple steps.

    The model is switched to train mode, training hooks are configured, and
    ``train_one_step`` is invoked for each step in the rollout.

    Args:
        rollout_id (int): Rollout identifier.
        model (Sequence[DDP]): Sequence of DDP-wrapped model chunks.
        optimizer (MegatronOptimizer): Optimizer instance.
        opt_param_scheduler (OptimizerParamScheduler): LR/WD scheduler.
        data_iterator (Sequence[DataIterator]): Iterable(s) yielding training batches.
        num_microbatches (Sequence[int]): Microbatches per step in the rollout.
    """
    args = get_args()

    for iterator in data_iterator:
        iterator.reset()

    # Turn on training mode which enables dropout.
    for model_module in model:
        model_module.train()

    # Setup some training config params.
    config = get_model_config(model[0])
    config.grad_scale_func = optimizer.scale_loss
    config.timers = None
    if isinstance(model[0], DDP) and args.overlap_grad_reduce:
        assert config.no_sync_func is None, (
            "When overlap_grad_reduce is True, config.no_sync_func must be None; "
            "a custom no_sync_func is not supported when overlapping grad-reduce"
        )
        config.no_sync_func = [model_chunk.no_sync for model_chunk in model]
        if len(model) == 1:
            config.no_sync_func = config.no_sync_func[0]
        if args.align_grad_reduce:
            config.grad_sync_func = [model_chunk.start_grad_sync for model_chunk in model]
            if len(model) == 1:
                config.grad_sync_func = config.grad_sync_func[0]
    if args.overlap_param_gather and args.align_param_gather:
        config.param_sync_func = [model_chunk.start_param_sync for model_chunk in model]
        if len(model) == 1:
            config.param_sync_func = config.param_sync_func[0]
    config.finalize_model_grads_func = finalize_model_grads_with_empty_cache

    pre_hook_enabled = False

    if args.reset_optimizer_states:
        if (
            mpu.get_data_parallel_rank(with_context_parallel=True) == 0
            and mpu.get_tensor_model_parallel_rank() == 0
            and mpu.get_pipeline_model_parallel_rank() == mpu.get_pipeline_model_parallel_world_size() - 1
        ):
            print("Reset optimizer states")
        for chained_optimizer in optimizer.chained_optimizers:
            for group in chained_optimizer.optimizer.param_groups:
                if "step" in group:
                    group["step"] = 0
            for state in chained_optimizer.optimizer.state.values():
                if "exp_avg" in state:
                    state["exp_avg"].zero_()
                if "exp_avg_sq" in state:
                    state["exp_avg_sq"].zero_()

    if args.manual_gc:
        # Disable the default garbage collector and perform the collection manually.
        # This is to align the timing of garbage collection across ranks.
        assert args.manual_gc_interval >= 0, "Manual garbage collection interval should be larger than or equal to 0"
        gc.disable()
        gc.collect()

    # Disable forward pre-hook to start training to ensure that errors in checkpoint loading
    # or random initialization don't propagate to all ranks in first all-gather (which is a
    # no-op if things work correctly).
    if should_disable_forward_pre_hook(args):
        disable_forward_pre_hook(model, param_sync=False)
        # Also remove param_sync_func temporarily so that sync calls made in
        # `forward_backward_func` are no-ops.
        param_sync_func = config.param_sync_func
        config.param_sync_func = None
        pre_hook_enabled = False

    num_steps_per_rollout = len(num_microbatches)

    # Run training iterations till done.
    for step_id in range(num_steps_per_rollout):

        # Run training step.
        loss_dict, grad_norm = train_one_step(
            args,
            rollout_id,
            step_id,
            data_iterator,
            model,
            optimizer,
            opt_param_scheduler,
            num_microbatches[step_id],
            parallel_state,
        )

        if step_id == 0:
            # Enable forward pre-hook after training step has successfully run. All subsequent
            # forward passes will use the forward pre-hook / `param_sync_func` in
            # `forward_backward_func`.
            if should_disable_forward_pre_hook(args):
                enable_forward_pre_hook(model)
                config.param_sync_func = param_sync_func
                pre_hook_enabled = True

        if args.enable_mtp_training:
            from megatron.core.transformer.multi_token_prediction import MTPLossLoggingHelper

            mtp_loss_scale = 1 / num_microbatches[step_id]
            tracker = MTPLossLoggingHelper.tracker
            if "values" in tracker:
                values = tracker["values"]
                if tracker.get("reduce_group") is not None:
                    torch.distributed.all_reduce(values, group=tracker.get("reduce_group"))
                if tracker.get("avg_group") is not None:
                    torch.distributed.all_reduce(values, group=tracker["avg_group"], op=torch.distributed.ReduceOp.AVG)
                # here we assume only one mtp layer
                mtp_losses = (tracker["values"] * mtp_loss_scale).item()
                MTPLossLoggingHelper.clean_loss_in_tracker()

                # CI check: verify MTP loss is within expected bounds
                if args.ci_test:
                    from miles.backends.megatron_utils.ci_utils import check_mtp_loss

                    check_mtp_loss(mtp_losses)

        # per train step log.
        if (
            mpu.get_data_parallel_rank(with_context_parallel=True) == 0
            and mpu.get_tensor_model_parallel_rank() == 0
            and mpu.get_pipeline_model_parallel_rank() == mpu.get_pipeline_model_parallel_world_size() - 1
        ):
            accumulated_step_id = rollout_id * num_steps_per_rollout + step_id
            role = getattr(model[0], "role", "actor")
            role_tag = "" if role == "actor" else f"{role}-"

            extra_metrics = {}
            if args.enable_mtp_training:
                extra_metrics["mtp_loss"] = mtp_losses

            for param_group_id, param_group in enumerate(optimizer.param_groups):
                extra_metrics[f"lr-pg_{param_group_id}"] = opt_param_scheduler.get_lr(param_group)

            log_dict = log_train_step(
                args=args,
                loss_dict=loss_dict,
                grad_norm=grad_norm,
                rollout_id=rollout_id,
                step_id=step_id,
                num_steps_per_rollout=num_steps_per_rollout,
                role=role,
                extra_metrics=extra_metrics,
                should_log=True,
            )

            if args.ci_test and not args.ci_disable_kl_checker:
                check_kl(args, log_dict, step_id, accumulated_step_id)

            logger.info(f"{role_tag}step {accumulated_step_id}: {log_dict}")

            if args.ci_test:
                check_grad_norm(
                    args=args,
                    grad_norm=grad_norm,
                    rollout_id=rollout_id,
                    step_id=step_id,
                    role=role,
                    rank=mpu.get_data_parallel_rank(),
                )

    # Close out pre-hooks if using distributed optimizer and overlapped param gather.
    if pre_hook_enabled:
        disable_forward_pre_hook(model)


def save(
    iteration: int, model: Sequence[DDP], optimizer: MegatronOptimizer, opt_param_scheduler: OptimizerParamScheduler
) -> None:
    """Persist a training checkpoint safely with forward hooks disabled.

    Args:
        iteration (int): Current global iteration number.
        model (Sequence[DDP]): Sequence of DDP-wrapped model chunks.
        optimizer (MegatronOptimizer): Optimizer instance.
        opt_param_scheduler (OptimizerParamScheduler): LR/WD scheduler.
    """
    args = get_args()
    if should_disable_forward_pre_hook(args):
        disable_forward_pre_hook(model)

    # Monkey-patch dist_checkpointing.save to add debug logging inside save_checkpoint.
    # save_checkpoint calls dist_checkpointing.save() via module attribute, so this works.
    import megatron.core.dist_checkpointing as _dist_ckpt_module
    import megatron.core.dist_checkpointing.serialization as _ser_module

    _orig_dist_save = _dist_ckpt_module.save
    _orig_save_preprocess = _ser_module.save_preprocess

    def _debug_save_preprocess(sharded_state_dict, validate_access_integrity, preprocess_fn=None):
        logger.info("[DEBUG save_checkpoint] >>> save_preprocess START (includes validation)")
        try:
            result = _orig_save_preprocess(sharded_state_dict, validate_access_integrity, preprocess_fn)
            logger.info("[DEBUG save_checkpoint] <<< save_preprocess DONE (validation passed)")
            return result
        except Exception as e:
            logger.error(f"[DEBUG save_checkpoint] !!! save_preprocess FAILED: {e}")
            raise

    def _debug_dist_save(sharded_state_dict, checkpoint_dir, sharded_strategy=None, **kwargs):
        """Instrumented version of dist_checkpointing.save with per-step logging."""
        import torch
        from megatron.core.dist_checkpointing.mapping import ShardedObject
        from megatron.core.dist_checkpointing.serialization import (
            CheckpointingConfig,
            SaveCommonStrategy,
            SaveShardedStrategy,
            AsyncSaveShardedStrategy,
            get_default_save_sharded_strategy,
            get_default_save_common_strategy,
            get_default_strategy,
            save_config,
            validate_sharded_objects_handling,
        )
        from megatron.core.dist_checkpointing.serialization import StrategyAction, _CONTENT_METADATA_KEY
        from megatron.core.dist_checkpointing.utils import extract_matching_values

        logger.info(f"[DEBUG dist_ckpt.save] >>> ENTER (checkpoint_dir={checkpoint_dir})")

        validate_access_integrity = kwargs.get('validate_access_integrity', True)
        async_sharded_save = kwargs.get('async_sharded_save', False)
        preprocess_fn = kwargs.get('preprocess_common_before_consistancy_check', None)
        common_strategy = kwargs.get('common_strategy', None)
        content_metadata = kwargs.get('content_metadata', None)

        # Strategy init
        if sharded_strategy is None:
            sharded_strategy = get_default_save_sharded_strategy()
        if not isinstance(sharded_strategy, SaveShardedStrategy):
            sharded_strategy = get_default_strategy(StrategyAction.SAVE_SHARDED, *sharded_strategy)
        if common_strategy is None:
            common_strategy = get_default_save_common_strategy()
        if not isinstance(common_strategy, SaveCommonStrategy):
            common_strategy = get_default_strategy(StrategyAction.SAVE_COMMON, *common_strategy)

        if content_metadata is not None:
            sharded_state_dict[_CONTENT_METADATA_KEY] = content_metadata

        # Step 1: save_preprocess (validation)
        logger.info("[DEBUG dist_ckpt.save] step 1/5: save_preprocess (validation)")
        sharded_state_dict, state_dict = _debug_save_preprocess(
            sharded_state_dict, validate_access_integrity, preprocess_fn
        )

        # Step 2: save_common
        logger.info("[DEBUG dist_ckpt.save] step 2/5: common_strategy.save_common")
        common_strategy.save_common(state_dict, checkpoint_dir)
        logger.info("[DEBUG dist_ckpt.save] step 2/5: common_strategy.save_common DONE")

        # Step 3: save sharded objects
        if not sharded_strategy.can_handle_sharded_objects:
            logger.info("[DEBUG dist_ckpt.save] step 3/5: save_sharded_objects")
            validate_sharded_objects_handling(sharded_strategy, common_strategy)
            sharded_objects_state_dict, sharded_state_dict = extract_matching_values(
                sharded_state_dict, lambda v: isinstance(v, ShardedObject)
            )
            common_strategy.save_sharded_objects(sharded_objects_state_dict, checkpoint_dir)
            logger.info("[DEBUG dist_ckpt.save] step 3/5: save_sharded_objects DONE")
        else:
            logger.info("[DEBUG dist_ckpt.save] step 3/5: skipped (strategy handles sharded objects)")

        def metadata_finalize_fn():
            if torch.distributed.get_rank() == 0:
                save_config(
                    CheckpointingConfig(sharded_strategy.backend, sharded_strategy.version),
                    checkpoint_dir,
                )
            torch.distributed.barrier()

        # Step 4: save sharded tensors
        if not async_sharded_save:
            logger.info("[DEBUG dist_ckpt.save] step 4/5: sharded_strategy.save (synchronous) - EXPANDED")

            # The sharded_strategy may be a FullyParallelSaveStrategyWrapper which wraps a
            # TorchDistSaveShardedStrategy as base_strategy. We instrument at the right level.
            from megatron.core.dist_checkpointing.strategies.fully_parallel import (
                FullyParallelSaveStrategyWrapper,
            )

            # Determine the actual TorchDistSaveShardedStrategy
            if isinstance(sharded_strategy, FullyParallelSaveStrategyWrapper):
                inner_strategy = sharded_strategy.base_strategy
                logger.info(f"[DEBUG dist_ckpt.save] strategy is FullyParallelSaveStrategyWrapper, inner={type(inner_strategy).__name__}")
            else:
                inner_strategy = sharded_strategy
                logger.info(f"[DEBUG dist_ckpt.save] strategy is {type(inner_strategy).__name__} (no wrapper)")

            # Monkey-patch inner_strategy.async_save to log sub-steps
            _orig_inner_async_save = inner_strategy.async_save

            def _debug_inner_async_save(sd, ckpt_dir):
                from megatron.core.dist_checkpointing.strategies.torch import (
                    _replace_state_dict_keys_with_sharded_keys,
                    mcore_to_pyt_state_dict,
                )
                from megatron.core.dist_checkpointing.strategies.state_dict_saver import (
                    save_state_dict_async_plan,
                )
                from megatron.core.dist_checkpointing.strategies.torch import (
                    MCoreSavePlanner,
                    MultiStorageClientFeature,
                    FileSystemWriterAsync,
                )

                logger.info("[DEBUG dist_ckpt.save] step 4a: _replace_state_dict_keys_with_sharded_keys START")
                (sd_transformed, flat_mapping, rename_mapping) = (
                    _replace_state_dict_keys_with_sharded_keys(
                        sd, inner_strategy.keep_only_main_replica
                    )
                )
                logger.info("[DEBUG dist_ckpt.save] step 4a: _replace_state_dict_keys_with_sharded_keys DONE")

                logger.info("[DEBUG dist_ckpt.save] step 4b: mcore_to_pyt_state_dict START")
                pyt_state_dict = mcore_to_pyt_state_dict(sd_transformed, False)
                logger.info(f"[DEBUG dist_ckpt.save] step 4b: mcore_to_pyt_state_dict DONE (keys={len(pyt_state_dict)})")

                logger.info("[DEBUG dist_ckpt.save] step 4c: FileSystemWriterAsync + save_state_dict_async_plan START")
                writer = FileSystemWriterAsync(
                    ckpt_dir,
                    separation_hint=inner_strategy.separation_hint,
                    thread_count=inner_strategy.thread_count,
                    use_msc=MultiStorageClientFeature.is_enabled(),
                )
                coordinator = 0
                args_cached_plans = None
                loaded_all_plans = None
                if inner_strategy.use_cached_ckpt_structure:
                    loaded_all_plans = getattr(inner_strategy.cached_global_metadata, "all_local_plans", None)
                    args_cached_plans = (
                        inner_strategy.cached_central_plan,
                        inner_strategy.cached_local_plan,
                        inner_strategy.validated_cache_reuse,
                    )

                (
                    save_state_dict_ret,
                    inner_strategy.cached_central_plan,
                    inner_strategy.cached_local_plan,
                    inner_strategy.validated_cache_reuse,
                    inner_strategy.validated_loaded_metadata_reuse,
                ) = save_state_dict_async_plan(
                    pyt_state_dict,
                    writer,
                    None,
                    coordinator,
                    planner=MCoreSavePlanner(
                        dedup_replicated_tensors=not inner_strategy.keep_only_main_replica,
                        flatten_state_dict=False,
                    ),
                    cached_ckpt_structure=args_cached_plans,
                    loaded_all_plans=loaded_all_plans,
                )
                logger.info("[DEBUG dist_ckpt.save] step 4c: save_state_dict_async_plan DONE")

                # Handle cached metadata reuse (same as async_save logic)
                rank = torch.distributed.get_rank()
                if inner_strategy.use_cached_ckpt_structure:
                    if (
                        loaded_all_plans
                        and inner_strategy.cached_global_metadata
                        and inner_strategy.validated_loaded_metadata_reuse
                    ):
                        if coordinator == rank:
                            save_state_dict_ret = list(save_state_dict_ret)
                            save_state_dict_ret[1] = inner_strategy.cached_global_metadata
                    elif inner_strategy.validated_cache_reuse:
                        if save_state_dict_ret[1]:
                            inner_strategy.cached_global_metadata = save_state_dict_ret[1]
                        elif coordinator == rank:
                            save_state_dict_ret = list(save_state_dict_ret)
                            save_state_dict_ret[1] = inner_strategy.cached_global_metadata

                logger.info("[DEBUG dist_ckpt.save] step 4d: _get_save_and_finalize_callbacks START")
                async_req = inner_strategy._get_save_and_finalize_callbacks(writer, save_state_dict_ret)
                logger.info("[DEBUG dist_ckpt.save] step 4d: _get_save_and_finalize_callbacks DONE")
                return async_req

            # Patch the inner strategy
            inner_strategy.async_save = _debug_inner_async_save

            def _debug_wrapper_save(sd, ckpt_dir):
                # FullyParallelSaveStrategyWrapper.save calls:
                #   apply_saving_parallelization(sd) then base_strategy.save(sd, ckpt_dir)
                # And base_strategy.save (AsyncSaveShardedStrategy.save) calls:
                #   async_request = async_save(sd, ckpt_dir) then async_request.execute_sync()
                #
                # We already patched inner async_save for sub-step logging.
                # Now we also need to log around execute_sync sub-steps.
                # Override base_strategy.save to break execute_sync apart.

                if isinstance(sharded_strategy, FullyParallelSaveStrategyWrapper):
                    logger.info("[DEBUG dist_ckpt.save] step 4-parallel: apply_saving_parallelization START")
                    sharded_strategy.apply_saving_parallelization(sd)
                    logger.info("[DEBUG dist_ckpt.save] step 4-parallel: apply_saving_parallelization DONE")
                    async_request = inner_strategy.async_save(sd, ckpt_dir)
                else:
                    async_request = inner_strategy.async_save(sd, ckpt_dir)

                # -- execute_sync broken into sub-steps --
                logger.info("[DEBUG dist_ckpt.save] step 4e: execute_sync START")
                async_fn_args = list(async_request.async_fn_args)
                if async_request.preload_fn:
                    logger.info("[DEBUG dist_ckpt.save] step 4e-preload: preload_fn START")
                    assert len(async_fn_args) == 3, "Expected 3 args"
                    async_fn_args[1] = async_request.preload_fn()
                    logger.info("[DEBUG dist_ckpt.save] step 4e-preload: preload_fn DONE")

                if async_request.async_fn is not None:
                    logger.info("[DEBUG dist_ckpt.save] step 4e-write: async_fn (file write) START")
                    # Expand write_preloaded_data_multiproc inline for sub-step logging.
                    # async_fn is partial(write_preloaded_data_multiproc, transform_list, use_msc)
                    # async_fn_args = [rank, write_buckets, results_queue]
                    import gc as _gc
                    from torch import multiprocessing as _mp
                    from functools import partial as _partial
                    from time import time as _time
                    from megatron.core.dist_checkpointing.strategies.filesystem_async import (
                        FileSystemWriterAsync as _FSWA,
                    )

                    _write_rank = async_fn_args[0]
                    _write_buckets = async_fn_args[1]
                    _results_queue = async_fn_args[2] if len(async_fn_args) > 2 else None

                    # Extract transform_list and use_msc from the partial
                    _transform_list = async_request.async_fn.args[0] if hasattr(async_request.async_fn, 'args') else []
                    _use_msc = async_request.async_fn.args[1] if hasattr(async_request.async_fn, 'args') and len(async_request.async_fn.args) > 1 else False

                    logger.info(f"[DEBUG dist_ckpt.save] step 4e-write: rank={_write_rank}, num_buckets={len(_write_buckets) if _write_buckets else 0}")

                    # Pre-load ctypes and libc in the parent so they're available immediately
                    # in the forked child (fork copies parent's memory, including loaded libraries).
                    # This avoids the child crashing during ctypes import/initialization.
                    import ctypes as _ctypes
                    import ctypes.util as _ctypes_util
                    import os as _child_os

                    _libc_path = _ctypes_util.find_library("c")
                    _libc = _ctypes.CDLL(_libc_path, use_errno=True)
                    _SIGSEGV_CONST = 11
                    _SIGBUS_CONST = 7
                    _SA_STRUCT_SIZE = 152  # struct sigaction on x86_64 Linux
                    _SIG_DFL_CONST = 0

                    # Pre-build the default sigaction struct in the parent
                    _sa_default_buf = _ctypes.create_string_buffer(_SA_STRUCT_SIZE)
                    _ctypes.memmove(_sa_default_buf, _ctypes.c_void_p(_SIG_DFL_CONST), 8)

                    # Register a fork handler that resets signal handlers in the child
                    # immediately after fork(), before any other Python code runs.
                    # This is the earliest possible hook after fork().
                    _fork_handler_registered = [False]

                    def _after_fork_in_child():
                        """Reset Go runtime's signal handlers immediately after fork."""
                        try:
                            _libc.sigaction(_SIGSEGV_CONST, _sa_default_buf, None)
                            _libc.sigaction(_SIGBUS_CONST, _sa_default_buf, None)
                            _libc.signal(_SIGSEGV_CONST, _SIG_DFL_CONST)
                            _libc.signal(_SIGBUS_CONST, _SIG_DFL_CONST)
                        except Exception:
                            pass  # Best effort - don't crash the child if this fails

                    if not _fork_handler_registered[0]:
                        _child_os.register_at_fork(after_in_child=_after_fork_in_child)
                        _fork_handler_registered[0] = True
                        logger.info("[DEBUG dist_ckpt.save] step 4e-write: registered after_fork_in_child signal reset handler")

                    # Child wrapper still does signal reset as belt-and-suspenders
                    # (in case register_at_fork didn't fully work)
                    def _child_target_wrapper(original_fn, **kwargs):
                        _pid = _child_os.getpid()
                        _child_os.write(2, f"[child pid={_pid}] _child_target_wrapper ENTERED\n".encode())

                        try:
                            # Signal handlers should already be reset by _after_fork_in_child,
                            # but reset again as belt-and-suspenders.
                            _libc.sigaction(_SIGSEGV_CONST, _sa_default_buf, None)
                            _libc.sigaction(_SIGBUS_CONST, _sa_default_buf, None)
                            _libc.signal(_SIGSEGV_CONST, _SIG_DFL_CONST)
                            _libc.signal(_SIGBUS_CONST, _SIG_DFL_CONST)

                            _child_os.write(2, f"[child pid={_pid}] signal handlers reset, calling original_fn\n".encode())
                            result = original_fn(**kwargs)
                            _child_os.write(2, f"[child pid={_pid}] original_fn completed successfully\n".encode())
                            return result
                        except Exception as _e:
                            _child_os.write(2, f"[child pid={_pid}] EXCEPTION in child: {_e}\n".encode())
                            import traceback as _tb
                            _child_os.write(2, f"[child pid={_pid}] {_tb.format_exc()}\n".encode())
                            raise

                    if _write_buckets:
                        _gc_was_enabled = _gc.isenabled()
                        if _gc_was_enabled:
                            _gc.disable()
                        try:
                            _w_start = _time()
                            _write_results_or_exc = dict()
                            _ctx = _mp.get_context("fork")
                            _local_results_queue = _ctx.Queue()
                            _count_queue = _ctx.JoinableQueue()
                            _p_list = []

                            logger.info(f"[DEBUG dist_ckpt.save] step 4e-write: creating {len(_write_buckets)} fork processes (with signal reset)")
                            for _i, _write_bucket in enumerate(_write_buckets):
                                _count_queue.put(_i)
                                _kwargs = {
                                    "local_proc_idx": _i,
                                    "write_bucket": _write_bucket,
                                    "results_queue": _local_results_queue,
                                    "count_queue": _count_queue,
                                    "use_fsync": True,
                                }
                                if _use_msc:
                                    import inspect as _inspect
                                    _signature = _inspect.signature(_FSWA.write_preloaded_data)
                                    if len(_signature.parameters) > 6:
                                        _kwargs["use_msc"] = _use_msc
                                _p_list.append(
                                    _ctx.Process(
                                        target=_partial(
                                            _child_target_wrapper,
                                            _partial(_FSWA.write_preloaded_data, _transform_list),
                                        ),
                                        kwargs=_kwargs,
                                    )
                                )
                            logger.info(f"[DEBUG dist_ckpt.save] step 4e-write: created {len(_p_list)} processes, starting them")
                            # NOTE: Do NOT reset parent's signal handlers before fork.
                            # Resetting to SIG_DFL makes the parent vulnerable to SIGSEGV during the
                            # fork syscall itself (Go runtime state can trigger SIGSEGV during fork).
                            # Instead, children reset their own handlers via _child_target_wrapper.

                            for _pi, _p in enumerate(_p_list):
                                logger.info(f"[DEBUG dist_ckpt.save] step 4e-write: starting process {_pi}")
                                _p.start()
                                logger.info(f"[DEBUG dist_ckpt.save] step 4e-write: started process {_pi} (pid={_p.pid})")

                            logger.info("[DEBUG dist_ckpt.save] step 4e-write: all processes started, waiting for completion")
                            # Poll for completion instead of count_queue.join() which blocks forever
                            # if a child dies (e.g. SIGSEGV with SIG_DFL kills process silently).
                            import time as _time_mod
                            _POLL_INTERVAL = 2.0
                            _MAX_WAIT = 1200  # 20 minutes max
                            _wait_start = _time_mod.time()
                            _all_done = False
                            while not _all_done:
                                # Check if all children are still alive
                                _dead_children = []
                                for _pi2, _p2 in enumerate(_p_list):
                                    if not _p2.is_alive() and _p2.exitcode is not None:
                                        if _p2.exitcode != 0:
                                            _dead_children.append((_pi2, _p2.exitcode))
                                if _dead_children:
                                    for _dc_idx, _dc_exit in _dead_children:
                                        logger.error(
                                            f"[DEBUG dist_ckpt.save] step 4e-write: child process {_dc_idx} "
                                            f"(pid={_p_list[_dc_idx].pid}) DIED with exit code {_dc_exit} "
                                            f"(signal {-_dc_exit if _dc_exit < 0 else 'N/A'})"
                                        )
                                    raise RuntimeError(
                                        f"Checkpoint write child process(es) died: "
                                        f"{[(idx, code) for idx, code in _dead_children]}"
                                    )
                                # Try non-blocking join on the count_queue
                                # All tasks done = unfinished_tasks == 0
                                if _count_queue._unfinished_tasks._semlock._get_value() == 0:  # type: ignore
                                    _all_done = True
                                    break
                                if _time_mod.time() - _wait_start > _MAX_WAIT:
                                    # Log status of all children
                                    for _pi2, _p2 in enumerate(_p_list):
                                        logger.error(
                                            f"[DEBUG dist_ckpt.save] step 4e-write: timeout - child {_pi2} "
                                            f"alive={_p2.is_alive()}, exitcode={_p2.exitcode}"
                                        )
                                    raise RuntimeError(
                                        f"Checkpoint write timed out after {_MAX_WAIT}s waiting for child processes"
                                    )
                                _time_mod.sleep(_POLL_INTERVAL)
                            logger.info("[DEBUG dist_ckpt.save] step 4e-write: all children completed, collecting results")

                            for _proc_idx in range(len(_write_buckets)):
                                _local_proc_idx, _local_results_or_exc = _local_results_queue.get()
                                if isinstance(_local_results_or_exc, Exception):
                                    logger.error(f"[DEBUG dist_ckpt.save] step 4e-write: process {_local_proc_idx} FAILED: {_local_results_or_exc}")
                                    _write_results_or_exc = _local_results_or_exc
                                    break
                                _write_results_or_exc[_local_proc_idx] = _local_results_or_exc
                                _p_list[_local_proc_idx].join()

                            logger.info(f"[DEBUG dist_ckpt.save] step 4e-write: results collected, putting to global queue")
                            _results_queue.put(_write_results_or_exc)
                            _w_end = _time()
                            logger.info(f"[DEBUG dist_ckpt.save] step 4e-write: done in {_w_end - _w_start:.2f}s")
                        finally:
                            if _gc_was_enabled:
                                _gc.enable()
                    else:
                        logger.info("[DEBUG dist_ckpt.save] step 4e-write: no write_buckets, skipping")

                    logger.info("[DEBUG dist_ckpt.save] step 4e-write: async_fn (file write) DONE")

                logger.info("[DEBUG dist_ckpt.save] step 4e-barrier: torch.distributed.barrier START")
                torch.distributed.barrier()
                logger.info("[DEBUG dist_ckpt.save] step 4e-barrier: torch.distributed.barrier DONE")

                logger.info(f"[DEBUG dist_ckpt.save] step 4e-finalize: {len(async_request.finalize_fns)} finalize_fns START")
                for i, finalize_fn in enumerate(async_request.finalize_fns):
                    logger.info(f"[DEBUG dist_ckpt.save] step 4e-finalize[{i}]: {finalize_fn} START")
                    finalize_fn()
                    logger.info(f"[DEBUG dist_ckpt.save] step 4e-finalize[{i}]: DONE")
                logger.info("[DEBUG dist_ckpt.save] step 4e: execute_sync DONE")

            try:
                _debug_wrapper_save(sharded_state_dict, checkpoint_dir)
            finally:
                inner_strategy.async_save = _orig_inner_async_save

            logger.info("[DEBUG dist_ckpt.save] step 4/5: sharded_strategy.save DONE (expanded)")

            # Step 5: metadata finalize
            logger.info("[DEBUG dist_ckpt.save] step 5/5: metadata_finalize_fn")
            metadata_finalize_fn()
            logger.info("[DEBUG dist_ckpt.save] step 5/5: metadata_finalize_fn DONE")
            logger.info("[DEBUG dist_ckpt.save] <<< EXIT (sync save complete)")
            return None

        # Async path
        logger.info("[DEBUG dist_ckpt.save] step 4/5: sharded_strategy.async_save")
        if not isinstance(sharded_strategy, AsyncSaveShardedStrategy):
            raise Exception(f'Cannot apply async_save to non-async strategy {sharded_strategy}')
        async_request = sharded_strategy.async_save(sharded_state_dict, checkpoint_dir)
        async_request.finalize_fns.append(metadata_finalize_fn)
        logger.info("[DEBUG dist_ckpt.save] <<< EXIT (async request created)")
        return async_request

    # Patch both the module-level references so save_checkpoint picks up our wrappers
    _dist_ckpt_module.save = _debug_dist_save
    _ser_module.save_preprocess = _debug_save_preprocess
    try:
        logger.info(f"[DEBUG save_checkpoint] >>> save_checkpoint START (iteration={iteration})")
        save_checkpoint(
            iteration,
            model,
            optimizer,
            opt_param_scheduler,
            num_floating_point_operations_so_far=0,
            checkpointing_context=None,
            train_data_iterator=None,
            preprocess_common_state_dict_fn=None,
        )
        logger.info(f"[DEBUG save_checkpoint] <<< save_checkpoint DONE (iteration={iteration})")
    finally:
        _dist_ckpt_module.save = _orig_dist_save
        _ser_module.save_preprocess = _orig_save_preprocess

    if should_disable_forward_pre_hook(args):
        enable_forward_pre_hook(model)


def save_hf_model(args, rollout_id: int, model: Sequence[DDP]) -> None:
    """Save Megatron model in HuggingFace format.

    Args:
        model (Sequence[DDP]): Sequence of DDP-wrapped model chunks.
        rollout_id (int): Rollout ID for path formatting.
    """
    should_log = (
        mpu.get_data_parallel_rank(with_context_parallel=True) == 0 and mpu.get_tensor_model_parallel_rank() == 0
    )

    try:
        from megatron.bridge import AutoBridge

        from miles.utils.megatron_bridge_utils import patch_megatron_model

        path = Path(args.save_hf.format(rollout_id=rollout_id))

        if should_log:
            logger.info(f"Saving model in HuggingFace format to {path}")

        bridge = AutoBridge.from_hf_pretrained(args.hf_checkpoint, trust_remote_code=True)

        path.mkdir(parents=True, exist_ok=True)

        with patch_megatron_model(model):
            bridge.save_hf_pretrained(
                model,
                path=path,
            )

        if should_log:
            logger.info(f"Successfully saved HuggingFace model to {path}")
    except Exception as e:
        if should_log:
            logger.error(f"Failed to save HuggingFace format: {e}")


def initialize_model_and_optimizer(
    args: Namespace, role: str = "actor"
) -> tuple[list[DDP], MegatronOptimizer, OptimizerParamScheduler, int]:
    """Initialize model(s), optimizer, scheduler, and load from checkpoint.

    Args:
        args (Namespace): Runtime arguments.
        role (str): Logical role of the model (e.g., "actor", "critic").

    Returns:
        tuple[list[DDP], MegatronOptimizer, OptimizerParamScheduler, int]:
            DDP-wrapped model chunks, optimizer, scheduler, and iteration index.
    """

    if torch.version.hip:
        import megatron.core.dist_checkpointing.strategies.filesystem_async as filesystem_async_module

        from miles.utils.rocm_checkpoint_writer import ROCmFileSystemWriterAsync

        filesystem_async_module.FileSystemWriterAsync = ROCmFileSystemWriterAsync
        print("[ROCm] Applied FileSystemWriterAsync patch for HIP compatibility")

    model, optimizer, opt_param_scheduler = setup_model_and_optimizer(args, role)
    model[0].role = role
    clear_memory()
    iteration, _ = load_checkpoint(
        model,
        optimizer,
        opt_param_scheduler,
        checkpointing_context={},
        skip_load_to_model_and_opt=False,
    )
    clear_memory()

    opt_param_scheduler.step(increment=iteration * args.global_batch_size)

    return model, optimizer, opt_param_scheduler, iteration
