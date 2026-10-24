# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Single Process Actor
"""

import logging
import os

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DTensor

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.utils.attention_utils import index_first_axis, pad_input, rearrange, unpad_input
from verl.utils.device import get_device_id, get_device_name
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.model import compute_position_id_with_mask
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_dtypes import PrecisionType
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor
from verl.workers.config import ActorConfig

__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOActor(BasePPOActor):
    """FSDP DataParallel PPO Actor or Ref worker

    Args:
        config (ActorConfig): Actor config
        actor_module (nn.Module): Actor or ref module
        actor_optimizer (torch.optim.Optimizer, optional): Actor optimizer. Defaults to None.
    """

    def __init__(
        self,
        config: ActorConfig,
        actor_module: nn.Module,
        actor_optimizer: torch.optim.Optimizer = None,
        tokenizer=None,
    ):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.tokenizer = tokenizer
        role = "Ref" if actor_optimizer is None else "Actor"

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        self.use_dynamic_bsz = self.config.get("use_dynamic_bsz", False)

        self.use_prefix_grouper = self.config.get("use_prefix_grouper", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_prefix_grouper={self.use_prefix_grouper}")

        if self.config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  # use torch compile by default
            else entropy_from_logits
        )
        self.device_name = get_device_name()
        self.param_dtype = PrecisionType.to_dtype(self.config.fsdp_config.get("dtype", "bfloat16"))
        if self.param_dtype == torch.float16:
            from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

            self.scaler = ShardedGradScaler(growth_interval=400)
        else:
            self.scaler = None

        # Sum of squared probabilities computation (for optimal_token_baseline)
        # Only initialize if calculate_sum_pi_squared config is enabled
        if self.config.get("calculate_sum_pi_squared", False):
            self.calculate_sum_pi_squared_from_logits = (
                torch.compile(verl_F.calculate_sum_pi_squared_from_logits, dynamic=True)
                if self.config.get("use_torch_compile", True)
                else verl_F.calculate_sum_pi_squared_from_logits
            )
            assert not (self.use_fused_kernels or self.use_prefix_grouper), (
                "calculate_sum_pi_squared is not supported with "
                f"{self.use_fused_kernels=} or {self.use_prefix_grouper=} for now."
            )

    def _forward_micro_batch(
        self,
        micro_batch: dict[str, torch.Tensor],
        temperature: float,
        calculate_entropy: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Returns:
            dict[str, torch.Tensor]:
                log_probs: (bs, response_len)
                if calculate_entropy is True:
                    entropys: (bs, response_len)
                if calculate_sum_pi_squared is False:
                    sum_pi_squared: (bs, response_len)
        """
        calculate_sum_pi_squared = self.config.get("calculate_sum_pi_squared", False)
        sum_pi_squared_checkpointing = self.config.get("sum_pi_squared_checkpointing", False)
        # PrefixGrouper path for shared-prefix optimization
        if self.use_prefix_grouper:
            can_use_pg = (
                not self.use_remove_padding
                and not self.use_ulysses_sp
                and not self.use_fused_kernels
                and not self.use_dynamic_bsz
            )
            if can_use_pg and "response_mask" in micro_batch and "uid" in micro_batch:
                from verl.trainer.ppo.prefix_grouper_utils import forward_micro_batch_with_prefix_grouper

                return forward_micro_batch_with_prefix_grouper(
                    micro_batch=micro_batch,
                    model=self.actor_module,
                    temperature=temperature,
                    calculate_entropy=calculate_entropy,
                    device_name=self.device_name,
                    param_dtype=self.param_dtype,
                    use_chunking_entropy=self.config.get("entropy_from_logits_with_chunking", False),
                )

        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            from verl.utils.model import extract_multi_modal_inputs

            multi_modal_inputs = extract_multi_modal_inputs(micro_batch["multi_modal_inputs"])

        with torch.autocast(device_type=self.device_name, dtype=self.param_dtype):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                is_mask_all_zero = attention_mask.sum() == 0
                if is_mask_all_zero:
                    input_ids_rmpad = torch.zeros(
                        (1, self.ulysses_sequence_parallel_size),
                        device=input_ids.device,
                        dtype=input_ids.dtype,
                    )
                    if position_ids.dim() == 3:
                        position_ids_rmpad = torch.zeros(
                            (position_ids.shape[0], 1, self.ulysses_sequence_parallel_size),
                            device=position_ids.device,
                            dtype=position_ids.dtype,
                        )
                    else:
                        position_ids_rmpad = torch.zeros(
                            (1, self.ulysses_sequence_parallel_size),
                            device=position_ids.device,
                            dtype=position_ids.dtype,
                        )

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = hasattr(
                        getattr(self.actor_module, "module", self.actor_module).config, "vision_config"
                    )
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        # ((total_nnz / sp) + pad)
                        entropy_rmpad = (
                            self.compute_entropy_from_logits(logits_rmpad)
                            if not self.config.entropy_checkpointing
                            else torch.utils.checkpoint.checkpoint(self.compute_entropy_from_logits, logits_rmpad)
                        )

                    # Compute sum_pi_squared if requested (for optimal_token_baseline)
                    if calculate_sum_pi_squared:
                        sum_pi_squared_rmpad = (
                            self.calculate_sum_pi_squared_from_logits(logits_rmpad)
                            if not sum_pi_squared_checkpointing
                            else torch.utils.checkpoint.checkpoint(
                                self.calculate_sum_pi_squared_from_logits, logits_rmpad
                            )
                        )

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outputs_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outputs_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                    if calculate_sum_pi_squared:
                        sum_pi_squared_rmpad = gather_outputs_and_unpad(
                            sum_pi_squared_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size
                        )

                if is_mask_all_zero:
                    log_probs = log_probs[:0]
                    if calculate_entropy:
                        entropy_rmpad = entropy_rmpad[:0]

                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                if calculate_sum_pi_squared:
                    full_sum_pi_squared = pad_input(
                        hidden_states=sum_pi_squared_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                if calculate_sum_pi_squared:
                    # (bsz, response_length)
                    sum_pi_squared = full_sum_pi_squared.squeeze(-1)[:, -response_length - 1 : -1]
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                        else:
                            entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)
                    # Compute sum_pi_squared if requested (for optimal_token_baseline)
                    if calculate_sum_pi_squared:
                        sum_pi_squared = (
                            self.calculate_sum_pi_squared_from_logits(logits)
                            if not sum_pi_squared_checkpointing
                            else torch.utils.checkpoint.checkpoint(self.calculate_sum_pi_squared_from_logits, logits)
                        )

            outputs = {"log_probs": log_probs}
            if calculate_entropy:
                outputs["entropys"] = entropy
            if calculate_sum_pi_squared:
                outputs["sum_pi_squared"] = sum_pi_squared
            return outputs

    def _optimizer_step(self):
        if self._is_tide_optimizer_enabled():
            return self._optimizer_step_without_grad_clip()

        assert self.config.grad_clip is not None
        if self.scaler is not None:
            self.scaler.unscale_(self.actor_optimizer)
        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        if isinstance(grad_norm, DTensor):
            grad_norm = grad_norm.full_tensor()

        # if grad_norm is not finite, skip the update
        if self.scaler is not None:
            self.scaler.step(self.actor_optimizer)
            self.scaler.update()
        else:
            if not torch.isfinite(grad_norm):
                print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
                self.actor_optimizer.zero_grad()
            else:
                self.actor_optimizer.step()

        # Clear cached weight scales for QAT (weights changed)
        if getattr(self.actor_module, "_qat_fuse_enabled", False):
            from verl.utils.qat import invalidate_all_scales

            invalidate_all_scales(self.actor_module)

        return grad_norm

    def _iter_trainable_params(self):
        return (param for param in self.actor_module.parameters() if param.requires_grad)

    def _compute_grad_norm(self) -> torch.Tensor:
        first_param = next(self._iter_trainable_params(), None)
        device = first_param.device if first_param is not None else torch.device(self.device_name)
        grad_norm_sq = torch.zeros((), device=device, dtype=torch.float32)
        for param in self._iter_trainable_params():
            if param.grad is not None:
                grad_norm_sq = grad_norm_sq + param.grad.detach().float().pow(2).sum()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(grad_norm_sq, op=torch.distributed.ReduceOp.SUM)
        return torch.sqrt(grad_norm_sq.clamp_min(0.0))

    def _apply_fisher_fd_perturbation(self, eps: float, normalize_direction: bool) -> torch.Tensor:
        grad_norm = self._compute_grad_norm()
        denom = grad_norm.clamp_min(1e-12) if normalize_direction else torch.ones_like(grad_norm)
        with torch.no_grad():
            for param in self._iter_trainable_params():
                if param.grad is not None:
                    param.add_(param.grad, alpha=-eps / denom.item())
        return grad_norm

    def _restore_fisher_fd_perturbation(self, eps: float, normalize_direction: bool, grad_norm: torch.Tensor):
        denom = grad_norm.clamp_min(1e-12) if normalize_direction else torch.ones_like(grad_norm)
        with torch.no_grad():
            for param in self._iter_trainable_params():
                if param.grad is not None:
                    param.add_(param.grad, alpha=eps / denom.item())

    def _is_meet_optimizer_enabled(self) -> bool:
        if self.actor_optimizer is None:
            return False
        optim_config = self.config.get("optim", {})
        return optim_config.get("optimizer", "").lower() == "meet"

    def _is_tide_optimizer_enabled(self) -> bool:
        if self.actor_optimizer is None:
            return False
        optim_config = self.config.get("optim", {})
        return optim_config.get("optimizer", "").lower() == "tide"

    def _optimizer_step_without_grad_clip(self):
        if self.scaler is not None:
            raise NotImplementedError("TIDE optimizer does not support fp16 gradient scaling in DataParallelPPOActor yet.")
        grad_norm = self._compute_grad_norm()
        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()

        # Clear cached weight scales for QAT (weights changed)
        if getattr(self.actor_module, "_qat_fuse_enabled", False):
            from verl.utils.qat import invalidate_all_scales

            invalidate_all_scales(self.actor_module)

        return grad_norm

    def _clone_trainable_grads(self) -> list[torch.Tensor | None]:
        return [None if param.grad is None else param.grad.detach().clone() for param in self._iter_trainable_params()]

    @staticmethod
    def _meet_direction(grad_a: torch.Tensor, grad_b: torch.Tensor) -> torch.Tensor:
        return 0.5 * (grad_a.sign() + grad_b.sign()) * torch.minimum(grad_a.abs(), grad_b.abs())

    def _assign_meet_grads(self, grads_a: list[torch.Tensor | None], grads_b: list[torch.Tensor | None]):
        for param, grad_a, grad_b in zip(self._iter_trainable_params(), grads_a, grads_b):
            if grad_a is None or grad_b is None:
                param.grad = None
                continue
            param.grad = self._meet_direction(grad_a, grad_b).to(dtype=param.dtype)

    def _optimizer_step_with_prepared_grads(self):
        if self.scaler is not None:
            raise NotImplementedError("MEET optimizer does not support fp16 gradient scaling in DataParallelPPOActor yet.")
        grad_norm = self._compute_grad_norm()
        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()

        # Clear cached weight scales for QAT (weights changed)
        if getattr(self.actor_module, "_qat_fuse_enabled", False):
            from verl.utils.qat import invalidate_all_scales

            invalidate_all_scales(self.actor_module)

        return grad_norm

    def _meet_backward_and_step(self, micro_batches, compute_loss_fn, metrics: dict, pad_token_id: int):
        if len(micro_batches) < 2:
            raise ValueError("MEET optimizer requires at least two independent micro-batches per mini-batch.")
        if self.scaler is not None:
            raise NotImplementedError("MEET optimizer does not support fp16 gradient scaling in DataParallelPPOActor yet.")

        first_micro_batches = micro_batches[0::2]
        second_micro_batches = micro_batches[1::2]
        pair_count = min(len(first_micro_batches), len(second_micro_batches))
        if pair_count == 0:
            raise ValueError("MEET optimizer requires at least one pair of micro-batches.")
        if len(first_micro_batches) != len(second_micro_batches) and torch.distributed.get_rank() == 0:
            print("WARN: MEET optimizer drops the last unpaired micro-batch in a mini-batch.")

        meet_loss_scale_factor = 1 / pair_count
        meet_metric_scale_factor = 0.5 / pair_count

        self.actor_optimizer.zero_grad()
        for micro_batch in first_micro_batches[:pair_count]:
            micro_batch = micro_batch.to(get_device_id())
            loss = compute_loss_fn(micro_batch, meet_loss_scale_factor, metrics, pad_token_id, meet_metric_scale_factor)
            loss.backward()
        grads_a = self._clone_trainable_grads()

        self.actor_optimizer.zero_grad()
        for micro_batch in second_micro_batches[:pair_count]:
            micro_batch = micro_batch.to(get_device_id())
            loss = compute_loss_fn(micro_batch, meet_loss_scale_factor, metrics, pad_token_id, meet_metric_scale_factor)
            loss.backward()
        grads_b = self._clone_trainable_grads()

        self.actor_optimizer.zero_grad()
        self._assign_meet_grads(grads_a, grads_b)
        grad_norm = self._optimizer_step_with_prepared_grads()
        self.actor_optimizer.zero_grad()
        return grad_norm

    def _postprocess_fisher_fd_credit(
        self, credit: torch.Tensor, advantages: torch.Tensor, response_mask: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        policy_loss_config = self.config.policy_loss
        response_mask_bool = response_mask.bool()
        valid_credit = credit.detach()[response_mask_bool]
        metrics = {}
        if valid_credit.numel() > 0:
            metrics["actor/fisher_fd_credit_mean"] = valid_credit.mean().item()
            metrics["actor/fisher_fd_credit_std"] = valid_credit.std(unbiased=False).item()
            metrics["actor/fisher_fd_credit_abs_mean"] = valid_credit.abs().mean().item()
        else:
            metrics["actor/fisher_fd_credit_mean"] = 0.0
            metrics["actor/fisher_fd_credit_std"] = 0.0
            metrics["actor/fisher_fd_credit_abs_mean"] = 0.0

        credit = credit * response_mask
        if policy_loss_config.get("fisher_fd_standardize_credit", False) and valid_credit.numel() > 1:
            credit_mean = valid_credit.mean()
            credit_std = valid_credit.std(unbiased=False).clamp_min(1e-6)
            credit = (credit - credit_mean) / credit_std
            credit = credit * response_mask

        clip_credit = policy_loss_config.get("fisher_fd_clip_credit", 0.0)
        if clip_credit is not None and clip_credit > 0:
            credit = torch.clamp(credit, min=-clip_credit, max=clip_credit)

        credit_coef = policy_loss_config.get("fisher_fd_credit_coef", 1.0)
        if credit_coef != 1.0:
            credit = credit_coef * credit + (1.0 - credit_coef) * advantages

        return credit.detach(), metrics

    def _build_sft_teacher_batch(
        self,
        model_inputs: dict[str, torch.Tensor],
        pad_token_id: int,
    ) -> dict[str, torch.Tensor]:
        if self.tokenizer is None:
            raise RuntimeError("fisher_sft_fd requires tokenizer to build teacher-forcing SFT batches.")
        if "sft_response" not in model_inputs:
            raise RuntimeError("fisher_sft_fd requires a non-tensor `sft_response` field in the training data.")

        prompts = model_inputs["prompts"]
        response_length = model_inputs["responses"].size(-1)
        prompt_attention_mask = model_inputs["attention_mask"][:, : prompts.size(-1)]
        teacher_responses = model_inputs["sft_response"]
        if not isinstance(teacher_responses, (list, tuple)):
            teacher_responses = teacher_responses.tolist()

        sft_loss_mask = model_inputs.get("sft_loss_mask", None)
        if sft_loss_mask is not None:
            sft_loss_mask = sft_loss_mask.to(device=prompts.device, dtype=torch.bool)

        teacher_token_ids = []
        device = prompts.device
        for idx, response in enumerate(teacher_responses):
            use_teacher = sft_loss_mask is None or bool(sft_loss_mask[idx].item())
            if not use_teacher:
                token_ids = []
            else:
                if isinstance(response, bytes):
                    response = response.decode("utf-8")
                token_ids = self.tokenizer.encode(str(response), add_special_tokens=False)
                if len(token_ids) == 0:
                    token_ids = [self.tokenizer.eos_token_id or pad_token_id]
            token_ids = token_ids[:response_length]
            if len(token_ids) < response_length:
                token_ids = token_ids + [pad_token_id] * (response_length - len(token_ids))
            teacher_token_ids.append(token_ids)

        teacher_responses_tensor = torch.tensor(teacher_token_ids, dtype=prompts.dtype, device=device)
        teacher_response_mask = teacher_responses_tensor.ne(pad_token_id).to(dtype=prompt_attention_mask.dtype)
        if sft_loss_mask is not None:
            teacher_response_mask = teacher_response_mask * sft_loss_mask[:, None].to(dtype=teacher_response_mask.dtype)
        input_ids = torch.cat([prompts, teacher_responses_tensor], dim=-1)
        attention_mask = torch.cat([prompt_attention_mask, teacher_response_mask], dim=-1)
        position_ids = compute_position_id_with_mask(attention_mask)
        return {
            "responses": teacher_responses_tensor,
            "response_mask": teacher_response_mask,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }

    def _compute_sft_loss_from_teacher_inputs(
        self,
        teacher_inputs: dict[str, torch.Tensor],
        temperature: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self._forward_micro_batch(teacher_inputs, temperature=temperature, calculate_entropy=False)
        log_prob = outputs["log_probs"]
        response_mask = teacher_inputs["response_mask"]
        if response_mask.sum() == 0:
            sft_loss = log_prob.sum() * 0.0
        else:
            sft_loss_mat = -log_prob
            sft_loss = agg_loss(loss_mat=sft_loss_mat, loss_mask=response_mask, loss_agg_mode=self.config.loss_agg_mode)
        return sft_loss, log_prob.detach()

    def _compute_sft_direction_loss(
        self,
        teacher_inputs: dict[str, torch.Tensor],
        temperature: float,
        loss_scale_factor: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sft_loss, log_prob = self._compute_sft_loss_from_teacher_inputs(
            teacher_inputs=teacher_inputs,
            temperature=temperature,
        )
        (sft_loss * loss_scale_factor).backward()
        return log_prob, sft_loss.detach()

    def _compute_fisher_sft_fd_token_credits(
        self,
        micro_batches: list[DataProto],
        temperature: float,
        pad_token_id: int,
        loss_scale_batch_size: int | None = None,
        sft_loss_scale_batch_size: int | None = None,
        sft_micro_batches: list[DataProto] | None = None,
    ) -> tuple[list[torch.Tensor], dict[str, float | list[float]]]:
        if self.scaler is not None:
            raise RuntimeError("fisher_sft_fd currently supports bf16/fp32 training only, not fp16 GradScaler.")

        policy_loss_config = self.config.policy_loss
        eps = policy_loss_config.get("fisher_fd_eps", 1e-3)
        normalize_direction = policy_loss_config.get("fisher_fd_normalize_direction", True)
        reuse_old_log_probs = policy_loss_config.get("fisher_fd_reuse_old_log_probs", loss_scale_batch_size is not None)

        base_log_probs = []
        model_inputs_list = []
        advantages_list = []
        response_masks = []
        direction_metrics = {
            "actor/fisher_sft_fd_teacher_loss": [],
            "actor/fisher_sft_fd_reuse_old_log_probs": float(reuse_old_log_probs),
        }

        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch, "pad_token_id": pad_token_id}
            response_mask = model_inputs["response_mask"]
            advantages = model_inputs["advantages"]
            base_log_prob = model_inputs.get("old_log_probs", None) if reuse_old_log_probs else None
            if base_log_prob is None:
                with torch.no_grad():
                    rollout_outputs = self._forward_micro_batch(
                        model_inputs, temperature=temperature, calculate_entropy=False
                    )
                    base_log_prob = rollout_outputs["log_probs"].detach()
            else:
                base_log_prob = base_log_prob.detach()

            base_log_probs.append(base_log_prob)
            model_inputs_list.append(model_inputs)
            advantages_list.append(advantages.detach())
            response_masks.append(response_mask.detach())

        direction_batches = sft_micro_batches if sft_micro_batches is not None else micro_batches
        self.actor_optimizer.zero_grad()
        for micro_batch in direction_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch, "pad_token_id": pad_token_id}
            response_mask = model_inputs["response_mask"]
            sft_loss_mask = model_inputs.get("sft_loss_mask", None)
            if sft_loss_mask is not None:
                sft_loss_mask = sft_loss_mask.to(device=response_mask.device, dtype=torch.bool)
                current_sft_batch_size = int(sft_loss_mask.sum().item())
            else:
                current_sft_batch_size = response_mask.shape[0]

            if self.config.use_dynamic_bsz:
                scale_batch_size = sft_loss_scale_batch_size or loss_scale_batch_size or self.config.ppo_mini_batch_size
                loss_scale_factor = current_sft_batch_size / scale_batch_size
            elif sft_loss_scale_batch_size is not None:
                loss_scale_factor = current_sft_batch_size / sft_loss_scale_batch_size
            elif loss_scale_batch_size is not None:
                loss_scale_factor = current_sft_batch_size / loss_scale_batch_size
            else:
                loss_scale_factor = 1 / self.gradient_accumulation

            teacher_inputs = self._build_sft_teacher_batch(model_inputs=model_inputs, pad_token_id=pad_token_id)
            _, sft_loss = self._compute_sft_direction_loss(
                teacher_inputs=teacher_inputs,
                temperature=temperature,
                loss_scale_factor=loss_scale_factor,
            )
            direction_metrics["actor/fisher_sft_fd_teacher_loss"].append(sft_loss.item())
            direction_metrics.setdefault("actor/fisher_sft_fd_num_teacher_samples", []).append(float(current_sft_batch_size))

        grad_norm = self._apply_fisher_fd_perturbation(eps=eps, normalize_direction=normalize_direction)
        token_credits = []
        credit_metrics = {}
        try:
            with torch.no_grad():
                for model_inputs, base_log_prob, advantages, response_mask in zip(
                    model_inputs_list, base_log_probs, advantages_list, response_masks
                ):
                    outputs = self._forward_micro_batch(model_inputs, temperature=temperature, calculate_entropy=False)
                    perturbed_log_prob = outputs["log_probs"]
                    credit = (perturbed_log_prob - base_log_prob) / eps
                    credit, current_credit_metrics = self._postprocess_fisher_fd_credit(
                        credit=credit,
                        advantages=advantages,
                        response_mask=response_mask,
                    )
                    token_credits.append(credit)
                    append_to_dict(credit_metrics, current_credit_metrics)
        finally:
            self._restore_fisher_fd_perturbation(
                eps=eps,
                normalize_direction=normalize_direction,
                grad_norm=grad_norm,
            )
            self.actor_optimizer.zero_grad()

        direction_metrics.update(credit_metrics)
        direction_metrics["actor/fisher_sft_fd_grad_norm"] = grad_norm.detach().item()
        direction_metrics["actor/fisher_sft_fd_eps"] = eps
        return token_credits, direction_metrics

    def _compute_fisher_fd_token_credits(
        self,
        micro_batches: list[DataProto],
        temperature: float,
        pad_token_id: int,
        on_policy: bool,
        loss_scale_batch_size: int | None = None,
    ) -> tuple[list[torch.Tensor], dict[str, float | list[float]]]:
        if self.scaler is not None:
            raise RuntimeError("fisher_grpo_fd currently supports bf16/fp32 training only, not fp16 GradScaler.")

        policy_loss_config = self.config.policy_loss
        base_loss_mode = policy_loss_config.get("fisher_fd_base_loss_mode", "vanilla")
        if base_loss_mode == "fisher_grpo_fd":
            base_loss_mode = "vanilla"
        base_policy_loss_fn = get_policy_loss_fn(base_loss_mode)
        loss_agg_mode = self.config.loss_agg_mode
        eps = policy_loss_config.get("fisher_fd_eps", 1e-3)
        normalize_direction = policy_loss_config.get("fisher_fd_normalize_direction", True)

        base_log_probs = []
        model_inputs_list = []
        advantages_list = []
        response_masks = []
        direction_metrics = {"actor/fisher_fd_base_pg_loss": []}

        self.actor_optimizer.zero_grad()
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch, "pad_token_id": pad_token_id}
            response_mask = model_inputs["response_mask"]
            advantages = model_inputs["advantages"]

            if self.config.use_dynamic_bsz:
                scale_batch_size = loss_scale_batch_size or self.config.ppo_mini_batch_size
                loss_scale_factor = response_mask.shape[0] / scale_batch_size
            elif loss_scale_batch_size is not None:
                loss_scale_factor = response_mask.shape[0] / loss_scale_batch_size
            else:
                loss_scale_factor = 1 / self.gradient_accumulation

            outputs = self._forward_micro_batch(model_inputs, temperature=temperature, calculate_entropy=False)
            log_prob = outputs["log_probs"]
            if hasattr(self.config, "use_rollout_log_probs") and self.config.use_rollout_log_probs:
                old_log_prob = model_inputs["old_log_probs"]
            else:
                old_log_prob = log_prob.detach() if on_policy else model_inputs["old_log_probs"]

            pg_loss, _ = base_policy_loss_fn(
                old_log_prob=old_log_prob,
                log_prob=log_prob,
                advantages=advantages,
                response_mask=response_mask,
                loss_agg_mode=loss_agg_mode,
                config=self.config,
                rollout_is_weights=model_inputs.get("rollout_is_weights", None),
            )
            (pg_loss * loss_scale_factor).backward()
            direction_metrics["actor/fisher_fd_base_pg_loss"].append(pg_loss.detach().item())

            base_log_probs.append(log_prob.detach())
            model_inputs_list.append(model_inputs)
            advantages_list.append(advantages.detach())
            response_masks.append(response_mask.detach())

        grad_norm = self._apply_fisher_fd_perturbation(eps=eps, normalize_direction=normalize_direction)
        token_credits = []
        credit_metrics = {}
        try:
            with torch.no_grad():
                for model_inputs, base_log_prob, advantages, response_mask in zip(
                    model_inputs_list, base_log_probs, advantages_list, response_masks
                ):
                    outputs = self._forward_micro_batch(model_inputs, temperature=temperature, calculate_entropy=False)
                    perturbed_log_prob = outputs["log_probs"]
                    credit = (perturbed_log_prob - base_log_prob) / eps
                    credit, current_credit_metrics = self._postprocess_fisher_fd_credit(
                        credit=credit,
                        advantages=advantages,
                        response_mask=response_mask,
                    )
                    token_credits.append(credit)
                    append_to_dict(credit_metrics, current_credit_metrics)
        finally:
            self._restore_fisher_fd_perturbation(
                eps=eps,
                normalize_direction=normalize_direction,
                grad_norm=grad_norm,
            )
            self.actor_optimizer.zero_grad()

        direction_metrics.update(credit_metrics)
        direction_metrics["actor/fisher_fd_grad_norm"] = grad_norm.detach().item()
        direction_metrics["actor/fisher_fd_eps"] = eps
        return token_credits, direction_metrics

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy: bool = False) -> dict[str, torch.Tensor]:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            dict[str, torch.Tensor]: a dict containing keys
                - ``log_probs``: tensor of shape [batch_size, response_length]. torch.float32.
                - ``entropys``: tensor of shape [batch_size, response_length]. torch.float32.
                - ``sum_pi_squared``: tensor of shape [batch_size, response_length]. torch.float32.
        """
        calculate_sum_pi_squared = self.config.get("calculate_sum_pi_squared", False)

        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        pad_token_id = data.meta_info.get("pad_token_id", 0)
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []
        if self.use_prefix_grouper:
            select_keys += [k for k in ["prompts", "response_mask"] if k in data.batch]
            if "uid" in data.non_tensor_batch:
                non_tensor_select_keys.append("uid")

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        sum_pi_squared_lst = []
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {
                **micro_batch.batch,
                **micro_batch.non_tensor_batch,
                "pad_token_id": pad_token_id,
            }
            with torch.no_grad():
                outputs = self._forward_micro_batch(
                    model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                )
            log_probs_lst.append(outputs["log_probs"])
            if calculate_entropy:
                entropy_lst.append(outputs["entropys"])
            if calculate_sum_pi_squared:
                sum_pi_squared_lst.append(outputs["sum_pi_squared"])

        log_probs = torch.concat(log_probs_lst, dim=0)
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        if calculate_sum_pi_squared:
            sum_pi_squared = torch.concat(sum_pi_squared_lst, dim=0)

        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if calculate_entropy:
                entropys = restore_dynamic_batch(entropys, batch_idx_list)
            if calculate_sum_pi_squared:
                sum_pi_squared = restore_dynamic_batch(sum_pi_squared, batch_idx_list)

        outputs = {"log_probs": log_probs}
        if calculate_entropy:
            outputs["entropys"] = entropys
        if calculate_sum_pi_squared:
            outputs["sum_pi_squared"] = sum_pi_squared
        return outputs

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        pad_token_id = data.meta_info.get("pad_token_id", 0)

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
        if (self.use_prefix_grouper or loss_mode in {"fisher_sft_fd", "sft"}) and "prompts" in data.batch.keys():
            select_keys.append("prompts")
        if loss_mode in {"fisher_sft_fd", "sft"} and "sft_loss_mask" in data.batch.keys():
            select_keys.append("sft_loss_mask")
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        # Include pre-computed IS weights if present in batch
        # Weights are computed centrally in trainer and added to batch when algorithm.rollout_is=True
        if "rollout_is_weights" in data.batch.keys():
            select_keys.append("rollout_is_weights")
        # Include rollout_log_probs for computing rollout_corr metrics in bypass mode
        if "rollout_log_probs" in data.batch.keys():
            select_keys.append("rollout_log_probs")

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = []
        if has_multi_modal_inputs:
            non_tensor_select_keys.append("multi_modal_inputs")
        if loss_mode in {"fisher_sft_fd", "sft"} and "sft_response" in data.non_tensor_batch.keys():
            non_tensor_select_keys.append("sft_response")
        if self.use_prefix_grouper and "uid" in data.non_tensor_batch.keys():
            non_tensor_select_keys.append("uid")

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        fisher_fd_scope = self.config.policy_loss.get("fisher_fd_scope", "mini_batch")
        if fisher_fd_scope not in {"mini_batch", "batch"}:
            raise ValueError(f"Invalid fisher_fd_scope: {fisher_fd_scope}. Expected 'mini_batch' or 'batch'.")

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1
        is_sft_loss = loss_mode == "sft"
        is_fisher_fd_loss = loss_mode in {"fisher_grpo_fd", "fisher_sft_fd"}
        is_meet_optimizer = self._is_meet_optimizer_enabled()
        if is_meet_optimizer and (is_sft_loss or is_fisher_fd_loss):
            raise NotImplementedError("MEET optimizer is only implemented for standard GRPO/PPO policy losses.")
        if is_meet_optimizer and self.config.use_dynamic_bsz:
            raise NotImplementedError("MEET optimizer currently requires actor_rollout_ref.actor.use_dynamic_bsz=False.")

        metrics = {
            "actor/pg_loss": 0.0,
        }

        if is_sft_loss:
            if "sft_loss_mask" in data.batch.keys():
                sft_data = data.select_idxs(data.batch["sft_loss_mask"].to(dtype=torch.bool))
            else:
                sft_data = data
            total_sft_samples = max(len(sft_data.batch), 1)
            for _ in range(self.config.ppo_epochs):
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(sft_data, max_token_len=max_token_len)
                else:
                    micro_batches = sft_data.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()
                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch, "pad_token_id": pad_token_id}
                    teacher_inputs = self._build_sft_teacher_batch(
                        model_inputs=model_inputs,
                        pad_token_id=pad_token_id,
                    )
                    sft_loss, _ = self._compute_sft_loss_from_teacher_inputs(
                        teacher_inputs=teacher_inputs,
                        temperature=temperature,
                    )
                    current_sft_batch_size = int((teacher_inputs["response_mask"].sum(dim=-1) > 0).sum().item())
                    loss_scale_factor = current_sft_batch_size / total_sft_samples
                    loss = sft_loss * loss_scale_factor
                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    metrics["actor/pg_loss"] += sft_loss.detach().item() * loss_scale_factor
                    append_to_dict(
                        metrics,
                        {
                            "actor/sft_loss": sft_loss.detach().item(),
                            "actor/sft_num_teacher_samples": float(current_sft_batch_size),
                            "actor/sft_enabled": 1.0,
                        },
                    )

                grad_norm = self._optimizer_step()
                append_to_dict(metrics, {"actor/grad_norm": grad_norm.detach().item()})
            self.actor_optimizer.zero_grad()
            return metrics

        if is_fisher_fd_loss and fisher_fd_scope == "batch":
            batch_size = int(data.batch.batch_size[0])
            if self.config.use_dynamic_bsz:
                max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                fisher_micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
            else:
                fisher_micro_batches = data.split(self.config.ppo_micro_batch_size_per_gpu)
                batch_idx_list = None

            self.actor_optimizer.zero_grad()
            if loss_mode == "fisher_sft_fd":
                if "sft_loss_mask" in data.batch.keys():
                    sft_data = data.select_idxs(data.batch["sft_loss_mask"].to(dtype=torch.bool))
                else:
                    sft_data = data
                sft_loss_scale_batch_size = len(sft_data.batch)
                if self.config.use_dynamic_bsz:
                    sft_micro_batches, _ = prepare_dynamic_batch(sft_data, max_token_len=max_token_len)
                else:
                    sft_micro_batches = sft_data.split(self.config.ppo_micro_batch_size_per_gpu)
                fisher_fd_token_credits, fisher_fd_metrics = self._compute_fisher_sft_fd_token_credits(
                    micro_batches=fisher_micro_batches,
                    temperature=temperature,
                    pad_token_id=pad_token_id,
                    loss_scale_batch_size=batch_size,
                    sft_loss_scale_batch_size=sft_loss_scale_batch_size,
                    sft_micro_batches=sft_micro_batches,
                )
            else:
                fisher_fd_token_credits, fisher_fd_metrics = self._compute_fisher_fd_token_credits(
                    micro_batches=fisher_micro_batches,
                    temperature=temperature,
                    pad_token_id=pad_token_id,
                    on_policy=on_policy,
                    loss_scale_batch_size=batch_size,
                )
            fixed_advantages = torch.concat(fisher_fd_token_credits, dim=0)
            if batch_idx_list is not None:
                fixed_advantages = restore_dynamic_batch(fixed_advantages, batch_idx_list)
            data.batch["advantages"] = fixed_advantages.to(
                device=data.batch["advantages"].device,
                dtype=data.batch["advantages"].dtype,
            )
            append_to_dict(metrics, fisher_fd_metrics)
            metrics["actor/fisher_fd_scope_batch"] = 1.0
            self.actor_optimizer.zero_grad()
            mini_batches = data.split(self.config.ppo_mini_batch_size)

        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                fisher_fd_token_credits = None
                if is_fisher_fd_loss and fisher_fd_scope == "mini_batch":
                    if loss_mode == "fisher_sft_fd":
                        if "sft_loss_mask" in mini_batch.batch.keys():
                            sft_mini_batch = mini_batch.select_idxs(mini_batch.batch["sft_loss_mask"].to(dtype=torch.bool))
                        else:
                            sft_mini_batch = mini_batch
                        sft_loss_scale_batch_size = len(sft_mini_batch.batch)
                        if self.config.use_dynamic_bsz:
                            sft_micro_batches, _ = prepare_dynamic_batch(sft_mini_batch, max_token_len=max_token_len)
                        else:
                            sft_micro_batches = sft_mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)
                        fisher_fd_token_credits, fisher_fd_metrics = self._compute_fisher_sft_fd_token_credits(
                            micro_batches=micro_batches,
                            temperature=temperature,
                            pad_token_id=pad_token_id,
                            sft_loss_scale_batch_size=sft_loss_scale_batch_size,
                            sft_micro_batches=sft_micro_batches,
                        )
                    else:
                        fisher_fd_token_credits, fisher_fd_metrics = self._compute_fisher_fd_token_credits(
                            micro_batches=micro_batches,
                            temperature=temperature,
                            pad_token_id=pad_token_id,
                            on_policy=on_policy,
                        )
                    append_to_dict(metrics, fisher_fd_metrics)
                    self.actor_optimizer.zero_grad()

                policy_loss_mode = self.config.policy_loss.get("fisher_fd_final_loss_mode", "vanilla") if is_fisher_fd_loss else loss_mode
                policy_loss_fn = get_policy_loss_fn(policy_loss_mode)
                entropy_coeff = self.config.entropy_coeff
                loss_agg_mode = self.config.loss_agg_mode
                calculate_entropy = self.config.calculate_entropy or (entropy_coeff != 0)

                def compute_policy_micro_batch_loss(
                    micro_batch, loss_scale_factor, metrics, pad_token_id, metric_scale_factor=None, micro_batch_idx=None
                ):
                    if metric_scale_factor is None:
                        metric_scale_factor = loss_scale_factor
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch, "pad_token_id": pad_token_id}
                    response_mask = model_inputs["response_mask"]
                    old_log_prob = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]

                    # all return: (bsz, response_length)
                    outputs = self._forward_micro_batch(
                        model_inputs,
                        temperature=temperature,
                        calculate_entropy=calculate_entropy,
                    )
                    log_prob = outputs["log_probs"]
                    entropy = outputs["entropys"] if calculate_entropy else None

                    # for fully_async_policy
                    if hasattr(self.config, "use_rollout_log_probs") and self.config.use_rollout_log_probs:
                        old_log_prob = model_inputs["old_log_probs"]
                    else:
                        if on_policy:
                            old_log_prob = log_prob.detach()
                        else:
                            old_log_prob = model_inputs["old_log_probs"]

                    # Extract pre-computed rollout correction weights if present
                    # Weights are computed centrally in trainer and added when algorithm.rollout_is=True
                    rollout_is_weights = model_inputs.get("rollout_is_weights", None)

                    if fisher_fd_token_credits is not None:
                        if micro_batch_idx is None:
                            raise RuntimeError("fisher_fd_token_credits requires micro_batch_idx for this micro-batch.")
                        advantages = fisher_fd_token_credits[micro_batch_idx].to(log_prob.device)
                    if is_fisher_fd_loss:
                        micro_batch_metrics["actor/fisher_fd_enabled"] = 1.0
                        micro_batch_metrics[f"actor/{loss_mode}_enabled"] = 1.0
                        micro_batch_metrics[f"actor/fisher_fd_final_loss_{policy_loss_mode}_enabled"] = 1.0

                    # Compute policy loss (any function is expected to return 2 values)
                    pg_loss, pg_metrics = policy_loss_fn(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        loss_agg_mode=loss_agg_mode,
                        config=self.config,
                        rollout_is_weights=rollout_is_weights,
                    )
                    micro_batch_metrics.update(pg_metrics)

                    # Skip if using bypass_mode loss (metrics already computed in pg_metrics)
                    rollout_log_prob = model_inputs.get("rollout_log_probs", None)
                    if loss_mode != "bypass_mode" and rollout_log_prob is not None:
                        # Compute metrics using CURRENT policy π_θ vs π_rollout
                        # Tracks evolving off-policy gap as π_θ updates during mini-batch training
                        from verl.trainer.ppo.rollout_corr_helper import compute_rollout_corr_metrics_from_logprobs

                        rollout_corr_metrics = compute_rollout_corr_metrics_from_logprobs(
                            log_prob=log_prob,
                            rollout_log_prob=rollout_log_prob,
                            response_mask=response_mask,
                        )
                        micro_batch_metrics.update(rollout_corr_metrics)

                    policy_loss = pg_loss
                    if calculate_entropy and entropy is not None:
                        entropy_agg = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                        micro_batch_metrics["actor/entropy"] = entropy_agg.detach().item()
                        if entropy_coeff != 0:
                            policy_loss -= entropy_agg * entropy_coeff

                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef

                    loss = policy_loss * loss_scale_factor
                    metrics["actor/pg_loss"] += pg_loss.detach().item() * metric_scale_factor
                    append_to_dict(metrics, micro_batch_metrics)
                    return loss

                if is_meet_optimizer:
                    grad_norm = self._meet_backward_and_step(
                        micro_batches=micro_batches,
                        compute_loss_fn=compute_policy_micro_batch_loss,
                        metrics=metrics,
                        pad_token_id=pad_token_id,
                    )
                else:
                    for micro_batch_idx, micro_batch in enumerate(micro_batches):
                        micro_batch = micro_batch.to(get_device_id())
                        loss_scale_factor = (
                            micro_batch.batch["response_mask"].shape[0] / self.config.ppo_mini_batch_size
                            if self.config.use_dynamic_bsz
                            else 1 / self.gradient_accumulation
                        )
                        loss = compute_policy_micro_batch_loss(
                            micro_batch, loss_scale_factor, metrics, pad_token_id, micro_batch_idx=micro_batch_idx
                        )
                        if self.scaler is not None:
                            self.scaler.scale(loss).backward()
                        else:
                            loss.backward()

                    grad_norm = self._optimizer_step()
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                if is_meet_optimizer:
                    mini_batch_metrics["actor/meet_optimizer_enabled"] = 1.0
                append_to_dict(metrics, mini_batch_metrics)
        self.actor_optimizer.zero_grad()
        return metrics
