#!/usr/bin/env bash
# Minimal single-node release, adapted from the author's shared launch script.
# Deliberately does not stop Ray, kill processes, or silently attach to a cluster.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
RELEASE_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python3}
VERL_DIR=${VERL_DIR:-${RELEASE_ROOT}/third_party/verl}
MODEL_PATH=${MODEL_PATH:-${RELEASE_ROOT}/model/Qwen3-1.7B-Base}
TEACHER_MODEL_PATH=${TEACHER_MODEL_PATH:-${RELEASE_ROOT}/model/Qwen3-30B-A3B-Instruct-2507}
MATH_REASONING_DATA_DIR=${MATH_REASONING_DATA_DIR:-${RELEASE_ROOT}/recipe/data/math_reasoning_parquet}
TRAIN_FILE=${TRAIN_FILE:-${MATH_REASONING_DATA_DIR}/dapo_math_17k.parquet}
AIME24_FILE=${AIME24_FILE:-${MATH_REASONING_DATA_DIR}/aime24.parquet}
AIME25_FILE=${AIME25_FILE:-${MATH_REASONING_DATA_DIR}/aime25.parquet}
AIME26_FILE=${AIME26_FILE:-${MATH_REASONING_DATA_DIR}/aime26.parquet}
CUSTOM_REWARD_PATH=${CUSTOM_REWARD_PATH:-${SCRIPT_DIR}/math_verify_reward.py}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-128}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
NNODES=${NNODES:-1}
ROLLOUT_N=${ROLLOUT_N:-8}
VAL_N=${VAL_N:-16}
SEED=${SEED:-42}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-16384}
ROLLOUT_TP=${ROLLOUT_TP:-1}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.70}
TEST_FREQ=${TEST_FREQ:-10}
SAVE_FREQ=${SAVE_FREQ:--1}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-10}
DRY_RUN=${DRY_RUN:-0}
PYTHON_ONLY_ARGS=${PYTHON_ONLY_ARGS:-0}
PROJECT_NAME=topd_qwen3_1p7b_30ba3b_dapo17k
EXP_NAME=${EXP_NAME:-${METHOD_TAG}_b${TRAIN_BATCH_SIZE}_avg${VAL_N}_seed${SEED}}
RUN_DIR=${RUN_DIR:-${RELEASE_ROOT}/recipe/runs/${EXP_NAME}}
LOGGER=${LOGGER:-'["console","tensorboard"]'}

for integer in TRAIN_BATCH_SIZE PPO_MINI_BATCH_SIZE GPUS_PER_NODE ROLLOUT_N VAL_N ROLLOUT_TP MAX_PROMPT_LENGTH MAX_RESPONSE_LENGTH; do
    value=${!integer:-}
    if ! [[ "${value}" =~ ^[1-9][0-9]*$ ]]; then
        printf '%s must be a positive integer.\n' "${integer}" >&2
        exit 1
    fi
done
if [ $((GPUS_PER_NODE % ROLLOUT_TP)) -ne 0 ] || [ $((PPO_MINI_BATCH_SIZE * ROLLOUT_N % GPUS_PER_NODE)) -ne 0 ]; then
    printf '%s\n' "GPU count must fit rollout TP and each PPO minibatch must divide across GPUs." >&2
    exit 1
fi
for path_var in MODEL_PATH TEACHER_MODEL_PATH TRAIN_FILE AIME24_FILE AIME25_FILE AIME26_FILE CUSTOM_REWARD_PATH RUN_DIR; do
    case "${!path_var}" in
        *\'*|*$'\n'*) printf 'Unsupported quote/newline in %s.\n' "${path_var}" >&2; exit 1 ;;
    esac
done

for required in ADV_ESTIMATOR METHOD_TAG PPO_MINI_BATCH_SIZE; do
    if [ -z "${!required:-}" ]; then
        printf 'Missing entry-script variable: %s\n' "${required}" >&2
        exit 1
    fi
done
if [ "${NNODES}" != "1" ]; then
    printf '%s\n' "This release launcher is single-node; multi-node setup is not validated." >&2
    exit 1
fi
if [ "${TRAIN_BATCH_SIZE}" -lt "${PPO_MINI_BATCH_SIZE}" ] || [ $((TRAIN_BATCH_SIZE % PPO_MINI_BATCH_SIZE)) -ne 0 ]; then
    printf '%s\n' "TRAIN_BATCH_SIZE must be divisible by PPO_MINI_BATCH_SIZE." >&2
    exit 1
fi
if [ $((TRAIN_BATCH_SIZE * ROLLOUT_N % GPUS_PER_NODE)) -ne 0 ]; then
    printf '%s\n' "Prompt batch times rollout count must be divisible by GPU count." >&2
    exit 1
fi

if [ -n "${VAL_FILES:-}" ]; then
    VAL_FILES_LIST=${VAL_FILES}
else
    VAL_FILES_LIST="["
    separator=""
    for f in "${AIME24_FILE}" "${AIME25_FILE}" "${AIME26_FILE}"; do
        if [ -f "${f}" ] || { [ "${DRY_RUN}" = "1" ] && [ "${f}" != "${AIME26_FILE}" ]; }; then
            VAL_FILES_LIST+="${separator}'${f}'"
            separator=","
        fi
    done
    VAL_FILES_LIST+="]"
    if [ "${VAL_FILES_LIST}" = "[]" ]; then
        printf '%s\n' "No validation parquet found. Run scripts/prepare_data.py or set VAL_FILES." >&2
        exit 1
    fi
fi

export PYTHONPATH=${VERL_DIR}${PYTHONPATH:+:${PYTHONPATH}}
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
export WANDB_MODE=${WANDB_MODE:-offline}
export TENSORBOARD_DIR=${RUN_DIR}/tensorboard
export PYTHONHASHSEED=${SEED}

# The pinned legacy FSDP worker already supports ref.model.path. Both models
# score exactly the same student token IDs; the teacher never generates rollouts.
ARGS=(
    "+ray_kwargs.ray_init.address=local"
    "algorithm.adv_estimator=${ADV_ESTIMATOR}"
    "algorithm.use_kl_in_reward=False"
    "algorithm.kl_ctrl.kl_coef=0.0"
    "data.train_files=['${TRAIN_FILE}']"
    "data.val_files=${VAL_FILES_LIST}"
    "data.prompt_key=prompt"
    "data.return_raw_chat=True"
    "data.truncation=error"
    "data.max_prompt_length=${MAX_PROMPT_LENGTH}"
    "data.max_response_length=${MAX_RESPONSE_LENGTH}"
    "data.train_batch_size=${TRAIN_BATCH_SIZE}"
    "data.shuffle=True"
    "data.seed=${SEED}"
    "data.dataloader_num_workers=0"
    "data.validation_shuffle=False"
    "data.filter_overlong_prompts=True"
    "actor_rollout_ref.hybrid_engine=True"
    "actor_rollout_ref.nccl_timeout=3600"
    "actor_rollout_ref.model.path='${MODEL_PATH}'"
    "actor_rollout_ref.model.use_remove_padding=True"
    "actor_rollout_ref.model.enable_gradient_checkpointing=True"
    "actor_rollout_ref.model.lora_rank=0"
    "actor_rollout_ref.actor.strategy=fsdp"
    "actor_rollout_ref.actor.optim.lr=1e-6"
    "actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}"
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1"
    "actor_rollout_ref.actor.ppo_epochs=1"
    "actor_rollout_ref.actor.data_loader_seed=${SEED}"
    "actor_rollout_ref.actor.fsdp_config.seed=${SEED}"
    "actor_rollout_ref.actor.fsdp_config.use_torch_compile=False"
    "actor_rollout_ref.actor.use_kl_loss=False"
    "actor_rollout_ref.actor.kl_loss_coef=0.0"
    "actor_rollout_ref.actor.clip_ratio=0.2"
    "actor_rollout_ref.actor.clip_ratio_low=0.2"
    "actor_rollout_ref.actor.clip_ratio_high=0.2"
    "actor_rollout_ref.actor.clip_ratio_c=3.0"
    "actor_rollout_ref.actor.entropy_coeff=0.0"
    "actor_rollout_ref.actor.use_dynamic_bsz=False"
    "actor_rollout_ref.actor.use_torch_compile=False"
    "actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean"
    "actor_rollout_ref.actor.policy_loss.loss_mode=vanilla"
    "actor_rollout_ref.actor.fsdp_config.param_offload=False"
    "actor_rollout_ref.actor.fsdp_config.optimizer_offload=False"
    "actor_rollout_ref.rollout.name=vllm"
    "actor_rollout_ref.rollout.mode=async"
    "actor_rollout_ref.rollout.enforce_eager=True"
    "actor_rollout_ref.rollout.seed=${SEED}"
    "actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}"
    "actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEMORY_UTILIZATION}"
    "actor_rollout_ref.rollout.n=${ROLLOUT_N}"
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1"
    "actor_rollout_ref.rollout.max_num_seqs=128"
    "actor_rollout_ref.rollout.max_num_batched_tokens=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))"
    "actor_rollout_ref.rollout.max_model_len=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))"
    "actor_rollout_ref.rollout.temperature=1.0"
    "actor_rollout_ref.rollout.top_p=1.0"
    "actor_rollout_ref.rollout.multi_turn.enable=False"
    "actor_rollout_ref.rollout.val_kwargs.do_sample=True"
    "actor_rollout_ref.rollout.val_kwargs.temperature=1.0"
    "actor_rollout_ref.rollout.val_kwargs.top_p=0.7"
    "actor_rollout_ref.rollout.val_kwargs.n=${VAL_N}"
    "+actor_rollout_ref.ref.model.path='${TEACHER_MODEL_PATH}'"
    "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1"
    "actor_rollout_ref.ref.use_torch_compile=False"
    "actor_rollout_ref.ref.fsdp_config.seed=${SEED}"
    "actor_rollout_ref.ref.fsdp_config.use_torch_compile=False"
    "actor_rollout_ref.ref.fsdp_config.param_offload=True"
    "actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16"
    "critic.enable=False"
    "reward.reward_model.enable=False"
    "reward.custom_reward_function.path='${CUSTOM_REWARD_PATH}'"
    "reward.custom_reward_function.name=compute_score"
    "trainer.use_legacy_worker_impl=enable"
    "trainer.critic_warmup=0"
    "trainer.logger=${LOGGER}"
    "trainer.project_name=${PROJECT_NAME}"
    "trainer.experiment_name=${EXP_NAME}"
    "trainer.n_gpus_per_node=${GPUS_PER_NODE}"
    "trainer.nnodes=${NNODES}"
    "trainer.save_freq=${SAVE_FREQ}"
    "trainer.test_freq=${TEST_FREQ}"
    "trainer.default_local_dir='${RUN_DIR}/checkpoints'"
    "trainer.resume_mode=disable"
    "trainer.total_epochs=${TOTAL_EPOCHS}"
    "trainer.val_before_train=True"
    "trainer.log_val_generations=0"
)
if [ -n "${TOPD_ALPHA:-}" ]; then
    ARGS+=("+algorithm.topd_alpha=${TOPD_ALPHA}")
fi
ARGS+=("$@")

if [ "${DRY_RUN}" = "1" ]; then
    if [ "${PYTHON_ONLY_ARGS}" = "1" ]; then
        "${PYTHON_BIN}" -c 'import json,sys; print(json.dumps(sys.argv[1:]))' "${ARGS[@]}"
    else
        printf 'DRY RUN ONLY: no models, Ray, files, or training started.\n'
        printf 'Teacher: %s\nPrompt batch: %s\nValidation: avg@%s\n' "${TEACHER_MODEL_PATH}" "${TRAIN_BATCH_SIZE}" "${VAL_N}"
        printf '%q ' "${PYTHON_BIN}" -m verl.trainer.main_ppo "${ARGS[@]}"
        printf '\n'
    fi
    return 0
fi

if [ -n "${RAY_ADDRESS:-}" ] && [ "${RAY_ADDRESS}" != "local" ]; then
    printf '%s\n' "RAY_ADDRESS points to an existing cluster. Unset it for this single-node launcher; nothing was stopped." >&2
    exit 1
fi
"${PYTHON_BIN}" "${RELEASE_ROOT}/scripts/apply_patch.py" --verl-dir "${VERL_DIR}" --require-applied
"${PYTHON_BIN}" "${RELEASE_ROOT}/scripts/preflight.py" --student "${MODEL_PATH}" --teacher "${TEACHER_MODEL_PATH}" --train "${TRAIN_FILE}" --validation-list "${VAL_FILES_LIST}" --gpus "${GPUS_PER_NODE}" --verl-dir "${VERL_DIR}"
mkdir -p -- "${RUN_DIR}/logs" "${TENSORBOARD_DIR}"
# Local Ray is created by verl; no ray stop, force reset, or global process cleanup.
"${PYTHON_BIN}" -m verl.trainer.main_ppo "${ARGS[@]}" 2>&1 | tee "${RUN_DIR}/logs/train_$(date +%Y%m%d_%H%M%S).log"
