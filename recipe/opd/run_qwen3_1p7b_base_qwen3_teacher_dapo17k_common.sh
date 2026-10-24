#!/usr/bin/env bash
set -euo pipefail

TRACE=${TRACE:-0}
if [ "${TRACE}" = "1" ]; then
    set -x
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
RELEASE_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)

required_entry_vars=(
    ADV_ESTIMATOR
    METHOD_TAG
    PPO_MINI_BATCH_SIZE
)

for name in "${required_entry_vars[@]}"; do
    if [ -z "${!name:-}" ]; then
        echo "${name} must be set by the entry script." >&2
        exit 1
    fi
done

WORKSPACE=${WORKSPACE:-${RELEASE_ROOT}}
VERL_DIR=${VERL_DIR:-${RELEASE_ROOT}/verl}
RUN_ROOT=${RELEASE_ROOT}/recipe/runs
PROJECT_NAME=qwen3_1p7b_base_qwen3_30a3b_teacher_dapo17k_${METHOD_TAG}_16k_kl0
EXP_NAME=${PROJECT_NAME}
RUN_DIR=${RUN_ROOT}/${EXP_NAME}
LOG_DIR=${RUN_DIR}/logs
CKPT_DIR=${RUN_DIR}/checkpoints

# Student model path. Download Qwen/Qwen3-1.7B-Base first, then keep it here
# or override MODEL_PATH with your local model directory.
MODEL_PATH=${MODEL_PATH:-${RELEASE_ROOT}/model/Qwen3-1.7B-Base}
# Release default requested by the author; teacher still uses the reference model.
TEACHER_MODEL_PATH=${TEACHER_MODEL_PATH:-${RELEASE_ROOT}/model/Qwen3-30B-A3B-Instruct-2507}
# The default teacher is MoE. Override this when using a different teacher architecture.
TEACHER_LAYER_CLS=${TEACHER_LAYER_CLS:-Qwen3MoeDecoderLayer}
# Math reasoning parquet directory. Put DAPO-17k and AIME parquet files here,
# or override MATH_REASONING_DATA_DIR / TRAIN_FILE / AIME*_FILE.
MATH_REASONING_DATA_DIR=${MATH_REASONING_DATA_DIR:-${RELEASE_ROOT}/recipe/data/math_reasoning_parquet}
TRAIN_FILE=${TRAIN_FILE:-${MATH_REASONING_DATA_DIR}/dapo_math_17k.parquet}
AIME24_FILE=${AIME24_FILE:-${MATH_REASONING_DATA_DIR}/aime24.parquet}
AIME25_FILE=${AIME25_FILE:-${MATH_REASONING_DATA_DIR}/aime25.parquet}
AIME26_FILE=${AIME26_FILE:-${MATH_REASONING_DATA_DIR}/aime26.parquet}
VAL_FILES=${VAL_FILES:-}
CUSTOM_REWARD_PATH=${CUSTOM_REWARD_PATH:-${SCRIPT_DIR}/math_verify_reward.py}
CUSTOM_REWARD_NAME=${CUSTOM_REWARD_NAME:-compute_score}

TRAIN_BATCH_SIZE=128
PPO_MICRO_BATCH_SIZE_PER_GPU=1
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=1

NNODES=${NNODES:-1}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
ROLLOUT_TP=${ROLLOUT_TP:-1}
ROLLOUT_N=${ROLLOUT_N:-8}
VAL_N=${VAL_N:-16}
SEED=42
FULL_DETERMINISM=True
DATALOADER_NUM_WORKERS=0
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-16384}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.70}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-128}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-18432}
ACTOR_USE_TORCH_COMPILE=False
REF_USE_TORCH_COMPILE=False
ROLLOUT_ENFORCE_EAGER=True
KL_LOSS_COEF=${KL_LOSS_COEF:-0.0}
KL_REWARD_COEF=${KL_REWARD_COEF:-0.0}
NCCL_TIMEOUT=${NCCL_TIMEOUT:-3600}
SAVE_FREQ=${SAVE_FREQ:--1}
TEST_FREQ=${TEST_FREQ:-10}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-10}
RESUME_MODE=disable
WANDB_MODE=${WANDB_MODE:-offline}
LOGGER=${LOGGER:-'["console","tensorboard"]'}
TENSORBOARD_ROOT=${RELEASE_ROOT}/recipe/tensorboard
TENSORBOARD_DIR=${TENSORBOARD_ROOT}/${EXP_NAME}
DRY_RUN=${DRY_RUN:-0}
PREPARE_ONLY=${PREPARE_ONLY:-0}
START_RAY=${START_RAY:-1}
RESET_RAY=${RESET_RAY:-1}
RAY_INCLUDE_DASHBOARD=${RAY_INCLUDE_DASHBOARD:-false}
RAY_DASHBOARD_PORT=${RAY_DASHBOARD_PORT:-8265}
RAY_PORT=${RAY_PORT:-6379}
TOPD_ALPHA=${TOPD_ALPHA:-}

TOPD_ALPHA_ARGS=()
if [ -n "${TOPD_ALPHA}" ]; then
    TOPD_ALPHA_ARGS+=(+algorithm.topd_alpha="${TOPD_ALPHA}")
fi

export PYTHONPATH=${VERL_DIR}${PYTHONPATH:+:${PYTHONPATH}}
export WANDB_MODE
export TENSORBOARD_DIR
export VERL_ENABLE_FULL_DETERMINISM=1
export PYTHONHASHSEED="${SEED}"
export CUBLAS_WORKSPACE_CONFIG=":16:8"
export NCCL_DETERMINISTIC="1"
export FLASH_ATTENTION_DETERMINISTIC="1"
export CUDA_DEVICE_MAX_CONNECTIONS="1"
export RAY_DEDUP_LOGS=${RAY_DEDUP_LOGS:-0}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export HYDRA_FULL_ERROR=${HYDRA_FULL_ERROR:-1}

for v in $(env | awk -F= '/^(PMI|PMIX|MPI|OMPI|SLURM)_/{print $1}'); do
    unset "$v"
done

if [ ! -f "${CUSTOM_REWARD_PATH}" ]; then
    echo "CUSTOM_REWARD_PATH does not exist: ${CUSTOM_REWARD_PATH}" >&2
    exit 1
fi

build_val_files() {
    if [ -n "${VAL_FILES}" ]; then
        echo "${VAL_FILES}"
        return
    fi

    local files=()
    for f in "${AIME24_FILE}" "${AIME25_FILE}" "${AIME26_FILE}"; do
        if [ -f "${f}" ]; then
            files+=("${f}")
        elif [ "${DRY_RUN}" = "1" ]; then
            files+=("${f}")
        fi
    done

    if [ "${#files[@]}" -eq 0 ]; then
        echo "No validation parquet found. Expected at least AIME24_FILE=${AIME24_FILE}" >&2
        exit 1
    fi

    local joined="["
    local first=1
    for f in "${files[@]}"; do
        if [ "${first}" -eq 0 ]; then
            joined+=", "
        fi
        joined+="'${f}'"
        first=0
    done
    joined+="]"
    echo "${joined}"
}

VAL_FILES_LIST=$(build_val_files)
TRAIN_FILES_LIST="['${TRAIN_FILE}']"
TRAIN_LOG=${LOG_DIR}/train_$(date +%Y%m%d_%H%M%S).log

cat <<EOF
WORKSPACE=${WORKSPACE}
VERL_DIR=${VERL_DIR}
MODEL_PATH=${MODEL_PATH}
TEACHER_MODEL_PATH=${TEACHER_MODEL_PATH}
TRAIN_FILES=${TRAIN_FILES_LIST}
VAL_FILES=${VAL_FILES_LIST}
RUN_DIR=${RUN_DIR}
LOG_DIR=${LOG_DIR}
CKPT_DIR=${CKPT_DIR}
TENSORBOARD_DIR=${TENSORBOARD_DIR}
RESET_RAY=${RESET_RAY}
PREPARE_ONLY=${PREPARE_ONLY}
CUSTOM_REWARD_PATH=${CUSTOM_REWARD_PATH}
CUSTOM_REWARD_NAME=${CUSTOM_REWARD_NAME}
ROLLOUT_N=${ROLLOUT_N}
VAL_N=${VAL_N}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS}
SEED=${SEED}
FULL_DETERMINISM=${FULL_DETERMINISM}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS}
DATA_SEED=${SEED}
ACTOR_DATA_LOADER_SEED=${SEED}
ACTOR_FSDP_SEED=${SEED}
ROLLOUT_SEED=${SEED}
REF_FSDP_SEED=${SEED}
CRITIC_DATA_LOADER_SEED=${SEED}
CRITIC_FSDP_SEED=${SEED}
PYTHONHASHSEED=${PYTHONHASHSEED}
CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG}
NCCL_DETERMINISTIC=${NCCL_DETERMINISTIC}
FLASH_ATTENTION_DETERMINISTIC=${FLASH_ATTENTION_DETERMINISTIC}
CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU}
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}
ADV_ESTIMATOR=${ADV_ESTIMATOR}
TOPD_ALPHA=${TOPD_ALPHA}
TOPD_ALPHA_ARGS=${TOPD_ALPHA_ARGS[*]:-}
KL_LOSS_COEF=${KL_LOSS_COEF}
KL_REWARD_COEF=${KL_REWARD_COEF}
RESUME_MODE=${RESUME_MODE}
TRAIN_LOG=${TRAIN_LOG}
EOF

mkdir -p "${LOG_DIR}" "${TENSORBOARD_DIR}"

if [ "${DRY_RUN}" = "1" ]; then
    echo "DRY_RUN=1, created output directories and skip validation checks and training launch."
    exit 0
fi

if [ "${PREPARE_ONLY}" = "1" ]; then
    echo "PREPARE_ONLY=1, created output directories and skip training launch."
    exit 0
fi

if [ ! -d "${VERL_DIR}" ] || [ ! -f "${VERL_DIR}/verl/trainer/main_ppo.py" ]; then
    echo "VERL_DIR is invalid: ${VERL_DIR}" >&2
    exit 1
fi

if [ ! -f "${MODEL_PATH}/config.json" ]; then
    echo "MODEL_PATH is invalid or missing config.json: ${MODEL_PATH}" >&2
    exit 1
fi

if [ ! -f "${TEACHER_MODEL_PATH}/config.json" ]; then
    echo "TEACHER_MODEL_PATH is invalid or missing config.json: ${TEACHER_MODEL_PATH}" >&2
    exit 1
fi

if [ ! -f "${TRAIN_FILE}" ]; then
    echo "TRAIN_FILE does not exist: ${TRAIN_FILE}" >&2
    exit 1
fi

python3 - <<PY
import json
from pathlib import Path
for p in [Path('${TRAIN_FILE}')]:
    assert p.exists(), p
for name, p in [('student', Path('${MODEL_PATH}') / 'config.json'), ('teacher', Path('${TEACHER_MODEL_PATH}') / 'config.json')]:
    cfg = json.loads(p.read_text())
    print(f'{name}_model_type=', cfg.get('model_type'))
    print(f'{name}_architectures=', cfg.get('architectures'))
PY

if [ "${START_RAY}" = "1" ]; then
    if [ "${RESET_RAY}" = "1" ]; then
        ray stop --force >/dev/null 2>&1 || true
    fi

    if ! ray status >/dev/null 2>&1; then
        ray start --head \
            --port="${RAY_PORT}" \
            --include-dashboard="${RAY_INCLUDE_DASHBOARD}" \
            --dashboard-port="${RAY_DASHBOARD_PORT}" \
            --disable-usage-stats \
            --num-gpus="${GPUS_PER_NODE}" \
            > "${LOG_DIR}/ray_start.log" 2>&1 || {
                cat "${LOG_DIR}/ray_start.log" >&2
                exit 1
            }
    fi
fi

cd "${VERL_DIR}"

python3 -m verl.trainer.main_ppo \
    +ray_kwargs.ray_init.address=auto \
    +ray_kwargs.ray_init.runtime_env.env_vars.TENSORBOARD_DIR="'${TENSORBOARD_DIR}'" \
    +ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH="'${PYTHONPATH}'" \
    +ray_kwargs.ray_init.runtime_env.env_vars.WANDB_MODE="'${WANDB_MODE}'" \
    +ray_kwargs.ray_init.runtime_env.env_vars.VERL_ENABLE_FULL_DETERMINISM="'${VERL_ENABLE_FULL_DETERMINISM}'" \
    +ray_kwargs.ray_init.runtime_env.env_vars.PYTHONHASHSEED="'${PYTHONHASHSEED}'" \
    +ray_kwargs.ray_init.runtime_env.env_vars.CUBLAS_WORKSPACE_CONFIG="'${CUBLAS_WORKSPACE_CONFIG}'" \
    +ray_kwargs.ray_init.runtime_env.env_vars.NCCL_DETERMINISTIC="'${NCCL_DETERMINISTIC}'" \
    +ray_kwargs.ray_init.runtime_env.env_vars.FLASH_ATTENTION_DETERMINISTIC="'${FLASH_ATTENTION_DETERMINISTIC}'" \
    +ray_kwargs.ray_init.runtime_env.env_vars.CUDA_DEVICE_MAX_CONNECTIONS="'${CUDA_DEVICE_MAX_CONNECTIONS}'" \
    +ray_kwargs.ray_init.runtime_env.env_vars.TOKENIZERS_PARALLELISM="'${TOKENIZERS_PARALLELISM}'" \
    algorithm.adv_estimator="${ADV_ESTIMATOR}" \
    "${TOPD_ALPHA_ARGS[@]}" \
    algorithm.use_kl_in_reward=True \
    algorithm.kl_ctrl.kl_coef="${KL_REWARD_COEF}" \
    reward_model.reward_manager=naive \
    reward.custom_reward_function.path="${CUSTOM_REWARD_PATH}" \
    reward.custom_reward_function.name="${CUSTOM_REWARD_NAME}" \
    data.train_files="${TRAIN_FILES_LIST}" \
    data.val_files="${VAL_FILES_LIST}" \
    data.prompt_key=prompt \
    data.return_raw_chat=True \
    data.truncation=error \
    data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH}" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.shuffle=True \
    data.seed="${SEED}" \
    data.dataloader_num_workers="${DATALOADER_NUM_WORKERS}" \
    data.validation_shuffle=False \
    data.filter_overlong_prompts=True \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.nccl_timeout="${NCCL_TIMEOUT}" \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
    actor_rollout_ref.actor.data_loader_seed="${SEED}" \
    actor_rollout_ref.actor.fsdp_config.seed="${SEED}" \
    actor_rollout_ref.actor.fsdp_config.full_determinism="${FULL_DETERMINISM}" \
    actor_rollout_ref.actor.fsdp_config.use_torch_compile="${ACTOR_USE_TORCH_COMPILE}" \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef="${KL_LOSS_COEF}" \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.2 \
    actor_rollout_ref.actor.clip_ratio_c=3.0 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.use_torch_compile="${ACTOR_USE_TORCH_COMPILE}" \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
    actor_rollout_ref.actor.policy_loss.loss_mode=vanilla \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    +actor_rollout_ref.actor.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=[Qwen3DecoderLayer] \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.enforce_eager="${ROLLOUT_ENFORCE_EAGER}" \
    actor_rollout_ref.rollout.seed="${SEED}" \
    actor_rollout_ref.rollout.deterministic_request_seed="${FULL_DETERMINISM}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}" \
    actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEMORY_UTILIZATION}" \
    actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}" \
    actor_rollout_ref.rollout.max_num_seqs="${MAX_NUM_SEQS}" \
    actor_rollout_ref.rollout.max_num_batched_tokens="${MAX_NUM_BATCHED_TOKENS}" \
    actor_rollout_ref.rollout.max_model_len=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH)) \
    actor_rollout_ref.rollout.prompt_length="${MAX_PROMPT_LENGTH}" \
    actor_rollout_ref.rollout.response_length="${MAX_RESPONSE_LENGTH}" \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7 \
    actor_rollout_ref.rollout.val_kwargs.n="${VAL_N}" \
    +actor_rollout_ref.ref.model.path="${TEACHER_MODEL_PATH}" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}" \
    actor_rollout_ref.ref.use_torch_compile="${REF_USE_TORCH_COMPILE}" \
    actor_rollout_ref.ref.fsdp_config.seed="${SEED}" \
    actor_rollout_ref.ref.fsdp_config.full_determinism="${FULL_DETERMINISM}" \
    actor_rollout_ref.ref.fsdp_config.use_torch_compile="${REF_USE_TORCH_COMPILE}" \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    "+actor_rollout_ref.ref.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=[${TEACHER_LAYER_CLS}]" \
    critic.data_loader_seed="${SEED}" \
    critic.model.fsdp_config.seed="${SEED}" \
    critic.model.fsdp_config.full_determinism="${FULL_DETERMINISM}" \
    trainer.critic_warmup=0 \
    trainer.logger="${LOGGER}" \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXP_NAME}" \
    trainer.n_gpus_per_node="${GPUS_PER_NODE}" \
    trainer.nnodes="${NNODES}" \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.test_freq="${TEST_FREQ}" \
    trainer.default_local_dir="${CKPT_DIR}" \
    trainer.resume_mode="${RESUME_MODE}" \
    trainer.total_epochs="${TOTAL_EPOCHS}" \
    trainer.val_before_train=True \
    trainer.log_val_generations=10 \
    "$@" 2>&1 | tee "${TRAIN_LOG}"
