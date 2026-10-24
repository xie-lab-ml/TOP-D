<h1 align="center">
  Trust Region Policy Distillation
</h1>

<p align="center">
  <a href="https://scholar.google.com/citations?user=QuK9uDcAAAAJ&amp;hl=en">Zhengpeng Xie</a><sup>*</sup>,
  <a href="https://scholar.google.com/citations?user=-_ItfAoAAAAJ&amp;hl=en">Li Lyna Zhang</a><sup>†</sup>,
  <a href="https://scholar.google.com/citations?user=ysXmZCMAAAAJ&amp;hl=en">Zeke Xie</a><sup>‡</sup>,
  <a href="https://scholar.google.com/citations?user=LgJqohwAAAAJ&amp;hl=en">Mao Yang</a>
</p>

<p align="center">
  <sub>* Work done during an internship. &nbsp; † Project leader. &nbsp; ‡ Corresponding author.</sub>
</p>

<p align="center">
  <b>Accepted to NeurIPS 2026</b>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2607.04751">Paper</a> &nbsp;·&nbsp;
  <a href="#minimal-implementation">Implementation</a> &nbsp;·&nbsp;
  <a href="#quick-start">Quick start</a> &nbsp;·&nbsp;
  <a href="#citing-top-d">Citation</a>
</p>

<!-- Figure 1, original vector asset: https://arxiv.org/html/2607.04751v1/teaser.svg -->
<p align="center">
  <img src="recipe/assets/teaser.svg" width="550" alt="Figure 1: While standard OPD forces direct teacher supervision and suffers from unstable optimization, TOP-D ensures stable training by dynamically constructing a proximal teacher at every iteration.">
</p>

<p align="center">
  <em>Figure 1: While standard OPD forces direct teacher supervision and suffers from unstable optimization, TOP-D ensures stable training by dynamically constructing a proximal teacher at every iteration.</em>
</p>

<!-- Figure 2, original vector asset: https://arxiv.org/html/2607.04751v1/aime_performance.svg -->
<p align="center">
  <img src="recipe/assets/aime_performance.svg" width="100%" alt="Figure 2: The avg@32 accuracy on several AIME benchmarks for RLVR applied to the Qwen3-8B-Base model, and for OPD and TOP-D using Qwen3-30B-A3B-Instruct-2507 as the teacher model.">
</p>

<p align="center">
  <em>Figure 2: The avg@32 accuracy on several AIME benchmarks for RLVR applied to the Qwen3-8B-Base model, and for OPD and TOP-D using Qwen3-30B-A3B-Instruct-2507 as the teacher model.</em>
</p>

A **minimal implementation of TOP-D and On-Policy Distillation (OPD)** built on [verl](https://github.com/verl-project/verl). The repository includes the working runtime, training recipes, and original math datasets—no separate upstream checkout or patch step.

## Minimal implementation

**Keep verl's rollout and actor update; add distillation in three places.** The excerpts below show the core additions, with surrounding checks and branches omitted.

1. **Teacher → the existing reference model.** Pass the teacher checkpoint through the reference-model input. The frozen teacher scores **student-generated tokens**—no teacher rollout or separate service.

   ```bash
   # Argument passed to verl.trainer.main_ppo by the recipe.
   +actor_rollout_ref.ref.model.path="${TEACHER_MODEL_PATH}"
   ```

   The [reference worker](verl/verl/workers/fsdp_workers.py#L931-L958) loads this separate checkpoint; the [reference gate](verl/verl/trainer/ppo/utils.py#L72-L88) keeps scoring enabled for distillation, even with zero extra KL coefficients.

2. **[`core_algos.py`](verl/verl/trainer/ppo/core_algos.py#L338-L472) → add OPD / TOP-D advantages.** OPD uses the teacher–student log-probability difference. TOP-D constructs its proximal teacher **algebraically**, using the same scores:

   ```python
   # OPD
   advantages = (ref_log_prob - old_log_probs) * response_mask

   # TOP-D: log_ratio is ref_log_prob - old_log_probs on valid tokens.
   topd_rewards = torch.logaddexp(
       log_ratio + math.log(alpha),
       torch.full_like(log_ratio, math.log1p(-alpha)),
   )
   token_returns = _compute_topd_token_returns(topd_rewards, response_mask)
   advantages = _normalize_topd_token_returns_by_prompt(
       token_returns, response_mask, index=index
   )
   ```

   Each TOP-D return is **current reward + mean future reward**, then normalized over valid tokens sharing a prompt UID. The full functions handle padding, `alpha` boundaries, and `no_grad`. **Naive TOP-D** keeps only the immediate-reward transform.

3. **[`ray_trainer.py`](verl/verl/trainer/ppo/ray_trainer.py#L261-L296) → wire the new estimators into the trainer.** Keep pre-update student and reference-teacher log probabilities on the same tokens, then pass the advantages to the existing clipped PPO update:

   ```python
   # TOP-D branch; OPD selects compute_opd_advantage instead.
   compute_topd_fn = core_algos.compute_topd_advantage
   advantages, returns = compute_topd_fn(
       token_level_rewards=data.batch["token_level_rewards"],
       response_mask=data.batch["response_mask"],
       old_log_probs=data.batch["old_log_probs"],
       ref_log_prob=data.batch["ref_log_prob"],
       index=data.non_tensor_batch["uid"] if "uid" in data.non_tensor_batch else None,
       config=config,
   )
   data.batch["advantages"] = advantages
   data.batch["returns"] = returns
   ```

## Quick start

### 1. Install

Use **Linux, Python 3.12, CUDA 12.9, PyTorch 2.10.0, and vLLM 0.17.0** with a compatible NVIDIA driver.

```bash
git clone https://github.com/xie-lab-ml/TOP-D.git
cd TOP-D
python3.12 -m venv .venv
source .venv/bin/activate
bash install.sh
```

The installer uses the bundled `verl/` and the [versions observed in the working environment](recipe/environment/requirements.txt). Use an isolated environment; FlashAttention may need a matching CUDA toolkit/compiler. If those dependencies are already installed, run `bash install.sh --existing-env`. Use `bash install.sh --check` to inspect versions.

<details>
<summary>Installation details</summary>

The installer intentionally avoids `pip install .[vllm]`: the inherited optional dependency has an older vLLM cap. Additional pinned packages include Transformers 5.2.0, Ray 2.54.0, FlashAttention 2.8.3, NumPy 1.26.4, Pandas 3.0.1, and PyArrow 23.0.1. The original container build recipe is retained at [`verl/docker/Dockerfile.stable.vllm`](verl/docker/Dockerfile.stable.vllm).

</details>

### 2. Put the models in `model/`

Run from the repository root. Skip downloading if you already have the complete checkpoints.

```bash
# Student: download the complete checkpoint into model/.
hf download Qwen/Qwen3-1.7B-Base \
  --local-dir ./model/Qwen3-1.7B-Base

# Teacher: loaded through the existing reference-model path.
hf download Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --local-dir ./model/Qwen3-30B-A3B-Instruct-2507
```

These are the paths used by the recipes by default. Keep all weight shards, configuration, and tokenizer files; no weights are committed to this repository. The original helper, `bash recipe/opd/download_models.sh`, downloads to the same default locations.

<details>
<summary>Use checkpoints stored elsewhere</summary>

Point the training recipe to your existing checkpoint directories:

```bash
export MODEL_PATH=/path/to/Qwen3-1.7B-Base
export TEACHER_MODEL_PATH=/path/to/Qwen3-30B-A3B-Instruct-2507
```

The download helper uses `STUDENT_DIR` / `TEACHER_DIR` for its output paths; training uses `MODEL_PATH` / `TEACHER_MODEL_PATH`. If you customize the download locations, set the training variables accordingly. The default teacher is MoE and uses `Qwen3MoeDecoderLayer`; `TEACHER_LAYER_CLS` can be overridden for another architecture.

</details>

### 3. Run a recipe

The default setup is **one eight-GPU node, 128 prompts per rollout batch, eight responses per prompt, and avg@16 evaluation**. Use an idle, dedicated node with sufficient GPU and host memory for the 30B-A3B teacher and 16K responses.

```bash
# TOP-D
RESET_RAY=0 bash recipe/opd/run_qwen3_1p7b_base_qwen3_30a3b_teacher_dapo17k_topd_alpha0p1_16k_kl0.sh

# OPD baseline
RESET_RAY=0 bash recipe/opd/run_qwen3_1p7b_base_qwen3_30a3b_teacher_dapo17k_opd_16k_kl0.sh

# Naive TOP-D ablation
RESET_RAY=0 bash recipe/opd/run_qwen3_1p7b_base_qwen3_30a3b_teacher_dapo17k_naive_topd_alpha0p1_16k_kl0.sh
```

`RESET_RAY=0` prevents the original script from stopping existing Ray processes. Do not attach to an unrelated cluster.

| Setting | Default |
|---|---|
| Student / teacher | Qwen3-1.7B-Base / Qwen3-30B-A3B-Instruct-2507 |
| Rollout batch | 128 prompts × 8 responses |
| PPO minibatch | OPD: 128 prompts; TOP-D / naive TOP-D: 32 prompts |
| TOP-D `alpha` | 0.1 |
| Prompt / response limit | 2,048 / 16,384 tokens |
| Learning rate | `1e-6` |
| Evaluation | avg@16 on AIME24, AIME25, and AIME26 |

> **Single-node defaults. To keep experiments practical and efficient on 1 node / 8 GPUs, this release uses a smaller global prompt batch (128 instead of the paper's 512) and lighter validation (avg@16 instead of avg@32). To reproduce the paper's results or explore the performance ceiling, we recommend the paper's 4-node / 32-GPU distributed setup with its original batch size, evaluation protocol, and other hyperparameters.**

Each entry script sets `TRAIN_BATCH_SIZE=128` and its own `PPO_MINI_BATCH_SIZE`; the shared launcher consumes these values without overriding them.

**avg@16** averages correctness over 16 completions per question, then over questions; it is not pass@16 or best-of-16. The original metric computation is retained.

## Data and layout

The original Parquet files are included under [`recipe/data/math_reasoning_parquet/`](recipe/data/math_reasoning_parquet/): **DAPO-Math-17k (17,398 training examples)** and **AIME24 / AIME25 / AIME26 (30 evaluation problems each)**. Their bytes, row order, prompts, answers, and metadata match the original experiment files. See the [checksums](recipe/data/data_manifest.json) and [dataset sources and licenses](recipe/data/NOTICE.txt).

```text
TOP-D/
├── model/                 # checkpoint destinations; initially only .gitkeep
├── recipe/
│   ├── opd/               # launch scripts and reward function
│   ├── data/              # original math datasets
│   ├── environment/       # dependency pins and environment checks
│   ├── verification/      # file-identity and CPU checks
│   └── assets/            # original paper figures
├── verl/                  # complete bundled runtime and install metadata
├── install.sh
└── README.md
```

<details>
<summary>Source fidelity and validation scope</summary>

The original 482-file runtime, installation metadata, and six recipe files were checked against the working project. Declared release changes select the 30B-A3B teacher and its MoE wrapping, mirror avg@16 in `val-core`, and clarify documentation/comments. Batch configuration now lives in each entry script: train batch stays 128; TOP-D / naive TOP-D minibatches change from 16 to 32, while OPD stays 128. Algorithms, sampling, and determinism are unchanged. Source hashes and declared differences are recorded in [`source_manifest.json`](recipe/verification/source_manifest.json).

```bash
# Verify source files and the original Parquet bytes / decoded row content.
python recipe/verification/verify_release.py --decode-data

# CPU-only checks of OPD / TOP-D and the avg@16 display alias.
python recipe/verification/check_core.py

# Optional: compare against an original project on the same filesystem.
python recipe/verification/verify_release.py --original /path/to/original/project --decode-data
```

File-identity and CPU checks have passed. A fresh Linux installation, GPU model loading, distributed training, memory requirements on other cards, and paper-level scores have **not been independently revalidated for this packaged release**. Matching files or package versions is not a substitute for an end-to-end GPU run. Eight GPUs of arbitrary memory capacity are not automatically sufficient.

</details>

## Citing TOP-D

If you find this work useful, please cite:

```bibtex
@article{xie2026trust,
title={Trust Region Policy Distillation},
author={Xie, Zhengpeng and Zhang, Li Lyna and Xie, Zeke and Yang, Mao},
journal={arXiv preprint arXiv:2607.04751},
year={2026}
}
```

## Acknowledgments and license

Built on [verl](https://github.com/verl-project/verl). We thank its contributors for the training infrastructure.

Code is licensed under [Apache-2.0](LICENSE), with upstream notices preserved in [`verl/Notice.txt`](verl/Notice.txt). **Datasets and model checkpoints retain their own terms. In particular, AIME26 is CC BY-NC-SA 4.0** (attribution, non-commercial, share-alike); see [`recipe/data/NOTICE.txt`](recipe/data/NOTICE.txt). The other dataset source cards identify Apache-2.0, without overriding rights in the original examination materials.
