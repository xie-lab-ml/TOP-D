# TOP-D

Minimal OPD and Trust Region Policy Distillation, using the **reference model as the teacher**.

This repository includes the original working `verl` source and the original math datasets. Clone once, install the environment, download the model checkpoints, and run the recipe. No upstream checkout or patch-application step is required.

```text
TOP-D/
├── model/                # empty placeholder; model checkpoints are downloaded here
├── recipe/
│   ├── opd/              # original OPD, TOP-D and naive TOP-D launch scripts
│   ├── data/             # original DAPO and AIME Parquet files
│   ├── environment/      # versions read from the original working environment
│   └── verification/     # offline file-identity checks
├── verl/                 # full bundled verl Python package and installation metadata
├── install.sh
├── LICENSE
└── README.md
```

## Environment

Use **Linux, Python 3.12, CUDA 12.9, PyTorch 2.10.0 and vLLM 0.17.0**. A CUDA-compatible NVIDIA driver and adequate GPU/host memory are required. `recipe/environment/requirements.txt` pins the other directly used packages to the versions observed in the original environment, including Transformers 5.2.0, Ray 2.54.0, FlashAttention 2.8.3, NumPy 1.26.4, Pandas 3.0.1 and PyArrow 23.0.1.

```bash
git clone https://github.com/xie-lab-ml/TOP-D.git
cd TOP-D
python3.12 -m venv .venv
source .venv/bin/activate
bash install.sh
```

Run the installer in a dedicated environment, not in an environment used by another running job. FlashAttention may require the matching CUDA toolkit/compiler to build. The original public Docker build recipe is retained at `verl/docker/Dockerfile.stable.vllm` for users who prefer to reproduce the base container.

If the original dependency environment is already installed, use:

```bash
bash install.sh --existing-env
```

The installer installs **this local `verl/`**, not the latest PyPI/GitHub version. It intentionally does not use `pip install .[vllm]`, whose inherited upstream optional dependency has an older vLLM cap. Check the installed versions with `bash install.sh --check`.

## Models

```bash
bash recipe/opd/download_models.sh
```

Defaults:
- Student: `Qwen/Qwen3-1.7B-Base`
- Teacher: **`Qwen/Qwen3-30B-A3B-Instruct-2507`**

The empty `model/` directory is retained with `.gitkeep`; no weights are committed. Already downloaded checkpoints can be selected with `MODEL_PATH` and `TEACHER_MODEL_PATH`. The teacher uses the existing reference-policy log-probability path; no additional teacher API is required.

## Original data included

| File in `recipe/data/math_reasoning_parquet/` | Rows | Use |
|---|---:|---|
| `dapo_math_17k.parquet` | 17,398 | Training |
| `aime24.parquet` | 30 | Evaluation |
| `aime25.parquet` | 30 | Evaluation |
| `aime26.parquet` | 30 | Evaluation |

All four files have **the same bytes and SHA-256 as the author's original files**: prompts, answers, row order, `extra_info`, Arrow schema and Parquet encoding are preserved. These are not substitute datasets or newly chosen preprocessing defaults. Their checksums and public-source attribution are under `recipe/data/`.

**Data licenses are separate from the code license. AIME26 retains CC BY-NC-SA 4.0 (non-commercial, attribution, share-alike).** See `recipe/data/NOTICE.txt` before redistribution or commercial use. The other source cards identify Apache-2.0; this does not override rights in the original examination materials.

## Run on one eight-GPU node

The recipe uses **128 prompts per rollout batch**, eight responses per prompt and **avg@16** evaluation. OPD uses a 128-prompt PPO minibatch; TOP-D and naive TOP-D use 16 prompts. TOP-D alpha is 0.1. The prompt/response limits are 2,048/16,384 tokens, with learning rate `1e-6` and the original policy-loss settings.

Use a dedicated available training node. **The original script contains a Ray reset option; `RESET_RAY=0` below prevents it from stopping existing Ray processes.** Do not attach to an unrelated cluster.

```bash
# OPD
RESET_RAY=0 bash recipe/opd/run_qwen3_1p7b_base_qwen3_30a3b_teacher_dapo17k_opd_16k_kl0.sh

# TOP-D
RESET_RAY=0 bash recipe/opd/run_qwen3_1p7b_base_qwen3_30a3b_teacher_dapo17k_topd_alpha0p1_16k_kl0.sh

# Naive TOP-D ablation
RESET_RAY=0 bash recipe/opd/run_qwen3_1p7b_base_qwen3_30a3b_teacher_dapo17k_naive_topd_alpha0p1_16k_kl0.sh
```

`avg@16` averages correctness over 16 completions per question and then over questions. It is not best-of-16. The original metric computation is retained; the `val-core` display alias now mirrors `avg@16`.

Only the explicitly requested release defaults differ from the working source: the default teacher and experiment label select 30B-A3B instead of the local 8B debug teacher, its reference wrapping uses `Qwen3MoeDecoderLayer` (overridable via `TEACHER_LAYER_CLS`), and the `val-core` alias uses 16 instead of 32. The original optimization, sampling, determinism, reward and trainer implementations are otherwise retained. Eight GPUs of arbitrary memory capacity are not automatically sufficient for the 30B teacher and 16K responses.

## Verify fidelity without training

```bash
python recipe/verification/verify_release.py --decode-data

# Optional, if the original project is available on the same filesystem:
python recipe/verification/verify_release.py --original /path/to/original/project --decode-data
```

The original `verl/verl` runtime contains 482 tracked files. All were matched before the declared display-only change: 462 were obtained as identical public Git blobs, and the other 20 were copied from the original source and independently checked. Original installation metadata and all six recipe files were also checked. The full code is vendored, so an inferred upstream version is not needed at install time. Upstream documentation, examples and CI directories not needed to install or run the package are omitted.

The author reports that the original project runs on their GPUs. Packaging/file-identity checks do not constitute a new GPU experiment: this release has not independently rerun the complete training, measured memory on other cards, or reproduced paper metrics in a clean new environment. Dependency installation instructions preserve the observed versions; GPU driver/toolkit and transitive-dependency differences may still require environment-specific fixes.

## License

Code: Apache-2.0, with upstream notices retained in `verl/LICENSE` and `verl/Notice.txt`. Dataset terms are described separately in `recipe/data/NOTICE.txt`. Model checkpoints retain their own terms. See [Trust Region Policy Distillation](https://arxiv.org/abs/2607.04751) for the method.
