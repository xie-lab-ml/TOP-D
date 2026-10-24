# TOP-D: minimal OPD and Trust Region Policy Distillation

A compact, reference-model-based implementation of [Trust Region Policy Distillation](https://arxiv.org/abs/2607.04751), built on a pinned public [verl](https://github.com/verl-project/verl) checkout. The teacher scores the student's sampled token IDs through verl's **reference policy**; no separate teacher API or distillation server is needed.

This is an initial release for review. CPU algorithm tests and configuration composition do not establish GPU training or paper-result reproduction. The eight-GPU preset below still needs an end-to-end run on the target hardware.

## What is preserved

The seven OPD/TOP-D algorithm functions match the author's reviewed source at the normalized Python AST level; their fingerprints are in `tests/original_algorithms.json`.

- **OPD:** `(teacher_log_prob - old_student_log_prob) * response_mask`.
- **TOP-D:** stable `logaddexp` reward transformation, immediate reward plus mean future-token reward, then prompt-group token normalization with population standard deviation and epsilon `1e-6`.
- **Naive TOP-D:** only the immediate-reward transformation, without returns or group normalization.
- Advantages are detached. The student is optimized with the existing vanilla PPO loss and `seq-mean-token-mean` aggregation.
- At `alpha=1`, the immediate reward reduces to OPD. Full TOP-D still computes returns and normalizes them; it is not identical to OPD.
- Scope: single-turn text, right-padded response masks, full-parameter FSDP, aligned teacher/student token IDs. No LoRA teacher shortcut or tokenizer remapping.

## Fixed baseline

`verl` is pinned to **`a4f53df1a3511648330de8c3db2fc29463f2b730`**, version `0.8.0.dev` (2026-03-21). Eleven framework/configuration file fingerprints match the reviewed release; see `upstream.lock.json`.

The original nested Git history was absent. This SHA is a selected public compatibility baseline, **not a recovered original experiment commit**. Do not apply this patch to current `main`: the worker and distillation interfaces have changed.

`patches/verl-topd.patch` modifies only five upstream files: the advantage algorithms, trainer dispatch/validation, reference-policy activation, and two rollout-seed configuration files. The pinned FSDP worker already supports `actor_rollout_ref.ref.model.path`, so its teacher loader is not copied or replaced. This baseline always computes old student log-probabilities in the normal path, so the later optimization that required `force_logprob_compute` is not imported.

## Eight-GPU preset

| Setting | Default |
|---|---|
| Student | `Qwen/Qwen3-1.7B-Base` |
| Teacher | **`Qwen/Qwen3-30B-A3B-Instruct-2507`** |
| Hardware layout | 1 node, 8 GPUs |
| Training batch | **128 prompts**, not 128 per GPU |
| Rollouts per prompt | 8 (1,024 responses per rollout batch) |
| PPO minibatch | OPD: 128 prompts; TOP-D / naive TOP-D: 16 prompts |
| PPO epochs | 1 |
| TOP-D alpha | 0.1 |
| Prompt / response limit | 2,048 / 16,384 tokens |
| Learning rate / clipping | `1e-6` / 0.2 |
| Validation | **avg@16**, temperature 1.0, top-p 0.7 |
| Additional KL penalties | Disabled, coefficient zero |
| Reference weights | BF16 with parameter offload |

`avg@16` means mean correctness over 16 sampled completions per question, then mean over questions. It is **not pass@16 or best-of-16**. Values are fractions in `[0,1]`. Logged keys are `val/acc/<dataset>/avg@16` and `val-core/<dataset>/acc/avg@16`. An incomplete sample group fails rather than receiving a misleading metric name. Validation parquet stores each question once.

Eight GPUs alone are not a memory guarantee. GPU model, available VRAM, host RAM, and sequence length matter. The 30B-A3B teacher loads all of its weights, not only its active parameters. Memory limits and distributed performance are unverified.

## Setup (Linux training machine)

1. Start with a CUDA/PyTorch/vLLM environment compatible with the pinned source. `requirements-observed.txt` records the versions observed in the author's environment: Python 3.12, CUDA 12.9, PyTorch 2.10.0, vLLM 0.17.0, Transformers 5.2.0, FlashAttention 2.8.3, Ray 2.54.0. It is not a complete environment lock. The pinned upstream Docker recipe is at `docker/Dockerfile.stable.vllm` in the linked verl commit. No internal container image is needed or redistributed.
2. Install the pinned source and patch:

```bash
bash scripts/setup_verl.sh
```

This downloads verl to `third_party/verl`, verifies the SHA, applies and verifies the patch, and installs it with `pip --no-deps -e`. It assumes the training dependencies are already installed; it does not replace CUDA or install all GPU libraries. Use `SKIP_INSTALL=1` for source-only setup. Existing unrelated checkouts or changes are never reset.

3. Download public model checkpoints:

```bash
bash recipe/opd/download_models.sh
```

The default teacher is 30B-A3B-Instruct-2507. Downloads are large and are performed only when this command is run. Exact resolved Hub revisions are saved in `model/models.lock.json` and reused on later runs. Set `STUDENT_REVISION` / `TEACHER_REVISION` to a commit SHA for an explicit choice. Weights are not part of this repository.

The preflight checks every shard named by the checkpoint index, student chat-template rendering, and the shared token-to-ID mapping. It never injects a different prompt template silently. The reviewed source student's tokenizer includes a Qwen chat template; if a downloaded tokenizer lacks the intended template, resolve that deliberately before training.

4. Prepare public datasets:

```bash
python scripts/prepare_data.py
```

This prepares DAPO-Math-17k and AIME24/AIME25 in verl parquet format and records data revisions and output checksums. Public defaults are `BytedTsinghua-SIA/DAPO-Math-17k`, `Maxwell-Jia/AIME_2024`, and `yentinglin/aime_2025`. Prepared local files can be supplied with `--dapo-file`, `--aime24-file`, and `--aime25-file`. Optional AIME26 is accepted through `--aime26-file`; no unverified source is guessed. Dataset and model licenses remain their own.

These public preprocessing defaults are not claimed to reconstruct unidentified original experiment parquet byte-for-byte. Prepared chat prompts are retained; raw questions receive the explicit boxed-answer suffix in `scripts/prepare_data.py`.

## Run

```bash
# Configuration preview only: no directories, models, Ray, or training.
DRY_RUN=1 bash recipe/opd/run_qwen3_1p7b_base_qwen3_30a3b_teacher_dapo17k_topd_alpha0p1_16k_kl0.sh

# OPD
bash recipe/opd/run_qwen3_1p7b_base_qwen3_30a3b_teacher_dapo17k_opd_16k_kl0.sh

# TOP-D
bash recipe/opd/run_qwen3_1p7b_base_qwen3_30a3b_teacher_dapo17k_topd_alpha0p1_16k_kl0.sh

# Immediate-reward-only ablation
bash recipe/opd/run_qwen3_1p7b_base_qwen3_30a3b_teacher_dapo17k_naive_topd_alpha0p1_16k_kl0.sh
```

Use environment variables such as `MODEL_PATH`, `TEACHER_MODEL_PATH`, `TRAIN_FILE`, `VAL_FILES`, `GPUS_PER_NODE`, `TRAIN_BATCH_SIZE`, `PPO_MINI_BATCH_SIZE`, `VAL_N`, and `RUN_DIR` to override the preset. `VAL_FILES` is a quoted list of prepared parquet paths. Avoid apostrophes in file paths. Additional Hydra arguments are forwarded at the end; if overriding model/data paths, use the environment variables so preflight checks the same files.

The launcher starts a **local** Ray instance through verl; it never runs `ray stop --force`, deletes processes, or silently attaches to an existing cluster. A nonlocal `RAY_ADDRESS` is rejected. It requires the release patch to already be installed. Existing Ray jobs are not stopped.

For a GPU smoke run, use shorter sequences and a distinct output directory, and add `trainer.total_training_steps=2 trainer.val_before_train=False`. Reducing batch size also requires a compatible `PPO_MINI_BATCH_SIZE`. Such a smoke run does not reproduce the long-response preset or paper accuracy. Do not run on occupied training GPUs without checking availability.

## Checks without GPU training

```bash
python -m pytest tests/test_cpu.py -q
python scripts/check_config.py
python scripts/apply_patch.py --require-applied
```

Tests require PyTorch, NumPy and pytest; the configuration checker additionally requires Hydra and Bash. `VERL_DIR` can point tests at another checkout of the pinned and patched source. AST isolation executes the actual patched tensor functions without importing the distributed stack. It verifies algorithm numerics, edge cases, gradient isolation, registry/dispatch, avg@16, original function fingerprints, patch integrity and setup checks.

Not covered: Linux math-verifier process timeouts in real workers, full dependency integration, checkpoint tensor contents, GPU kernels, FSDP/vLLM execution, VRAM behavior, throughput, bitwise determinism, or paper-level metrics. The original source contained extra determinism and logging changes that are not wholesale copied into this minimal port. This release does not claim bitwise equivalence to the author's environment.

## License and provenance

Apache-2.0; retain `LICENSE`, `NOTICE`, and upstream notices. The original algorithm functions were reviewed against the author's minimal source; release guards, the avg@N output layer and portable setup scripts are explicit packaging changes. No model weights, datasets, credentials, internal service addresses or experiment logs are included.
