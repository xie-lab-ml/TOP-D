#!/usr/bin/env python3
"""Read-only checks before allocating a training run; this is not a training test."""
from __future__ import annotations

import argparse
import ast
import importlib.metadata
import json
import platform
from pathlib import Path


def check_weights(path: Path) -> list[str]:
    """Check local shard completeness without reading tensor payloads."""
    path = path.resolve()
    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index = path / index_name
        if index.is_file():
            mapping = json.loads(index.read_text(encoding="utf-8")).get("weight_map", {})
            if not mapping or not all(isinstance(v, str) for v in mapping.values()):
                raise ValueError(f"Empty or invalid weight index: {index}")
            names = sorted(set(mapping.values()))
            for name in names:
                shard = (path / name).resolve()
                if not shard.is_relative_to(path) or not shard.is_file() or shard.stat().st_size == 0:
                    raise ValueError(f"Missing or invalid model shard: {name}")
            return names
    for name in ("model.safetensors", "pytorch_model.bin"):
        weight = path / name
        if weight.is_file() and weight.stat().st_size > 0:
            return [name]
    raise ValueError(f"Missing weights or shard index in {path}; config.json alone is not a model.")


def check_parquet(path: Path, *, validation: bool) -> int:
    import pyarrow.parquet as pq

    if not path.is_file():
        raise ValueError(f"Missing parquet: {path}")
    table = pq.read_table(path)
    required = {"prompt", "reward_model", "data_source"}
    if not required.issubset(table.column_names):
        raise ValueError(f"{path.name} needs {sorted(required)}")
    rows = table.select(sorted(required)).to_pylist()
    if not rows:
        raise ValueError(f"Empty dataset: {path.name}")
    seen = set()
    for row in rows:
        prompt = row["prompt"]
        if not isinstance(prompt, list) or not prompt or not all(
            isinstance(m, dict) and isinstance(m.get("content"), str) and m.get("role") for m in prompt
        ):
            raise ValueError(f"Invalid chat prompt in {path.name}")
        if row["reward_model"].get("ground_truth") is None:
            raise ValueError(f"Missing answer in {path.name}")
        key = json.dumps(prompt, sort_keys=True, ensure_ascii=False)
        if validation and key in seen:
            raise ValueError("Validation prompts must not be pre-repeated; val_kwargs.n supplies 16 samples.")
        seen.add(key)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student", required=True, type=Path)
    parser.add_argument("--teacher", required=True, type=Path)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--validation-list", required=True)
    parser.add_argument("--gpus", required=True, type=int)
    parser.add_argument("--verl-dir", required=True, type=Path)
    args = parser.parse_args()
    if platform.system() != "Linux":
        raise RuntimeError("GPU training and the original math-verifier timeout require Linux.")
    if not 0 < args.gpus:
        raise ValueError("GPU count must be positive.")
    from transformers import AutoConfig, AutoTokenizer
    import torch
    import verl
    from verl.trainer.ppo.core_algos import AdvantageEstimator

    if not Path(verl.__file__).resolve().is_relative_to(args.verl_dir.resolve()):
        raise RuntimeError("Python imported another verl installation, not the pinned checkout.")
    if not all(hasattr(AdvantageEstimator, name) for name in ("OPD", "TOPD", "NAIVE_TOPD")):
        raise RuntimeError("The imported verl package is missing the release estimators.")

    versions = {name: importlib.metadata.version(name) for name in (
        "torch", "vllm", "transformers", "ray", "math-verify", "hydra-core", "tensordict", "torchdata"
    )}
    print("Installed training packages:", json.dumps(versions, sort_keys=True))
    if torch.cuda.device_count() < args.gpus:
        raise RuntimeError("Fewer visible GPUs than requested; no training was launched.")
    if args.student.resolve() == args.teacher.resolve():
        raise ValueError("Teacher and student paths must be distinct.")
    tokenizers = []
    for path in (args.student, args.teacher):
        if not (path / "config.json").is_file():
            raise ValueError(f"Missing local model configuration: {path}")
        shards = check_weights(path)
        cfg = AutoConfig.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        tok = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        if path == args.student:
            if not tok.chat_template:
                raise ValueError(f"Missing student chat template: {path}. Restore the intended tokenizer configuration; no implicit template is injected.")
            rendered = tok.apply_chat_template([{"role": "user", "content": "What is 1 + 1?"}], tokenize=False, add_generation_prompt=True)
            if not isinstance(rendered, str) or not rendered:
                raise ValueError("Student chat template did not render a generation prompt.")
        print(path.name, "weight_shards=", len(shards))
        print(path.name, "model_type=", cfg.model_type, "vocab_size=", cfg.vocab_size)
        tokenizers.append(tok)
    student, teacher = tokenizers
    teacher_vocab = teacher.get_vocab()
    mismatches = [(token, idx) for token, idx in student.get_vocab().items() if teacher_vocab.get(token) != idx]
    if mismatches:
        raise ValueError("Teacher/student token-ID mapping differs. This release does not implement tokenizer remapping.")
    print("Teacher/student token-ID mapping verified.")
    print("Training prompts:", check_parquet(args.train, validation=False))
    validation_paths = ast.literal_eval(args.validation_list)
    if not isinstance(validation_paths, list) or not validation_paths:
        raise ValueError("Provide at least one validation dataset.")
    for path in validation_paths:
        print("Validation prompts:", Path(path).name, check_parquet(Path(path), validation=True))
    print("Preflight passed; model loading, GPU memory use and training are still unverified.")


if __name__ == "__main__":
    main()
