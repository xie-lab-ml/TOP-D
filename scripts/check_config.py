#!/usr/bin/env python3
"""Compose actual launcher arguments against the pinned Hydra config, without Ray."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verl-dir", type=Path, default=ROOT / "third_party/verl")
    args = parser.parse_args()
    bash = shutil.which("bash")
    if not bash:
        raise RuntimeError("Bash is required to inspect the actual shell launchers.")
    for entry in sorted((ROOT / "recipe/opd").glob("run_*_kl0.sh")):
        env = os.environ.copy()
        env.update({"DRY_RUN": "1", "PYTHON_ONLY_ARGS": "1", "PYTHON_BIN": Path(sys.executable).as_posix(),
                    "VERL_DIR": args.verl_dir.resolve().as_posix()})
        result = subprocess.run([bash, entry.as_posix()], env=env, text=True, encoding="utf-8", capture_output=True, check=True)
        overrides = json.loads(result.stdout)
        with initialize_config_dir(config_dir=str((args.verl_dir / "verl/trainer/config").resolve()), version_base=None):
            cfg = compose(config_name="ppo_trainer", overrides=overrides)
            OmegaConf.resolve(cfg)
            assert cfg.data.train_batch_size == 128
            assert cfg.trainer.n_gpus_per_node == 8
            assert cfg.actor_rollout_ref.rollout.val_kwargs.n == 16
            assert cfg.actor_rollout_ref.ref.model.path.endswith("Qwen3-30B-A3B-Instruct-2507")
            assert cfg.trainer.use_legacy_worker_impl == "enable"
            assert cfg.actor_rollout_ref.ref.fsdp_config.model_dtype == "bfloat16"
            assert cfg.actor_rollout_ref.rollout.seed == 42
            assert cfg.ray_kwargs.ray_init.address == "local"
            assert not cfg.actor_rollout_ref.actor.use_kl_loss and not cfg.algorithm.use_kl_in_reward
            print(entry.name, "Hydra composition OK", "mode=" + cfg.algorithm.adv_estimator)
    print("Configuration composition is not model, worker or GPU execution.")


if __name__ == "__main__":
    main()
