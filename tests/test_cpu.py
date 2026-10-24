"""CPU-only tests of the exact patched functions, not a parallel reimplementation.

AST isolation avoids importing Ray/FSDP/vLLM just to test pure tensor routines.
These tests do not validate GPU kernels, model loading, distributed execution,
math-verify's Linux worker timeout, or paper-level accuracy.
"""
from __future__ import annotations

import ast
import copy
import importlib.util
import json
import math
import os
from collections import defaultdict
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
VERL = Path(os.environ.get("VERL_DIR", ROOT / "third_party/verl"))


def extract(path, names, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    selected = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            node = copy.deepcopy(node)
            if isinstance(node, ast.FunctionDef):
                node.decorator_list = []
            selected.append(node)
    assert {node.name for node in selected} == set(names)
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *selected], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace


@pytest.fixture(scope="module")
def alg():
    if not (VERL / "verl/trainer/ppo/core_algos.py").is_file():
        pytest.fail("Set VERL_DIR to the pinned, patched checkout; do not skip missing source.")
    ns = {"torch": torch, "np": np, "math": math, "defaultdict": defaultdict, "Enum": Enum,
          "ADV_ESTIMATOR_REGISTRY": {}, "POLICY_LOSS_REGISTRY": {}, "Any": Any, "Optional": Optional,
          "AlgoConfig": type("AlgoConfig", (), {}), "ActorConfig": object}
    functional = extract(VERL / "verl/utils/torch_functional.py", ["masked_sum", "masked_mean"], {"torch": torch})
    ns["verl_F"] = SimpleNamespace(**{k: functional[k] for k in ("masked_sum", "masked_mean")})
    names = ["AdvantageEstimator", "register_adv_est", "get_adv_estimator_fn", "compute_opd_advantage",
             "_get_topd_alpha", "_compute_topd_immediate_rewards", "_compute_topd_token_returns",
             "_normalize_topd_token_returns_by_prompt", "compute_topd_advantage", "compute_naive_topd_advantage",
             "agg_loss", "compute_policy_loss_vanilla"]
    extract(VERL / "verl/trainer/ppo/core_algos.py", names, ns)
    for name in ("opd", "topd", "naive_topd"):
        ns["register_adv_est"](name)(ns[f"compute_{name}_advantage"])
    return ns


@pytest.fixture
def batch():
    old = torch.tensor([[-2., -3., -4., -1.], [-1., -2., -2., -2.], [-4., -2., -1., -3.]], dtype=torch.float64)
    ref = torch.tensor([[-1., -5., -4., -6.], [-3., -1., -5., -2.], [-2., -4., -1., -6.]], dtype=torch.float64)
    mask = torch.tensor([[1., 1., 1., 0.], [1., 1., 0., 0.], [1., 1., 1., 1.]], dtype=torch.float64)
    return dict(old_log_probs=old, ref_log_prob=ref, response_mask=mask,
                token_level_rewards=torch.zeros_like(old), index=np.array(["p1", "p1", "p2"]))


def test_opd_exact_and_detached(alg, batch):
    batch["old_log_probs"].requires_grad_()
    batch["ref_log_prob"].requires_grad_()
    a, r = alg["compute_opd_advantage"](**batch)
    torch.testing.assert_close(a, (batch["ref_log_prob"] - batch["old_log_probs"]) * batch["response_mask"])
    assert a is r and not a.requires_grad


@pytest.mark.parametrize("alpha", [0.01, 0.1, 0.2, 0.9, 1.0])
def test_reward_transform(alg, batch, alpha):
    r = alg["_compute_topd_immediate_rewards"](batch["old_log_probs"], batch["ref_log_prob"], batch["response_mask"], alpha)
    want = torch.log(alpha * (batch["ref_log_prob"] - batch["old_log_probs"]).exp() + 1. - alpha) * batch["response_mask"]
    torch.testing.assert_close(r, want)


def test_extreme_ratios_and_masked_nan(alg):
    old = torch.zeros(1, 4)
    ref = torch.tensor([[-1000., 1000., float("nan"), float("-inf")]])
    mask = torch.tensor([[1., 1., 0., 0.]])
    r = alg["_compute_topd_immediate_rewards"](old, ref, mask, .1)
    assert torch.isfinite(r).all()
    assert r[0, 0].item() == pytest.approx(math.log(.9), abs=1e-6)
    assert r[0, 1].item() == pytest.approx(1000. + math.log(.1), abs=1e-4)
    assert r[0, 2:].eq(0).all()


def test_suffix_average_with_right_padding(alg):
    r = torch.tensor([[1., 2., 3., 0.], [4., 0., 0., 0.]])
    mask = torch.tensor([[1., 1., 1., 0.], [1., 0., 0., 0.]])
    want = torch.tensor([[3.5, 5., 3., 0.], [4., 0., 0., 0.]])
    torch.testing.assert_close(alg["_compute_topd_token_returns"](r, mask), want)


def test_topd_matches_scalar_reference(alg, batch):
    a, returns = alg["compute_topd_advantage"](**batch, config={"topd_alpha": .1})
    mask = batch["response_mask"]
    reference = torch.zeros_like(returns)
    for row in range(len(mask)):
        length = int(mask[row].sum())
        reward = [math.log(.1 * math.exp(float(batch["ref_log_prob"][row, j] - batch["old_log_probs"][row, j])) + .9) for j in range(length)]
        for j in range(length):
            suffix = reward[j + 1:]
            reference[row, j] = reward[j] + (sum(suffix) / len(suffix) if suffix else 0.)
    torch.testing.assert_close(returns, reference)
    normalized = torch.zeros_like(reference)
    for uid in set(batch["index"]):
        rows = np.where(batch["index"] == uid)[0]
        values = [float(reference[i, j]) for i in rows for j in range(reference.shape[1]) if mask[i, j]]
        mean, std = np.mean(values), np.std(values, ddof=0)
        for i in rows:
            for j in range(reference.shape[1]):
                normalized[i, j] = (reference[i, j] - mean) / (std + 1e-6) if mask[i, j] else 0.
    torch.testing.assert_close(a, normalized)
    assert not a.requires_grad and not returns.requires_grad


def test_group_order_and_padding_do_not_change_valid_values(alg, batch):
    a, _ = alg["compute_topd_advantage"](**batch)
    order = [2, 0, 1]
    shuffled = {k: v[order] for k, v in batch.items()}
    b, _ = alg["compute_topd_advantage"](**shuffled)
    torch.testing.assert_close(b, a[order])
    padded = {k: (torch.nn.functional.pad(v, (0, 3)) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
    c, _ = alg["compute_topd_advantage"](**padded)
    torch.testing.assert_close(c[:, :4], a)
    assert c[:, 4:].eq(0).all()


@pytest.mark.parametrize("index", [None, np.array(["same", "same"])])
def test_empty_or_zero_variance_groups(alg, index):
    for mask in [torch.zeros(2, 3), torch.ones(2, 3)]:
        z = torch.zeros(2, 3)
        a = alg["_normalize_topd_token_returns_by_prompt"](z, mask, index)
        assert torch.isfinite(a).all() and a.eq(0).all()


@pytest.mark.parametrize("alpha", [0., -1., 1.01, float("nan"), float("inf")])
def test_alpha_validation(alg, alpha):
    with pytest.raises(ValueError):
        alg["_get_topd_alpha"]({"topd_alpha": alpha})


def test_naive_alpha_one_is_opd_but_full_topd_is_not(alg, batch):
    opd, _ = alg["compute_opd_advantage"](**batch)
    naive, _ = alg["compute_naive_topd_advantage"](**batch, config={"topd_alpha": 1.})
    full, _ = alg["compute_topd_advantage"](**batch, config={"topd_alpha": 1.})
    torch.testing.assert_close(naive, opd)
    assert not torch.allclose(full, opd)


def test_registry_and_reference_activation(alg):
    ns = {"AdvantageEstimator": alg["AdvantageEstimator"]}
    extract(VERL / "verl/trainer/ppo/utils.py", ["need_reference_policy"], ns)
    for name in ("opd", "topd", "naive_topd"):
        assert alg["get_adv_estimator_fn"](name) is alg[f"compute_{name}_advantage"]
        cfg = SimpleNamespace(algorithm=SimpleNamespace(use_kl_in_reward=False, adv_estimator=name),
                              actor_rollout_ref=SimpleNamespace(actor=SimpleNamespace(use_kl_loss=False)))
        assert ns["need_reference_policy"](cfg)
    cfg.algorithm.adv_estimator = "grpo"
    assert not ns["need_reference_policy"](cfg)


def test_trainer_dispatch_and_missing_teacher_error(alg, batch):
    ns = {"AdvantageEstimator": alg["AdvantageEstimator"], "core_algos": SimpleNamespace(**alg)}
    extract(VERL / "verl/trainer/ppo/ray_trainer.py", ["compute_advantage"], ns)
    for mode in ("opd", "topd", "naive_topd"):
        data = SimpleNamespace(batch={k: v for k, v in batch.items() if k != "index"}, non_tensor_batch={"uid": batch["index"]})
        got = ns["compute_advantage"](data, mode, config={"topd_alpha": .1})
        want, _ = alg[f"compute_{mode}_advantage"](**batch, config={"topd_alpha": .1})
        torch.testing.assert_close(got.batch["advantages"], want)
    del data.batch["ref_log_prob"]
    with pytest.raises(AssertionError, match="ref_log_prob"):
        ns["compute_advantage"](data, "topd", config={})


class Config(dict):
    __getattr__ = dict.__getitem__


def test_vanilla_ppo_gradient_only_updates_current_student(alg):
    current = torch.zeros(1, 3, dtype=torch.float64, requires_grad=True)
    old = torch.zeros_like(current)
    advantages = torch.tensor([[2., -1., 0.]], dtype=torch.float64)
    mask = torch.tensor([[1., 1., 0.]], dtype=torch.float64)
    cfg = Config(clip_ratio=.2, clip_ratio_low=.2, clip_ratio_high=.2, clip_ratio_c=3., global_batch_info={})
    loss, metrics = alg["compute_policy_loss_vanilla"](old, current, advantages, mask, "seq-mean-token-mean", cfg)
    loss.backward()
    torch.testing.assert_close(current.grad, torch.tensor([[-1., .5, 0.]], dtype=torch.float64))
    assert all(math.isfinite(v) for v in metrics.values())


def test_avg16_is_average_correctness_not_pass16():
    ns = {"np": np, "defaultdict": defaultdict}
    extract(VERL / "verl/trainer/ppo/ray_trainer.py", ["compute_distillation_avg_metrics"], ns)
    fn = ns["compute_distillation_avg_metrics"]
    got = fn(["aime24"] * 32, ["a"] * 16 + ["b"] * 16, {"acc": [1.] * 8 + [0.] * 8 + [0.] * 16}, 16)
    assert got["val/acc/aime24/avg@16"] == .25
    assert got["val-core/aime24/acc/avg@16"] == .25
    assert not any("pass@" in key for key in got)
    with pytest.raises(ValueError, match="Expected 16"):
        fn(["aime24"], ["a"], {"acc": [1.]}, 16)


def test_release_hashes_and_teacher_loader():
    spec = importlib.util.spec_from_file_location("release_patch", ROOT / "scripts/apply_patch.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.apply(VERL, check=True) == "already applied and verified"
    worker = (VERL / "verl/workers/fsdp_workers.py").read_text(encoding="utf-8")
    assert 'ref_model = self.config.ref.get("model", None)' in worker
    assert 'ref_model_path = ref_model.get("path", self.config.model.path)' in worker
    assert "model_path=local_path" in worker
    assert mod.apply(VERL, require_applied=True) == "already applied and verified"


def test_original_algorithm_ast_fingerprints():
    import hashlib

    expected = json.loads((ROOT / "tests/original_algorithms.json").read_text(encoding="utf-8"))["functions"]

    def normalize(node):
        if isinstance(node, ast.AST):
            return {"_type": type(node).__name__, **{k: normalize(v) for k, v in ast.iter_fields(node) if k != "type_params"}}
        if isinstance(node, list):
            return [normalize(v) for v in node]
        return node

    source = ast.parse((VERL / "verl/trainer/ppo/core_algos.py").read_text(encoding="utf-8"))
    found = {}
    for node in source.body:
        if isinstance(node, ast.FunctionDef) and node.name in expected:
            found[node.name] = hashlib.sha256(json.dumps(normalize(node), sort_keys=True).encode()).hexdigest()
    assert found == expected


def load_release_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_weight_check_rejects_config_only_and_missing_shards(tmp_path):
    mod = load_release_script("preflight")
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="Missing weights"):
        mod.check_weights(tmp_path)
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"w": "model-00001-of-00001.safetensors"}}))
    with pytest.raises(ValueError, match="Missing or invalid model shard"):
        mod.check_weights(tmp_path)
    (tmp_path / "model-00001-of-00001.safetensors").write_bytes(b"test-presence-only")
    assert mod.check_weights(tmp_path) == ["model-00001-of-00001.safetensors"]


def test_weight_check_rejects_path_escape(tmp_path):
    mod = load_release_script("preflight")
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"w": "../outside.safetensors"}}))
    with pytest.raises(ValueError, match="Missing or invalid model shard"):
        mod.check_weights(tmp_path)


def test_data_normalization_preserves_prepared_prompt():
    mod = load_release_script("prepare_data")
    row = {"prompt": [{"role": "user", "content": "Original formatting"}], "reward_model": {"ground_truth": "2"}}
    out = mod.normalize(row, 3, source="aime24", split="test")
    assert out["prompt"] == row["prompt"]
    assert out["reward_model"]["ground_truth"] == "2"
    assert out["extra_info"]["index"] == 3


def test_data_normalization_raw_schema():
    mod = load_release_script("prepare_data")
    out = mod.normalize({"Problem": "1+1?", "Answer": 2}, 0, source="aime24", split="test")
    assert out["prompt"][0]["content"].endswith(mod.PROMPT_SUFFIX)
    assert out["reward_model"]["ground_truth"] == "2"
