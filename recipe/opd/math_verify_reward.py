#!/usr/bin/env python3
"""Custom math reward based on math_verify for boxed final answers."""

from __future__ import annotations

import multiprocessing as mp
import re
from multiprocessing.connection import Connection
from typing import Any

try:
    from math_verify.errors import TimeoutException
    from math_verify.metric import verify
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig, parse
except ImportError as exc:  # pragma: no cover
    raise ImportError("Please install math-verify before using rlvr/math_verify_reward.py") from exc

_GOLD_EXTRACTION_TARGET = (LatexExtractionConfig(),)
_PRED_EXTRACTION_TARGET = (ExprExtractionConfig(), LatexExtractionConfig())
_DEFAULT_TIMEOUT_SECONDS = 1.0
_BOXED_ANSWER_RE = re.compile(r"\\boxed\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")


def _extract_vote_answer(solution_str: str) -> str:
    """Extract a lightweight final answer key for majority voting."""
    text = str(solution_str).strip()
    boxed_matches = _BOXED_ANSWER_RE.findall(text)
    if boxed_matches:
        return boxed_matches[-1].strip()
    non_empty_lines = [line.strip() for line in text.splitlines() if line.strip()]
    if non_empty_lines:
        return non_empty_lines[-1]
    return text


def _thread_safe_math_score(golds: list[str], predictions: list[str], precision: int = 6) -> float:
    """Compute math-verify score without signal-based timeouts.

    math-verify's default parser/verifier timeout uses ``signal.alarm()``, which is not
    supported when verl calls reward functions from worker threads. Disable the internal
    timeout here and rely on the outer Ray/verl task lifecycle for interruption.
    """
    extracted_predictions = [
        parse(pred, _PRED_EXTRACTION_TARGET, parsing_timeout=None) for pred in predictions
    ]
    extracted_golds = [
        parse(gold, _GOLD_EXTRACTION_TARGET, parsing_timeout=None) for gold in golds
    ]

    if any(len(gold) == 0 for gold in extracted_golds):
        raise ValueError(f"No gold targets found for at least one gold: {golds}")

    scores = [
        1.0
        if any(
            verify(gold, pred, float_rounding=precision, timeout_seconds=None)
            for gold in extracted_golds
        )
        else 0.0
        for pred in extracted_predictions
    ]
    return max(scores) if scores else 0.0


def _math_score_worker(
    conn: Connection,
    golds: list[str],
    predictions: list[str],
    precision: int,
) -> None:
    try:
        conn.send(("ok", float(_thread_safe_math_score(golds, predictions, precision))))
    except Exception:
        conn.send(("error", 0.0))
    finally:
        conn.close()


def _math_score_with_timeout(
    golds: list[str],
    predictions: list[str],
    precision: int = 6,
    timeout_seconds: float | None = _DEFAULT_TIMEOUT_SECONDS,
    timeout_score: float = 0.0,
) -> float:
    if timeout_seconds is None or timeout_seconds <= 0:
        return _thread_safe_math_score(golds, predictions, precision)

    ctx = mp.get_context("fork")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    process = ctx.Process(
        target=_math_score_worker,
        args=(child_conn, golds, predictions, precision),
        daemon=True,
    )

    try:
        process.start()
        child_conn.close()
        if parent_conn.poll(timeout_seconds):
            status, score = parent_conn.recv()
            process.join(timeout=0.1)
            return float(score) if status == "ok" else 0.0
        return timeout_score
    except Exception:
        return timeout_score
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=0.1)
        if process.is_alive():
            process.kill()
            process.join(timeout=0.1)
        parent_conn.close()
        try:
            process.close()
        except ValueError:
            pass


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
    timeout_score: float = 0.0,
    timeout_seconds: float | None = _DEFAULT_TIMEOUT_SECONDS,
    **kwargs: Any,
) -> dict[str, Any]:
    ground_truth_boxed = "\\boxed{" + str(ground_truth) + "}"
    score = _math_score_with_timeout(
        [ground_truth_boxed],
        [solution_str],
        timeout_seconds=timeout_seconds,
        timeout_score=timeout_score,
    )

    score = float(score)
    return {
        "score": score,
        "acc": bool(score > 0.0),
        "pred": _extract_vote_answer(solution_str),
    }
