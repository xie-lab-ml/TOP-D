#!/usr/bin/env python3
"""Author's math-verify reward and timeout semantics; Linux training only."""
from __future__ import annotations

import multiprocessing as mp
import re
from multiprocessing.connection import Connection
from typing import Any

try:
    from math_verify.metric import verify
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig, parse
except ImportError as exc:
    raise ImportError("Install math-verify==0.9.0 before using this reward function.") from exc

_GOLD_EXTRACTION_TARGET = (LatexExtractionConfig(),)
_PRED_EXTRACTION_TARGET = (ExprExtractionConfig(), LatexExtractionConfig())
_DEFAULT_TIMEOUT_SECONDS = 1.0
_BOXED_ANSWER_RE = re.compile(r"\\boxed\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")


def _extract_vote_answer(solution_str: str) -> str:
    text = str(solution_str).strip()
    boxed_matches = _BOXED_ANSWER_RE.findall(text)
    if boxed_matches:
        return boxed_matches[-1].strip()
    non_empty_lines = [line.strip() for line in text.splitlines() if line.strip()]
    return non_empty_lines[-1] if non_empty_lines else text


def _thread_safe_math_score(golds: list[str], predictions: list[str], precision: int = 6) -> float:
    extracted_predictions = [parse(p, _PRED_EXTRACTION_TARGET, parsing_timeout=None) for p in predictions]
    extracted_golds = [parse(g, _GOLD_EXTRACTION_TARGET, parsing_timeout=None) for g in golds]
    if any(len(gold) == 0 for gold in extracted_golds):
        raise ValueError("No gold targets found for at least one gold.")
    scores = [
        1.0 if any(verify(g, p, float_rounding=precision, timeout_seconds=None) for g in extracted_golds) else 0.0
        for p in extracted_predictions
    ]
    return max(scores) if scores else 0.0


def _math_score_worker(conn: Connection, golds: list[str], predictions: list[str], precision: int) -> None:
    try:
        conn.send(("ok", float(_thread_safe_math_score(golds, predictions, precision))))
    except Exception:
        conn.send(("error", 0.0))
    finally:
        conn.close()


def _math_score_with_timeout(
    golds: list[str], predictions: list[str], precision: int = 6,
    timeout_seconds: float | None = _DEFAULT_TIMEOUT_SECONDS, timeout_score: float = 0.0,
) -> float:
    if timeout_seconds is None or timeout_seconds <= 0:
        return _thread_safe_math_score(golds, predictions, precision)
    # Preserve the author's process timeout. Explicitly fail on unsupported OS.
    if "fork" not in mp.get_all_start_methods():
        raise RuntimeError("This training reward requires Linux fork; do not score all answers as zero on Windows.")
    ctx = mp.get_context("fork")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    process = ctx.Process(target=_math_score_worker, args=(child_conn, golds, predictions, precision), daemon=True)
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
        child_conn.close()
        try:
            process.close()
        except ValueError:
            pass


def compute_score(
    data_source: str, solution_str: str, ground_truth: str,
    extra_info: dict[str, Any] | None = None, timeout_score: float = 0.0,
    timeout_seconds: float | None = _DEFAULT_TIMEOUT_SECONDS, **kwargs: Any,
) -> dict[str, Any]:
    score = float(_math_score_with_timeout(
        ["\\boxed{" + str(ground_truth) + "}"], [solution_str],
        timeout_seconds=timeout_seconds, timeout_score=timeout_score,
    ))
    return {"score": score, "acc": bool(score > 0.0), "pred": _extract_vote_answer(solution_str)}
