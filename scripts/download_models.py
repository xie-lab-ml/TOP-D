#!/usr/bin/env python3
"""Download public weights on the training machine and record exact Hub revisions."""
from __future__ import annotations

import json
import os
from pathlib import Path

from preflight import check_weights


def main() -> None:
    from huggingface_hub import HfApi, snapshot_download

    root = Path(os.environ.get("MODEL_DIR", Path(__file__).resolve().parents[1] / "model"))
    root.mkdir(parents=True, exist_ok=True)
    choices = [
        ("student", os.environ.get("STUDENT_REPO", "Qwen/Qwen3-1.7B-Base")),
        ("teacher", os.environ.get("TEACHER_REPO", "Qwen/Qwen3-30B-A3B-Instruct-2507")),
    ]
    manifest_path = root / "models.lock.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest = {}
    api = HfApi()
    for role, repo in choices:
        default_revision = previous.get(role, {}).get("revision", "main") if previous.get(role, {}).get("repo") == repo else "main"
        requested = os.environ.get(role.upper() + "_REVISION", default_revision)
        revision = api.model_info(repo, revision=requested).sha
        destination = root / repo.rsplit("/", 1)[-1]
        print(f"Downloading {role}: {repo}@{revision} -> {destination}")
        snapshot_download(repo_id=repo, revision=revision, local_dir=str(destination))
        if not (destination / "config.json").is_file():
            raise RuntimeError(f"Incomplete model download: {destination}")
        shards = check_weights(destination)
        manifest[role] = {"repo": repo, "revision": revision, "directory": str(destination), "weight_shards": len(shards)}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("Exact model revisions saved beside downloaded weights; do not commit the weights.")


if __name__ == "__main__":
    main()
