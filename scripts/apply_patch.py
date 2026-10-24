#!/usr/bin/env python3
"""Apply the release patch only to the pinned public baseline; never reset user work."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def normalized_bytes(path: Path) -> bytes:
    return path.read_bytes().replace(b"\r\n", b"\n")


def sha256(path: Path) -> str:
    return hashlib.sha256(normalized_bytes(path)).hexdigest()


def git_blob(path: Path) -> str:
    content = normalized_bytes(path)
    return hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True, encoding="utf-8").strip()


def apply(repo: Path, *, check: bool = False, require_applied: bool = False) -> str:
    repo = repo.resolve()
    lock = json.loads((ROOT / "upstream.lock.json").read_text(encoding="utf-8"))
    if git(repo, "rev-parse", "HEAD") != lock["commit"]:
        raise ValueError("Wrong verl commit. Use scripts/setup_verl.sh; do not apply this patch to main.")
    patch = ROOT / lock["patch"]
    if sha256(patch) != lock["patch_sha256_lf"]:
        raise ValueError("Release patch checksum mismatch.")
    files = lock["patched_files"]
    if all(sha256(repo / p) == spec["sha256_lf"] for p, spec in files.items()):
        return "already applied and verified"
    if require_applied:
        raise ValueError("The release patch is not installed. Run scripts/setup_verl.sh before training.")
    for p, spec in files.items():
        if git_blob(repo / p) != spec["upstream_blob"]:
            raise ValueError(f"Refusing to overwrite a changed or partially patched file: {p}")
    subprocess.run(["git", "-C", str(repo), "apply", "--check", str(patch)], check=True)
    if check:
        return "clean pinned baseline; patch can be applied"
    subprocess.run(["git", "-C", str(repo), "apply", str(patch)], check=True)
    for p, spec in files.items():
        if sha256(repo / p) != spec["sha256_lf"]:
            raise ValueError(f"Post-apply checksum mismatch: {p}")
    return "applied and verified"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verl-dir", type=Path, default=ROOT / "third_party" / "verl")
    parser.add_argument("--check", action="store_true", help="Read-only validation of an installed or applicable patch.")
    parser.add_argument("--require-applied", action="store_true", help="Fail unless the patch is already installed; do not modify files.")
    args = parser.parse_args()
    print(apply(args.verl_dir, check=args.check, require_applied=args.require_applied))


if __name__ == "__main__":
    main()
