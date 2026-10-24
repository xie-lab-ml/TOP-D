"""Read-only version check against the original working environment."""
from __future__ import annotations

import importlib.metadata as metadata
import platform
import sys
from pathlib import Path


def main():
    expected = {"torch": "2.10.0", "flash-attn": "2.8.3"}
    for line in Path(__file__).with_name("requirements.txt").read_text().splitlines():
        if line and not line.startswith("#"):
            package, version = line.split("==", 1)
            expected[package.split("[")[0]] = version
    errors = []
    for package, version in expected.items():
        try:
            installed = metadata.version(package)
        except metadata.PackageNotFoundError:
            errors.append(f"{package}: missing (expected {version})")
            continue
        if installed.split("+")[0] != version:
            errors.append(f"{package}: {installed} (expected {version})")
    if platform.system() != "Linux" or sys.version_info[:2] != (3, 12):
        errors.append("Original training uses Linux / Python 3.12")
    if errors:
        raise SystemExit("Environment differs:\n  " + "\n  ".join(errors))
    import torch
    if torch.version.cuda != "12.9":
        raise SystemExit(f"Expected the CUDA 12.9 PyTorch build, got {torch.version.cuda!r}")
    print(f"Verified {len(expected)} pinned package versions and PyTorch CUDA 12.9.")
    print("This checks the environment, not GPU memory capacity or experiment accuracy.")


if __name__ == "__main__":
    main()
