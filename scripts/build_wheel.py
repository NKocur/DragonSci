#!/usr/bin/env python3
"""
Build a distribution wheel cleanly.

`maturin develop` places a compiled extension (.pyd/.so/.dylib) directly
inside the Python source package so the package is importable without
installation.  `maturin build` (wheel) copies every file from that same
directory, so the dev extension collides with the freshly-compiled one.

This script removes those dev artifacts before invoking maturin build,
making the wheel build safe to run from any working tree state.

Usage:
    python scripts/build_wheel.py [--release] [extra maturin args...]
"""
from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON_PKG = os.path.join(REPO_ROOT, "python", "dragonsci")

DEV_ARTIFACT_PATTERNS = ["*.pyd", "*.so", "*.dylib"]


def remove_dev_artifacts() -> list[str]:
    removed = []
    for pat in DEV_ARTIFACT_PATTERNS:
        for path in glob.glob(os.path.join(PYTHON_PKG, pat)):
            os.remove(path)
            removed.append(os.path.relpath(path, REPO_ROOT))
    return removed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, add_help=True)
    parser.add_argument("--release", action="store_true", default=True)
    parser.add_argument("maturin_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    removed = remove_dev_artifacts()
    if removed:
        print("Removed dev artifacts:")
        for r in removed:
            print(f"  {r}")

    cmd = ["maturin", "build"]
    if args.release:
        cmd.append("--release")
    cmd.extend(args.maturin_args)

    env = os.environ.copy()
    env.setdefault("PYO3_PYTHON", sys.executable)

    print("Running:", " ".join(cmd))
    return subprocess.call(cmd, cwd=REPO_ROOT, env=env)


if __name__ == "__main__":
    sys.exit(main())
