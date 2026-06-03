#!/usr/bin/env python3
"""
Wrapper for HLA-I unified ESM training (corrected imbalance).
Uses peptide length range 8-12 (inclusive).
"""

import os
import subprocess
import sys


def main() -> int:
    base_script = os.path.join(
        os.path.dirname(__file__), "run_unified_esm_training_corrected.py"
    )
    cmd = [
        sys.executable,
        base_script,
        "--min_length",
        "7",
        "--max_length",
        "12",
        *sys.argv[1:],
    ]
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())

