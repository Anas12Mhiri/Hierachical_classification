#!/usr/bin/env python
"""Entry point: python run_baseline.py --data-dir <path> [--out-dir <path>] [--limit N]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from aragenre_baseline.pipeline import main  # noqa: E402

if __name__ == "__main__":
    main()
