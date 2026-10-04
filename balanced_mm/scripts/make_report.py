#!/usr/bin/env python
"""Thin CLI wrapper: generate results/ (RESULTS.md + plots) from runs/.

    python scripts/make_report.py --runs runs --out results
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bml.analysis import main  # noqa: E402

if __name__ == "__main__":
    main()
