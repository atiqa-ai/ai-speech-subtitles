"""Pytest configuration: make the repo root importable.

`translator_module` and `subtitle_generator` are imported as `src.<module>`,
which requires the repository root on sys.path. Using the installed package
name would mean restructuring the project, so the path is added here instead.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
