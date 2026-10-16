import os
import sys
from pathlib import Path

os.environ.setdefault("PGYM_LIVE", "1")  # tests exercise the live backends; the app defaults to recorded mode

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
