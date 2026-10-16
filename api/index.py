"""Vercel entrypoint: serves the Pragmatic Gym API in recorded mode (no model runs)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from pgym.api.main import app  # noqa: E402,F401
