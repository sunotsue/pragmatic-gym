"""Paths and runtime configuration (environment variables, all optional)."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
CONFIG_DIR = ROOT / "configs"
EXPERIMENTS_DIR = ROOT / "experiments"
# Where the API writes at runtime (run index, saved counterfactual runs, exports).
# Serverless hosts only allow writes under /tmp.
RUNTIME_DIR = Path(os.environ.get("PGYM_RUNTIME_DIR") or ("/tmp/pgym" if os.environ.get("VERCEL") else EXPERIMENTS_DIR))
DB_PATH = Path(os.environ.get("PGYM_DB", RUNTIME_DIR / "pragmatic_gym.sqlite"))

# Recorded mode is the default: the app serves saved local-model outputs
# (data/recorded/) and never runs a model. PGYM_LIVE=1 runs models live.
LIVE = os.environ.get("PGYM_LIVE", "0") == "1"

# Inference backend selection: "anthropic" | "openai_compat" | "replay" | "fixture" | "auto"
BACKEND = os.environ.get("PGYM_BACKEND", "auto")
ANTHROPIC_MODEL = os.environ.get("PGYM_ANTHROPIC_MODEL", "claude-opus-5-5")
ANTHROPIC_EFFORT = os.environ.get("PGYM_ANTHROPIC_EFFORT", "medium")
OPENAI_BASE_URL = os.environ.get("PGYM_OPENAI_BASE_URL", "")  # e.g. http://localhost:8001/v1 (vLLM, llama.cpp)
OPENAI_MODEL = os.environ.get("PGYM_OPENAI_MODEL", "")
OPENAI_API_KEY = os.environ.get("PGYM_OPENAI_API_KEY", "EMPTY")
# Researcher/debug view of latent state. The API refuses latent fields unless
# the request carries this token (default is fine for local use).
RESEARCHER_TOKEN = os.environ.get("PGYM_RESEARCHER_TOKEN", "local-researcher")
