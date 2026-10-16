"""SQLite store for experiment metadata (designs, runs, rollouts, datasets).

Large artifacts (datasets, adapters, per-item predictions) live as files under
experiments/ and data/; the database indexes them with hashes and seeds so any
number shown in the UI can be traced back to the exact inputs that produced it.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

from pgym.config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
  id TEXT PRIMARY KEY,
  kind TEXT NOT NULL,          -- counterfactual | corpus_eval | rollout | dataset | training
  created_at TEXT NOT NULL,
  title TEXT,
  backend TEXT,
  model TEXT,
  is_real_inference INTEGER,
  config TEXT,                 -- JSON
  summary TEXT,                -- JSON
  artifact_path TEXT
);
"""


def conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB_PATH)
    c.execute(SCHEMA)
    return c


def save(eid: str, kind: str, title: str, config: dict, summary: dict, backend: str = "", model: str = "",
         is_real: bool | None = None, artifact_path: str = "") -> None:
    with conn() as c:
        c.execute("INSERT OR REPLACE INTO experiments VALUES (?,?,?,?,?,?,?,?,?,?)",
                  (eid, kind, datetime.now(timezone.utc).isoformat(timespec="seconds"), title, backend, model,
                   None if is_real is None else int(is_real), json.dumps(config), json.dumps(summary), artifact_path))


def list_experiments(kind: str | None = None, limit: int = 50) -> list[dict]:
    with conn() as c:
        c.row_factory = sqlite3.Row
        q = "SELECT * FROM experiments" + (" WHERE kind = ?" if kind else "") + " ORDER BY created_at DESC LIMIT ?"
        rows = c.execute(q, (kind, limit) if kind else (limit,)).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["config"] = json.loads(d["config"] or "{}")
        d["summary"] = json.loads(d["summary"] or "{}")
        out.append(d)
    return out


def get(eid: str) -> dict | None:
    with conn() as c:
        c.row_factory = sqlite3.Row
        r = c.execute("SELECT * FROM experiments WHERE id = ?", (eid,)).fetchone()
    if not r:
        return None
    d = dict(r)
    d["config"] = json.loads(d["config"] or "{}")
    d["summary"] = json.loads(d["summary"] or "{}")
    return d
