PY := backend/.venv/bin/python

.PHONY: dev install test pool score-pool experiment experiment-quick screenshots audit

dev:            ## API + web (one command)
	./scripts/dev.sh

install:
	cd backend && uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python -e ".[dev,train]"
	cd apps/web && npm install
	$(PY) -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-0.5B-Instruct')"  # ~1 GB, so the first demo run isn't a download

test:           ## full test suite (~30s, CPU only, no model downloads)
	$(PY) -m pytest tests -q

pool:           ## (re)build candidate pool + eval sets (deterministic)
	$(PY) -c "import sys; sys.path.insert(0,'backend'); from pgym.selection.pool import build_all; print(build_all())"

score-pool:     ## base-model failure scores for failure-targeted selection (~7 min on Apple M-series)
	$(PY) scripts/score_pool.py

experiment:     ## the full pre-registered experiment: 5 arms x 3 seeds + base (~100 min on M-series)
	$(PY) scripts/run_experiment.py --exp social-sft-v1

experiment-quick: ## one seed, three arms (~20 min)
	$(PY) scripts/run_experiment.py --exp quick --arms base random counterfactual_contrast --seeds 0

audit:          ## reward audit for every environment
	$(PY) -c "import sys,json; sys.path.insert(0,'backend'); from pgym.environments.policies import audit; [print(e, json.dumps(audit(e, n=300)['checks'])) for e in ['soft_no','same_page','group_chat','family_subtext','its_fine']]"

screenshots:    ## 1440x900 screenshots of every view (app must be running)
	node scripts/screenshot_app.mjs docs/design/app --interact
