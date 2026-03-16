# Experiments

The multi-session workflow is split into two pipelines so you can build a dataset once and run many experiments without re-extracting sessions.

---

## 1. Dataset pipeline (run once)

**Purpose:** Extract sessions from the Allen cache, filter units, and write a single windowed Parquet.

- **Entry point:** [`experiments/data/create_dataset.py`](data/create_dataset.py)
- **Input:** YAML config with `data_path` (output Parquet) and `dataset_config` (cache dir, session IDs, brain areas, quality thresholds, windowing).
- **Output:** One Parquet at `data_path` — one row per temporal window, with `session_id`, `window_id`, `events_units`, `events_times_ms`, `stimulus`, `behavior`.

Re-run only when you want to change sessions, brain areas, or windowing parameters.

**Details:** [experiments/data/README.md](data/README.md)

---

## 2. Experiment pipeline (run many times)

**Purpose:** Load the pre-built Parquet, split by session, train LeJEPA, evaluate on held-out sessions, and distill into an SNN.

- **Primary interface:** [`experiments/multi_session/multi_session_runner.ipynb`](multi_session/multi_session_runner.ipynb)
- **Script entry point:** [`experiments/multi_session/multi_session.py`](multi_session/multi_session.py)
- **No session extraction:** reads only the Parquet from step 1.

**Details:** [experiments/multi_session/README.md](multi_session/README.md)

---

## Quick reference

| Step | Command | Config contains |
|------|---------|-----------------|
| **1. Create dataset** | `python -m experiments.data.create_dataset <config.yaml>` | `data_path`, `dataset_config` (cache_dir, session_ids, brain_areas, quality, windowing) |
| **2. Run experiment** | notebook or `python -m experiments.multi_session.multi_session <config.yaml>` | `data_path` (Parquet from step 1), data splits, model config, training config |
