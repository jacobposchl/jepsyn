# Dataset pipeline (step 1)

Extracts sessions from the Allen Visual Behavior Neuropixels cache and writes a single windowed Parquet. Run this **once** (or when you change sessions, regions, or windowing). The experiment pipeline reads the resulting Parquet and never touches the Allen cache.

Entry points:

- **`data_analysis.py`** — interactive exploration of sessions and metadata (choose sessions/regions before building a dataset).
- **`create_dataset.py`** — build the windowed Parquet consumed by the experiment pipeline.
- **`verify_dataset.py`** — validate a Parquet before running experiments.

---

## `data_analysis.py` — session and metadata exploration

```bash
# Overview of available sessions (fast, metadata only)
python -m experiments.data.data_analysis --overview \
  --cache-dir ./visual_behavior_neuropixels_data

# Detailed summary for one session
python -m experiments.data.data_analysis --session 1064415305 \
  --cache-dir ./visual_behavior_neuropixels_data

# Filter sessions by region and unit count (metadata only, fast)
python -m experiments.data.data_analysis --filter \
  --cache-dir ./visual_behavior_neuropixels_data \
  --animals all \
  --regions VISp VISl \
  --units-required 300 300 \
  --phase 1
```

Use this to decide which sessions and brain areas to include in your `create_dataset.py` config.

---

## `create_dataset.py` — build the Parquet

```bash
python -m experiments.data.create_dataset path/to/config.yaml
```

### Output format

One row per temporal window:

| Column | Description |
|---|---|
| `session_id` | Allen ecephys session ID |
| `window_id` | Unique integer per window |
| `window_start_ms`, `window_end_ms` | Window bounds in absolute session time (ms) |
| `events_units` | Array of unit IDs (raw Allen IDs) for each spike |
| `events_times_ms` | Array of spike times relative to `window_start_ms` — values in `[0, window_size_ms)` |
| `stimulus` | List of stimulus event dicts for overlapping stimuli (image_name, is_change, stimulus_block) |
| `behavior` | List of behavioral event dicts |

### Config format

```yaml
data_path: ./datasets/visual_cortex_windows.parquet  # output path; use this as data_path in experiment config

dataset_config:
  cache_dir: ./visual_behavior_neuropixels_data       # Allen cache (hundreds of GB)
  session_ids: [1064415305, 1064644573]               # explicit session IDs to include
  brain_areas: ["VISp", "VISl"]                       # areas to keep
  quality:
    min_snr: 1.0
    min_firing_rate: 0.1
    max_isi_violations: 1.0
  windowing:
    window_size_ms: 400   # window length; set model_config.window_size_s = window_size_ms / 1000
    stride_ms: 200        # step between window starts
```

> **Important:** `window_size_ms` here must match `model_config.window_size_s * 1000` in your experiment config.

---

## `verify_dataset.py` — validate before training

```bash
python -m experiments.data.verify_dataset path/to/dataset.parquet
```

Checks schema, session count (≥3 required for train/test split), window integrity, and array alignment. Saves a diagnostic plot (`dataset_verification_advanced.png`) next to the Parquet with a raster, spike count histogram, and per-session PSTHs.
