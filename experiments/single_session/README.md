# Single-session proof-of-concept (legacy)

> **Note:** This is an early proof-of-concept. The active pipeline is the multi-session experiment in [`experiments/multi_session/`](../multi_session/README.md).

Trains a JEPA teacher on one Visual Behavior Neuropixels session (binned spikes around image-change events), then distills into an SNN student. Uses an older binned-spike data format rather than the event-based format used in the multi-session pipeline.

## How to run

```bash
export MPLBACKEND=Agg  # for headless environments
python -m experiments.single_session.single_session <SESSION_ID> \
  --dataset-dir /path/to/preprocessed/session_<SESSION_ID>.pkl
```

If `--dataset-dir` is omitted, the script downloads and preprocesses the session from the Allen cache into `preprocessed/session_<SESSION_ID>.pkl`.

## Outputs

Written to `runs/proof_of_concept/run_*/`:

- `phase1_teacher.png` — JEPA losses, prediction quality, spike stats.
- `phase2_distillation.png` — CCA similarity and homeostatic penalty traces.
