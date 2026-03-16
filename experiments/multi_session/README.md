# Multi-session experiment pipeline

Loads a pre-built windowed Parquet (from the dataset pipeline), splits sessions into train/test, trains the LeJEPA teacher, then distills into an SNN student. Does **not** touch the Allen cache or re-extract sessions.

## Prerequisite

Run the dataset pipeline first:

```bash
python -m experiments.data.create_dataset path/to/dataset_config.yaml
```

The experiment config's `data_path` must point to the resulting Parquet.

---

## Running

**Preferred:** open `multi_session_runner.ipynb` in VS Code or Colab. It walks through each stage interactively and shows plots inline.

**From the command line:**

```bash
python -m experiments.multi_session.multi_session path/to/config.yaml
```

---

## Pipeline stages

1. **Load & validate** — reads the Parquet, validates schema and integrity, builds per-session unit maps (raw unit ID → 1-indexed contiguous index).
2. **Session split** — splits *entire sessions* (not windows) into train / test (default 70/30, seeded). No windows from test sessions appear in training.
3. **Train LeJEPA** — context encoder (masked input) + EMA target encoder (full input) + narrow predictor. Optimizer updates context encoder and predictor only; target encoder is updated via EMA after each step.
4. **Evaluate on test sessions** — runs `identify_units` (test-time adaptation of unit embeddings for held-out sessions), then linear probes and UMAP.
5. **Distill SNN** *(in progress)* — trains a spiking student to match teacher latents via CCA loss + homeostatic penalty.

---

## Configs

All configs live in `configs/`. They share the same model architecture; only the regularization strategy differs.

| Config | Regularization | Notes |
|---|---|---|
| `lejepa_visual_cortex.yaml` | SIGReg | Main config |
| `vicreg_visual_cortex.yaml` | VICReg | Variance-covariance regularization |
| `no_reg_visual_cortex.yaml` | None | Pure JEPA prediction loss |
| `ablation_poyo.yaml` | SIGReg (minimal) | Deep predictor, higher mask ratio |

All paths in configs are relative to the config file's directory (`../../../` resolves to the repo root).

### Key config fields

```yaml
data_path: ../../../datasets/vis_all_animals_dataset.parquet

data:
  train_split: 0.7
  test_split:  0.15     # val_split also accepted but not used in evaluation
  random_state: 42

model_config:
  encoder_type: perceiver
  d_model: 256           # embedding and latent dimension
  n_latents: 64          # number of latent slots
  window_size_s: 0.4     # must match window_size_ms/1000 used in create_dataset
  n_cross_attn_heads: 4
  n_self_attn_layers: 4
  n_self_attn_heads: 8
  dim_feedforward: 1024
  dropout: 0.1
  rope_t_min: 1.0e-3     # RoPE minimum period (seconds)
  rope_t_max: 4.0        # RoPE maximum period (seconds)
  use_delimiter_tokens: true
  predictor_type: transformer
  predictor_n_layers: 2
  predictor_n_heads: 4
  predictor_dim_feedforward: 512

training_config:
  epochs: 100
  batch_size: 32
  lr: 1.0e-4
  weight_decay: 0.05
  ema_momentum: 0.996    # target encoder EMA decay
  mask_ratio: 0.5        # fraction of real tokens hidden from context encoder
  reg_type: "sigreg"     # sigreg | vicreg | no_reg
  lambd: 0.05            # SIGReg weight (ignored for vicreg/no_reg)
  num_slices: 256        # SIGReg random projections
  unit_dropout: 0.4      # fraction of units dropped per training step (POYO augmentation)
  unit_id_steps: 200     # steps for test-time unit identification
  unit_id_lr: 1.0e-4
```

---

## Checkpoint format

Saved to `<results_out_path>/lejepa_checkpoint.pt`:

```python
{
    "context_encoder": ...,   # state_dict
    "target_encoder":  ...,   # state_dict
    "predictor":       ...,   # state_dict
    "config":          ...,   # full config dict
    "unit_maps":       ...,   # {session_id: {raw_unit_id: 1-indexed idx}}
}
```

Load with `load_checkpoint(ckpt_path)` from `multi_session.py`.

---

## Evaluation

`evaluate_model` is the single entry point for the test pipeline:

```python
metrics_df, probe_results = evaluate_model(
    jepa_model, test_loader,
    stage="LeJEPA",
    mask_ratio=0.5,
    test_session_ids=test_session_ids,
    config=config,
)
```

It runs in sequence:
1. **`identify_units`** — adapts unit embeddings for test sessions (200 steps, unit embeds only).
2. **JEPA metrics** — pred loss and cosine similarity on the test set.
3. **Linear probes** — 5-fold stratified logistic regression on `is_change`, `image_name`, and `session_id`. Balanced accuracy is the primary result.

UMAP visualizations are generated separately from the returned `probe_results` dict (see notebook).

**Note on training loss:** Training loss tracks SSL task convergence and collapse prevention. It is not a direct measure of representation quality. The linear probe accuracy on held-out sessions is the primary scientific result.
