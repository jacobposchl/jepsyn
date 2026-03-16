# SNN-JEPA: Joint-Embedding Predictive Architecture for Neural Population Activity

Self-supervised learning framework for learning cross-session representations of multi-electrode Neuropixels recordings. A **LeJEPA teacher** (PerceiverIO encoder + EMA target + predictor) is trained via masked prediction, then distilled into a **Spiking Neural Network** student via CCA-based knowledge distillation.

---

## Architecture

### LeJEPA Teacher
- **Context encoder** (`PerceiverEncoder`): POYO-style PerceiverIO with per-session unit embedding tables and RoPE temporal encoding. Takes masked spike events and compresses them to a fixed set of latent slots `[B, L, D]`.
- **Target encoder**: EMA copy of the context encoder (no gradients). Sees full (unmasked) spikes; provides stable prediction targets.
- **Predictor** (`NeuralPredictor`): Narrow transformer (intentionally shallower than encoder) mapping context latents → predicted target latents.
- **Loss**: MSE prediction loss + regularization. Three modes: `sigreg` (Sketched Isotropic Gaussian), `vicreg` (Variance-Invariance-Covariance), or `no_reg`.

### SNN Student
- Distilled from teacher latents using CCA loss + homeostatic firing-rate penalty (`DistillationLoss`).
- Implemented via `snntorch` Leaky Integrate-and-Fire neurons.

---

## Project structure

```
snn-jepa/
├── jepsyn/                          # Core library
│   ├── models/
│   │   ├── encoder.py               # PerceiverEncoder (NeuralEncoder wrapper)
│   │   ├── predictor.py             # NeuralPredictor
│   │   └── snn.py                   # SNN model definition
│   ├── losses/
│   │   ├── lejepa.py                # lejepa_loss (SIGReg / VICReg / no_reg)
│   │   └── distillation.py          # DistillationLoss (CCA + homeostatic)
│   ├── data/
│   │   ├── dataset.py               # SpikeWindowDataset + spike_collate_fn
│   │   ├── data_handler.py          # Allen VBN session loading
│   │   └── preprocess.py            # Unit filtering and spike extraction
│   ├── plots/
│   │   ├── latent_space.py          # UMAP plots (by session, is_change, image)
│   │   └── training.py              # Loss curves
│   └── utils/
│       ├── training.py              # create_context_mask, update_ema, apply_unit_dropout, load_and_prepare_data
│       ├── evaluation.py            # evaluate_model, identify_units, run_linear_probe
│       ├── config_helper.py         # verify_config
│       └── results.py               # save_results
│
├── experiments/
│   ├── data/                        # Dataset pipeline (run once)
│   │   ├── create_dataset.py        # Extract sessions → write Parquet
│   │   ├── verify_dataset.py        # Validate Parquet before training
│   │   └── data_analysis.py         # Interactive session/metadata explorer
│   ├── multi_session/               # Main experiment pipeline
│   │   ├── multi_session.py         # train_lejepa, distill_snn, load_checkpoint
│   │   ├── multi_session_runner.ipynb  # Interactive runner (primary interface)
│   │   └── configs/                 # Experiment configs (see below)
│   └── single_session/              # Legacy single-session proof-of-concept
│
├── datasets/                        # Pre-built Parquet files (gitignored)
├── results/                         # Checkpoints and metrics CSVs (gitignored)
└── plots/                           # Generated figures (gitignored)
```

---

## Getting started

### 1. Build a dataset (run once)

Extract sessions from the Allen Visual Behavior Neuropixels cache and write a windowed Parquet:

```bash
python -m experiments.data.create_dataset path/to/dataset_config.yaml
```

See [experiments/data/README.md](experiments/data/README.md) for config format and session selection.

### 2. Run an experiment

The primary interface is the interactive notebook:

```
experiments/multi_session/multi_session_runner.ipynb
```

Or from the command line:

```bash
python -m experiments.multi_session.multi_session experiments/multi_session/configs/lejepa_visual_cortex.yaml
```

See [experiments/multi_session/README.md](experiments/multi_session/README.md) for config options and evaluation details.

---

## Evaluation

The model is evaluated on held-out **test sessions** — sessions the encoder has never seen during training. Because the encoder uses per-session unit embedding tables, test sessions require **unit identification** (test-time adaptation of unit embeddings via a short optimization loop) before probing.

Primary metrics:
- **Linear probe accuracy** on `is_change` (change detection), `image_name` (stimulus identity), and `session_id` — all on held-out sessions after unit identification.
- **UMAP visualizations** of the latent space colored by each of the above labels.

Training loss is a secondary diagnostic (SSL task convergence), not the primary measure of representation quality.

---

## Dependencies

```bash
pip install -r requirements.txt
```

Key libraries: `torch`, `snntorch`, `torch_brain`, `allensdk`, `numpy`, `pandas`, `scikit-learn`, `umap-learn`, `pyarrow`, `temporaldata`, `pyyaml`.
