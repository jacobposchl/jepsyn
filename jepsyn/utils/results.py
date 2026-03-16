"""
Experiment results saving: metrics CSV + plots.
"""

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from jepsyn.plots.latent_space import plot_umap_by_change, plot_umap_by_session
from jepsyn.plots.model_performance import plot_test_metrics_bar
from jepsyn.plots.training import plot_distillation_curves, plot_training_curves


def save_results(
    stage: str, phase: str, metrics: pd.DataFrame, config: Dict[str, Any]
) -> None:
    """
    Save metrics to CSV and generate phase-appropriate plots.

    Args:
        stage:   Experiment stage ("LeJEPA", "VICReg", "LeJEPA-NoReg", "SNN").
        phase:   One of "training", "test", "distillation".
        metrics: DataFrame of per-epoch or per-batch metrics.
        config:  Validated config dict; must contain results_out_path.
    """
    import matplotlib.pyplot as plt

    results_path = config.get("results_out_path")
    if not results_path:
        print("No results_out_path in config; skip saving metrics.")
        return

    out_dir = Path(results_path) / stage / phase
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics CSV (drop array columns used only for latent-space plots)
    csv_path = out_dir / "metrics.csv"
    metrics.drop(
        columns=["h_tgt", "session_ids", "is_change", "image_name", "stim_block"],
        errors="ignore",
    ).to_csv(csv_path, index=False)
    print(f"Saved metrics to {csv_path}")

    if phase == "training":
        fig, _ = plot_training_curves(metrics, stage)
        fig_path = out_dir / "training_curves.png"
        fig.savefig(fig_path)
        plt.close(fig)
        print(f"Saved training curves to {fig_path}")

    elif phase == "test":
        fig = plot_test_metrics_bar(metrics, stage)
        fig.savefig(out_dir / "test_metrics.png")
        plt.close(fig)
        print(f"Saved test metrics to {out_dir / 'test_metrics.png'}")

        if "h_tgt" in metrics.columns:
            try:
                latent_vectors = np.vstack(metrics["h_tgt"].values)
                session_labels = np.concatenate(metrics["session_ids"].values)
                fig, embeddings2d = plot_umap_by_session(latent_vectors, session_labels, stage)
                fig.savefig(out_dir / "latent_space.png")
                plt.close(fig)
                print(f"Saved latent space plot to {out_dir / 'latent_space.png'}")

                if "is_change" in metrics.columns and "stim_block" in metrics.columns:
                    all_change = np.concatenate(metrics["is_change"].values).astype(int)
                    all_block = np.concatenate(metrics["stim_block"].values)
                    valid = all_block >= 0
                    if valid.sum() >= 10:
                        fig = plot_umap_by_change(embeddings2d, all_change, valid, stage)
                        fig.savefig(out_dir / "latent_space_change.png")
                        plt.close(fig)
                        print(f"Saved change UMAP to {out_dir / 'latent_space_change.png'}")

            except ImportError:
                print("umap-learn not installed; skipping latent space plot.")

    elif phase == "distillation":
        fig, _ = plot_distillation_curves(metrics, stage)
        fig_path = out_dir / "distillation_curves.png"
        fig.savefig(fig_path)
        plt.close(fig)
        print(f"Saved distillation curves to {fig_path}")
