"""
Training visualization utilities for monitoring model training progress.
"""

import matplotlib.pyplot as plt
import pandas as pd
from typing import Tuple


def plot_training_curves(
    metrics: pd.DataFrame,
    stage: str,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot train total loss and pred/reg loss breakdown.

    Args:
        metrics: DataFrame with columns epoch, train_loss,
                 and optionally train_pred_loss, train_reg_loss.
        stage:   Experiment stage label (e.g. "LeJEPA", "VICReg").

    Returns:
        (fig, axes) — 1x2 figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"{stage} - Training Curves")

    if "train_loss" in metrics.columns:
        axes[0].plot(metrics["epoch"], metrics["train_loss"], label="train")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Total Loss")
        axes[0].legend()

    if "train_pred_loss" in metrics.columns:
        axes[1].plot(metrics["epoch"], metrics["train_pred_loss"], label="pred loss")
        axes[1].plot(metrics["epoch"], metrics["train_reg_loss"], label="reg loss")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Loss")
        axes[1].set_title("Pred vs Reg Loss")
        axes[1].legend()

    return fig, axes


def plot_distillation_curves(
    metrics: pd.DataFrame,
    stage: str,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot distillation training curves (total loss, distill vs homeostatic loss).

    Args:
        metrics: DataFrame with columns epoch, train_loss, and optionally
                 distill_loss, homeo_loss.
        stage:   Experiment stage label (e.g. "SNN").

    Returns:
        (fig, axes) — 1x2 figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"{stage} - Distillation Curves")

    if "train_loss" in metrics.columns:
        axes[0].plot(metrics["epoch"], metrics["train_loss"], label="train")
        if "val_loss" in metrics.columns:
            axes[0].plot(metrics["epoch"], metrics["val_loss"], label="val")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Total Loss")
        axes[0].legend()

    if "distill_loss" in metrics.columns:
        axes[1].plot(metrics["epoch"], metrics["distill_loss"], label="distill loss")
        if "homeo_loss" in metrics.columns:
            axes[1].plot(metrics["epoch"], metrics["homeo_loss"], label="homeostatic loss")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Loss")
        axes[1].set_title("Distill vs Homeostatic Loss")
        axes[1].legend()

    return fig, axes
