"""
Latent space visualization utilities for analyzing learned representations.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple


def plot_umap_by_session(
    latent_vectors: np.ndarray,
    session_labels: np.ndarray,
    stage: str,
) -> Tuple[plt.Figure, np.ndarray]:
    """
    2-D UMAP scatter coloured by session ID.

    Args:
        latent_vectors: [N, D] float array of context representations.
        session_labels: [N] int array of session IDs.
        stage:          Experiment stage label for the title.

    Returns:
        (fig, embeddings2d) — figure and the [N, 2] UMAP coordinates.
        embeddings2d is returned so the caller can reuse it for plot_umap_by_change
        without running UMAP twice.
    """
    import umap

    reducer = umap.UMAP(n_components=2, random_state=42)
    embeddings2d = reducer.fit_transform(latent_vectors)

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        embeddings2d[:, 0], embeddings2d[:, 1],
        c=session_labels, cmap="tab10", alpha=0.5, s=10,
    )
    plt.colorbar(scatter, ax=ax, label="Session ID")
    ax.set_title(f"{stage} - Latent Space (UMAP)")
    ax.set_xlabel("DIM 1")
    ax.set_ylabel("DIM 2")
    plt.tight_layout()
    return fig, embeddings2d


def plot_umap_by_image(
    embeddings2d: np.ndarray,
    image_labels: np.ndarray,
    stage: str,
) -> plt.Figure:
    """
    2-D UMAP scatter coloured by image identity.

    Reuses pre-computed 2D coordinates from plot_umap_by_session so UMAP
    is only fitted once.

    Args:
        embeddings2d: [N, 2] float array from a prior UMAP fit (already filtered
                      to valid-image rows by the caller).
        image_labels: [N] array of image name strings.
        stage:        Experiment stage label for the title.

    Returns:
        Matplotlib figure.
    """
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    encoded = le.fit_transform(image_labels)
    n_classes = len(le.classes_)
    cmap = "tab20" if n_classes <= 20 else "viridis"

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        embeddings2d[:, 0], embeddings2d[:, 1],
        c=encoded, cmap=cmap, alpha=0.5, s=10,
    )
    plt.colorbar(scatter, ax=ax, label=f"Image identity ({n_classes} images)")
    ax.set_title(f"{stage} - Latent Space by Image Identity (UMAP)")
    ax.set_xlabel("DIM 1")
    ax.set_ylabel("DIM 2")
    plt.tight_layout()
    return fig


def plot_umap_by_change(
    embeddings2d: np.ndarray,
    all_change: np.ndarray,
    valid: np.ndarray,
    stage: str,
) -> plt.Figure:
    """
    2-D UMAP scatter coloured by change-detection label.

    Reuses pre-computed 2D coordinates from plot_umap_by_session so UMAP
    is only fitted once.

    Args:
        embeddings2d: [N, 2] float array from a prior UMAP fit.
        all_change:   [N] int array (1 = change, 0 = no change).
        valid:        [N] bool array — True for stimulus windows.
        stage:        Experiment stage label for the title.

    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        embeddings2d[~valid, 0], embeddings2d[~valid, 1],
        c="lightgray", alpha=0.2, s=8, label="no stimulus",
    )
    for val, label, color in [(0, "no change", "steelblue"), (1, "change", "tomato")]:
        mask = valid & (all_change == val)
        ax.scatter(
            embeddings2d[mask, 0], embeddings2d[mask, 1],
            c=color, alpha=0.6, s=10, label=label,
        )
    ax.legend()
    ax.set_title(f"{stage} - Latent Space by Change Detection (UMAP)")
    ax.set_xlabel("DIM 1")
    ax.set_ylabel("DIM 2")
    plt.tight_layout()
    return fig
