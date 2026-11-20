"""Figures for the report.

All text is English so the figures render identically without a CJK font
installed.  Nothing here recomputes a metric: plots receive the same arrays the
tables were built from, so a figure and a number can never disagree.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display in this environment
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

INK = "#1b2733"
MUTED = "#6b7a8d"
ACCENT = "#2f6f9f"
WARM = "#c2703d"
GRID = "#dde3ea"

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.dpi": 130,
        "font.size": 9,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def data_overview(
    labels: np.ndarray,
    scaffold_group_sizes: np.ndarray,
    out_path: Path,
    *,
    label_name: str = "logD7.4",
) -> Path:
    """Label distribution and how concentrated the scaffold series are."""
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4))

    axes[0].hist(labels, bins=45, color=ACCENT, alpha=0.85, edgecolor="white", linewidth=0.4)
    axes[0].axvline(float(np.median(labels)), color=WARM, linewidth=1.4,
                    label=f"median {np.median(labels):.2f}")
    axes[0].set_xlabel(f"experimental {label_name}")
    axes[0].set_ylabel("compounds")
    axes[0].set_title(f"A. {label_name} distribution (n={labels.size})")
    axes[0].legend()

    counts = np.sort(scaffold_group_sizes)[::-1]
    axes[1].plot(np.arange(1, counts.size + 1), counts, color=ACCENT, linewidth=1.4)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("scaffold rank (log)")
    axes[1].set_ylabel("compounds sharing it (log)")
    singletons = int(np.sum(counts == 1))
    axes[1].set_title(
        f"B. {counts.size} scaffolds; {singletons} seen once "
        f"({singletons / counts.size:.0%})"
    )
    return _save(fig, out_path)


def duplicate_spread(spreads: np.ndarray, out_path: Path, *, label_name: str = "logD7.4") -> Path:
    """How far apart repeat measurements of the same compound are."""
    fig, ax = plt.subplots(figsize=(5.2, 3.3))
    if spreads.size == 0:
        ax.text(0.5, 0.5, "no repeat measurements found", ha="center", va="center",
                transform=ax.transAxes, color=MUTED)
        ax.set_axis_off()
        return _save(fig, out_path)
    ax.hist(spreads, bins=30, color=WARM, alpha=0.85, edgecolor="white", linewidth=0.4)
    ax.axvline(float(np.mean(spreads)), color=INK, linewidth=1.4,
               label=f"mean {np.mean(spreads):.2f}")
    ax.set_xlabel(f"max - min {label_name} within one compound")
    ax.set_ylabel("compounds")
    ax.set_title(f"Repeat-measurement spread (n={spreads.size} compounds)")
    ax.legend()
    return _save(fig, out_path)


def model_comparison(
    names: list[str],
    scaffold_mae: list[float],
    scaffold_lo: list[float],
    scaffold_hi: list[float],
    random_mae: list[float],
    out_path: Path,
    *,
    noise_floor: float | None = None,
) -> Path:
    """Test MAE per model, scaffold split vs random split."""
    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    y = np.arange(len(names))
    height = 0.38

    lower = np.maximum(np.array(scaffold_mae) - np.array(scaffold_lo), 0)
    upper = np.maximum(np.array(scaffold_hi) - np.array(scaffold_mae), 0)
    ax.barh(y + height / 2, scaffold_mae, height=height, color=ACCENT,
            label="scaffold split (reported)",
            xerr=np.vstack([lower, upper]), error_kw={"ecolor": INK, "elinewidth": 0.9,
                                                      "capsize": 2.5})
    ax.barh(y - height / 2, random_mae, height=height, color=MUTED, alpha=0.65,
            label="random split (optimistic)")

    if noise_floor:
        ax.axvline(noise_floor, color=WARM, linestyle="--", linewidth=1.3,
                   label=f"experimental noise floor ≈ {noise_floor:.2f}")

    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("test MAE in logD units (lower is better; bars = 95% bootstrap CI)")
    ax.set_title("Model ladder: what each rung adds")
    ax.legend(loc="lower right")
    return _save(fig, out_path)


def observed_vs_predicted(
    panels: list[tuple[str, np.ndarray, np.ndarray]],
    out_path: Path,
    *,
    label_name: str = "logD7.4",
) -> Path:
    """Scatter of observed against predicted for a few named models."""
    fig, axes = plt.subplots(1, len(panels), figsize=(4.0 * len(panels), 3.9),
                             squeeze=False)
    everything = np.concatenate([np.concatenate([obs, pred]) for _, obs, pred in panels])
    low, high = float(everything.min()) - 0.3, float(everything.max()) + 0.3

    for ax, (title, observed, predicted) in zip(axes[0], panels):
        ax.scatter(observed, predicted, s=7, alpha=0.4, color=ACCENT,
                   edgecolors="none")
        ax.plot([low, high], [low, high], color=INK, linewidth=1.0)
        ax.plot([low, high], [low + 1, high + 1], color=MUTED, linewidth=0.7,
                linestyle=":")
        ax.plot([low, high], [low - 1, high - 1], color=MUTED, linewidth=0.7,
                linestyle=":")
        mae = float(np.mean(np.abs(predicted - observed)))
        rmse = float(np.sqrt(np.mean((predicted - observed) ** 2)))
        ax.set_xlim(low, high)
        ax.set_ylim(low, high)
        ax.set_aspect("equal")
        ax.set_xlabel(f"observed {label_name}")
        ax.set_ylabel(f"predicted {label_name}")
        ax.set_title(f"{title}\nMAE {mae:.3f} · RMSE {rmse:.3f}")
    fig.suptitle("Test set, scaffold split (dotted lines = ±1 log unit)", y=1.02,
                 fontsize=9, color=MUTED)
    return _save(fig, out_path)


def residual_diagnostics(
    observed: np.ndarray, predicted: np.ndarray, out_path: Path,
    *, label_name: str = "logD7.4",
) -> Path:
    """Residual distribution and residual against prediction."""
    residual = predicted - observed
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4))

    axes[0].hist(residual, bins=40, color=ACCENT, alpha=0.85, edgecolor="white",
                 linewidth=0.4)
    axes[0].axvline(0, color=INK, linewidth=1.0)
    axes[0].axvline(float(np.mean(residual)), color=WARM, linewidth=1.3,
                    label=f"mean {np.mean(residual):+.3f}")
    axes[0].set_xlabel(f"predicted - observed ({label_name})")
    axes[0].set_ylabel("compounds")
    axes[0].set_title("A. Residual distribution")
    axes[0].legend()

    axes[1].scatter(predicted, residual, s=7, alpha=0.4, color=ACCENT,
                    edgecolors="none")
    axes[1].axhline(0, color=INK, linewidth=1.0)
    # A simple binned mean shows whether the error is flat across the range.
    order = np.argsort(predicted)
    n_bins = 10
    edges = np.linspace(0, predicted.size, n_bins + 1).astype(int)
    centres, means = [], []
    for start, stop in zip(edges[:-1], edges[1:]):
        if stop > start:
            chunk = order[start:stop]
            centres.append(float(np.mean(predicted[chunk])))
            means.append(float(np.mean(residual[chunk])))
    axes[1].plot(centres, means, color=WARM, linewidth=1.6, marker="o",
                 markersize=3.5, label="binned mean residual")
    axes[1].set_xlabel(f"predicted {label_name}")
    axes[1].set_ylabel("residual")
    axes[1].set_title("B. Residual vs prediction")
    axes[1].legend()
    return _save(fig, out_path)


def applicability_domain(
    similarity: np.ndarray, absolute_error: np.ndarray, out_path: Path
) -> Path:
    """Error against distance from the training set."""
    fig, ax = plt.subplots(figsize=(5.8, 3.5))
    ax.scatter(similarity, absolute_error, s=7, alpha=0.3, color=MUTED,
               edgecolors="none")

    bins = np.array([0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.01])
    centres, means, counts = [], [], []
    for low, high in zip(bins[:-1], bins[1:]):
        mask = (similarity >= low) & (similarity < high)
        if mask.sum() >= 10:
            centres.append((low + min(high, 1.0)) / 2)
            means.append(float(np.mean(absolute_error[mask])))
            counts.append(int(mask.sum()))
    ax.plot(centres, means, color=WARM, linewidth=1.8, marker="o", markersize=4.5,
            label="mean |error| per similarity bin")
    for x, y, n in zip(centres, means, counts):
        ax.annotate(f"n={n}", (x, y), textcoords="offset points", xytext=(0, 7),
                    ha="center", fontsize=7, color=MUTED)

    ax.set_xlabel("Tanimoto similarity to the nearest training compound")
    ax.set_ylabel("absolute error (logD units)")
    ax.set_title("Applicability domain: error grows with novelty")
    ax.legend()
    return _save(fig, out_path)


def descriptor_coefficients(
    names: list[str], coefficients: np.ndarray, out_path: Path
) -> Path:
    """Standardised ridge coefficients, as a readability check on the model."""
    order = np.argsort(np.abs(coefficients))[::-1]
    order = order[:14][::-1]
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    colours = [ACCENT if coefficients[i] > 0 else WARM for i in order]
    ax.barh(np.arange(len(order)), coefficients[order], color=colours)
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels([names[i] for i in order])
    ax.axvline(0, color=INK, linewidth=0.9)
    ax.set_xlabel("standardised ridge coefficient (logD units per SD)")
    ax.set_title("What the linear descriptor model relies on")
    return _save(fig, out_path)
