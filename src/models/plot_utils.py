# src/models/plot_utils.py
# Shared plotting utilities for LSTM and EDMD model evaluation.
# All functions write to disk (save_path) and optionally show interactively.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import List, Dict, Any, Optional


# ------------------------------------------------------------------
# 1. Next-day prediction plot (historical context + predicted point)
# ------------------------------------------------------------------

def plot_next_day_prediction(
    historical_features_dicts: List[Dict[str, Any]],
    predicted_features_dict: Dict[str, Any],
    num_days_for_state: int,
    plot_keys: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    Plots key predicted metrics against historical data.
    Supports nested keys via dot notation (e.g. 'sleep_metrics.total_sleep_seconds').

    Args:
        historical_features_dicts: Full list of processed feature dicts, oldest→newest.
        predicted_features_dict:   Single predicted next-day feature dict.
        num_days_for_state:        Number of days used as state context.
        plot_keys:                 Keys to plot. Defaults to 4 core metrics.
        save_path:                 If provided, saves figure to this path.
        show:                      Whether to call plt.show().
    """
    if plot_keys is None:
        plot_keys = [
            'avg_heart_rate',
            'avg_stress',
            'body_battery_end_value',
            'sleep_metrics.total_sleep_seconds',
        ]

    if len(historical_features_dicts) < num_days_for_state + 1:
        print("Not enough historical data to generate prediction plot.")
        return

    relevant = historical_features_dicts[-(num_days_for_state + 1):]

    def _get(d, key):
        if '.' in key:
            main, sub = key.split('.', 1)
            return d.get(main, {}).get(sub, np.nan)
        return d.get(key, np.nan)

    dates = [d['date'] for d in relevant] + [predicted_features_dict['date']]
    historical_vals = {k: [_get(d, k) for d in relevant] for k in plot_keys}
    predicted_vals  = {k: _get(predicted_features_dict, k) for k in plot_keys}

    fig, axes = plt.subplots(len(plot_keys), 1,
                             figsize=(12, 4 * len(plot_keys)), sharex=True)
    if len(plot_keys) == 1:
        axes = [axes]

    x = np.arange(len(dates))
    for i, key in enumerate(plot_keys):
        ax = axes[i]
        ax.plot(x[:num_days_for_state], historical_vals[key][:num_days_for_state],
                marker='o', linestyle='-', color='steelblue', label='Historical')
        ax.plot(x[num_days_for_state], historical_vals[key][num_days_for_state],
                marker='o', color='green', markersize=8, label='Actual next day')
        ax.plot(x[num_days_for_state], predicted_vals[key],
                marker='x', color='red', markersize=10, mew=2, label='Predicted')
        ax.set_title(key.replace('_', ' ').replace('sleep metrics ', '').title())
        ax.set_ylabel('Value')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.xticks(x, dates, rotation=45, ha='right')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Prediction plot saved to {save_path}")
    if show:
        plt.show()
    plt.close()


# ------------------------------------------------------------------
# 2. Model diagnostic plot (scatter + residuals + action influence)
# ------------------------------------------------------------------

def plot_model_diagnostics(
    pred_unscaled: np.ndarray,
    true_unscaled: np.ndarray,
    model_feature_keys: List[str],
    agent_feature_keys: Optional[List[str]] = None,
    linear_action_weights: Optional[np.ndarray] = None,
    model_name: str = "Model",
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    Three-panel diagnostic figure:
        1. Predicted vs actual scatter per feature
        2. Normalised residual boxplots
        3. Action feature total influence (if weights provided)

    Args:
        pred_unscaled:         (N, 12) model predictions in human-readable units.
        true_unscaled:         (N, 12) ground truth in human-readable units.
        model_feature_keys:    Names of the 12 model output features.
        agent_feature_keys:    Names of the 11 agent action features (for panel 3).
        linear_action_weights: (12, 11) absolute linear K/W matrix weights (for panel 3).
                               If None, panel 3 is skipped.
        model_name:            Used in figure title.
        save_path:             If provided, saves figure to this path.
        show:                  Whether to call plt.show().
    """
    n_features  = len(model_feature_keys)
    show_panel3 = (linear_action_weights is not None and agent_feature_keys is not None)
    n_panels    = 3 if show_panel3 else 2

    fig = plt.figure(figsize=(20, 8 * n_panels))
    gs  = gridspec.GridSpec(n_panels, 1, hspace=0.45)

    # --- Panel 1: Predicted vs Actual ---
    gs1 = gridspec.GridSpecFromSubplotSpec(4, 3, subplot_spec=gs[0],
                                           hspace=0.55, wspace=0.35)
    for i, key in enumerate(model_feature_keys):
        ax = fig.add_subplot(gs1[i // 3, i % 3])
        ax.scatter(true_unscaled[:, i], pred_unscaled[:, i],
                   alpha=0.45, s=14, color='steelblue', edgecolors='none')
        mn = min(true_unscaled[:, i].min(), pred_unscaled[:, i].min())
        mx = max(true_unscaled[:, i].max(), pred_unscaled[:, i].max())
        ax.plot([mn, mx], [mn, mx], 'r--', lw=1)
        mae  = float(np.mean(np.abs(pred_unscaled[:, i] - true_unscaled[:, i])))
        short = key.replace('_seconds', 's').replace('_', ' ')
        ax.set_title(f"{short}\nMAE={mae:.1f}", fontsize=8)
        ax.set_xlabel("Actual", fontsize=7)
        ax.set_ylabel("Predicted", fontsize=7)
        ax.tick_params(labelsize=7)

    fig.text(0.5, 0.99, f"{model_name} — Predicted vs Actual (unscaled)",
             ha='center', fontsize=13, fontweight='bold')

    # --- Panel 2: Residuals boxplot ---
    ax2 = fig.add_subplot(gs[1])
    residuals = pred_unscaled - true_unscaled
    stds      = true_unscaled.std(axis=0) + 1e-8
    norm_res  = residuals / stds

    bp = ax2.boxplot(norm_res, patch_artist=True,
                     medianprops=dict(color='red', lw=2))
    for patch in bp['boxes']:
        patch.set_facecolor('lightsteelblue')
        patch.set_alpha(0.7)
    ax2.axhline(0, color='gray', lw=1, ls='--')
    ax2.set_xticks(range(1, n_features + 1))
    ax2.set_xticklabels(
        [k.replace('_seconds', 's').replace('_', '\n') for k in model_feature_keys],
        fontsize=7,
    )
    ax2.set_ylabel("Normalised residual (pred − actual) / std", fontsize=9)
    ax2.set_title("Residual Distribution per Feature (normalised)", fontsize=11)
    ax2.grid(axis='y', alpha=0.3)

    # --- Panel 3: Action feature influence (optional) ---
    if show_panel3:
        ax3 = fig.add_subplot(gs[2])
        total_influence = linear_action_weights.sum(axis=0)  # (11,)
        colors = [
            '#e74c3c' if 'bed' in k or 'wake' in k
            else '#2ecc71' if 'step' in k
            else '#3498db'
            for k in agent_feature_keys
        ]
        bars = ax3.bar(agent_feature_keys, total_influence,
                       color=colors, edgecolor='white', lw=0.5)
        ax3.set_xticks(range(len(agent_feature_keys)))
        ax3.set_xticklabels(
            [k.replace('_', '\n').replace('activity\n', '') for k in agent_feature_keys],
            fontsize=8,
        )
        ax3.set_ylabel("Total |weight| across all predictions", fontsize=9)
        ax3.set_title(
            f"{model_name} — Action Feature Influence (linear weight matrix)\n"
            "Red=sleep timing   Green=steps   Blue=activity",
            fontsize=11,
        )
        ax3.grid(axis='y', alpha=0.3)
        for bar, val in zip(bars, total_influence):
            ax3.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + total_influence.max() * 0.01,
                     f"{val:.4f}", ha='center', fontsize=7)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Diagnostic plot saved to {save_path}")
    if show:
        plt.show()
    plt.close()


# ------------------------------------------------------------------
# 3. Training loss curve (for LSTM)
# ------------------------------------------------------------------

def plot_training_history(
    train_losses: List[float],
    val_losses: List[float],
    model_name: str = "LSTM",
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    Plots train and validation loss over epochs.

    Args:
        train_losses: List of per-epoch training losses.
        val_losses:   List of per-epoch validation losses.
        model_name:   Used in figure title.
        save_path:    If provided, saves figure to this path.
        show:         Whether to call plt.show().
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label='Train loss', color='steelblue')
    ax.plot(epochs, val_losses,   label='Val loss',   color='tomato')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"{model_name} — Training History")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training history saved to {save_path}")
    if show:
        plt.show()
    plt.close()