# src/models/train_edmd.py
# Trains the EDMDModel on a hybrid dataset:
#   - Real Garmin data for biometric state distribution
#   - Synthetic action variants + DeterministicEnv analytical targets
#     to inject action→biometric sensitivity that real data lacks
#
# Run from project root:
#   python3 -m src.models.train_edmd

import numpy as np
from datetime import date
from sklearn.model_selection import train_test_split

from src import config
from src.data_ingestion.garmin_parser import get_historical_metrics
from src.features.feature_engineer import extract_daily_features
from src.models.data_processor import DataProcessor
from src.models.edmd_model import EDMDModel


# ------------------------------------------------------------------
# Analytical biometric function (extracted from DeterministicEnv)
# Returns unscaled (12,) model features given unscaled (11,) action
# ------------------------------------------------------------------

def _analytical_biometrics(action: np.ndarray) -> np.ndarray:
    """
    Standalone version of DeterministicEnv._predict_next_state().
    Maps an unscaled 11-feature action vector to unscaled 12 biometric features.

    Action order:
        [total_steps, Strength, Cardio, Yoga, Stretching, OtherActivity,
         NoActivity, bed_hour, bed_minute, wake_hour, wake_minute]

    Returns (12,) in model_features order:
        [avg_hr, rhr, resp, stress, body_battery,
         total_sleep, deep_sleep, rem_sleep, awake_sleep,
         restless, avg_sleep_stress, sleep_rhr]
    """
    total_steps    = action[0]
    is_strength    = action[1]
    is_cardio      = action[2]
    is_no_activity = action[6]
    bed_hour       = action[7]
    bed_minute     = action[8]
    wake_hour      = action[9]
    wake_minute    = action[10]

    # Bedtime score — peak at 22.5h, unwrap late hours
    bed_hour_unwrapped = bed_hour + 24.0 if bed_hour < 6 else bed_hour
    bed_time  = bed_hour_unwrapped + bed_minute / 60.0
    bed_score = float(np.exp(-0.5 * ((bed_time - 22.5) / 2.0) ** 2))

    # Wake time score — peak at 7h
    wake_time  = wake_hour + wake_minute / 60.0
    wake_score = float(np.exp(-0.5 * ((wake_time - 7.0) / 2.0) ** 2))

    # Steps score
    steps_score = float(np.clip(1.0 - ((total_steps - 8500) / 6000.0) ** 2, 0.0, 1.0))

    # Activity score
    vigorous      = float(is_strength or is_cardio)
    any_activity  = 1.0 - float(is_no_activity)
    activity_score = 0.3 * any_activity + 0.7 * vigorous

    # Sleep window
    if wake_time < bed_time:
        sleep_hours = (24.0 - bed_time) + wake_time
    else:
        sleep_hours = wake_time + (24.0 - bed_time)
    sleep_hours = float(np.clip(sleep_hours, 0.0, 12.0))
    sleep_score = float(np.clip(1.0 - ((sleep_hours - 8.0) / 3.0) ** 2, 0.0, 1.0))
    total_sleep_seconds = sleep_hours * 3600.0

    # Biometric mappings
    avg_heart_rate     = 80.0 - 15.0 * steps_score - 8.0 * vigorous
    resting_heart_rate = 65.0 - 12.0 * steps_score - 5.0 * bed_score
    avg_resp           = 14.5 - 0.5 * any_activity
    avg_stress         = 55.0 - 20.0 * bed_score - 15.0 * activity_score - 10.0 * steps_score
    body_battery       = 20.0 + 35.0 * bed_score + 25.0 * sleep_score + 20.0 * steps_score

    asleep = total_sleep_seconds * 0.92
    deep_ratio = np.clip(0.10 + 0.15 * vigorous * bed_score, 0.0, 0.30)
    rem_ratio  = np.clip(0.12 + 0.14 * wake_score * bed_score, 0.0, 0.28)
    deep_sleep = asleep * float(deep_ratio)
    rem_sleep  = asleep * float(rem_ratio)
    awake      = float(np.clip(total_sleep_seconds * (0.15 - 0.10 * bed_score), 0.0, None))

    restless       = float(np.clip(75.0 - 30.0 * activity_score - 25.0 * bed_score - 15.0 * steps_score, 3.0, 80.0))
    avg_sleep_stress = float(np.clip(45.0 - 20.0 * bed_score - 12.0 * activity_score - 8.0 * steps_score, 3.0, 55.0))
    sleep_rhr      = resting_heart_rate - 3.0

    return np.array([
        avg_heart_rate, resting_heart_rate, avg_resp, avg_stress, body_battery,
        total_sleep_seconds, deep_sleep, rem_sleep, awake,
        restless, avg_sleep_stress, sleep_rhr,
    ], dtype=np.float32)


# ------------------------------------------------------------------
# Random action sampler
# ------------------------------------------------------------------

def _sample_random_action(rng: np.random.Generator) -> np.ndarray:
    """
    Sample a physiologically plausible random action.
    Covers a wider range than Lukas's actual behaviour to train action sensitivity.
    """
    steps     = rng.uniform(0, 25000)
    act_idx   = rng.integers(0, 6)  # 0=Strength..5=NoActivity
    flags     = np.zeros(6, dtype=np.float32)
    flags[act_idx] = 1.0

    # Bed time: 19:00–02:00 (represented as 19–26h, clipped later)
    bed_h_raw = rng.uniform(19.0, 26.0)
    bed_h     = bed_h_raw % 24
    bed_m     = rng.uniform(0, 59)
    wake_h    = rng.uniform(5.0, 11.0)
    wake_m    = rng.uniform(0, 59)

    return np.array([
        steps,
        flags[0], flags[1], flags[2], flags[3], flags[4], flags[5],
        bed_h, bed_m, wake_h, wake_m,
    ], dtype=np.float32)


# ------------------------------------------------------------------
# Synthetic dataset generation
# ------------------------------------------------------------------

def generate_synthetic_data(
    processed_features: list,
    processor: DataProcessor,
    n_variants: int = 5,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    For each real day, generate n_variants synthetic days by:
        1. Sampling a random action
        2. Computing analytical biometrics via DeterministicEnv function
        3. Assembling full 23-feature vector (synthetic_action + analytical_biometrics)
        4. Scaling inputs and targets

    Also includes the original real days as additional training samples.

    Args:
        processed_features: Full list of processed feature dicts.
        processor:          Fitted DataProcessor (scalers must be fitted already).
        n_variants:         Synthetic action variants per real day.
        seed:               RNG seed for reproducibility.

    Returns:
        X_scaled: (N_total, 23) scaled full-day feature vectors
        y_scaled: (N_total, 12) scaled model-feature targets
    """
    rng = np.random.default_rng(seed)

    X_rows, y_rows = [], []

    for day_dict in processed_features:
        real_flat = processor.flatten_features_for_day(day_dict)  # (23,) unscaled

        # --- Real day as training sample (use real biometrics as target) ---
        real_model_features = real_flat[11:]  # (12,) unscaled
        X_rows.append(real_flat)
        y_rows.append(real_model_features)

        # --- Synthetic variants ---
        for _ in range(n_variants):
            action = _sample_random_action(rng)              # (11,) unscaled
            bio    = _analytical_biometrics(action)          # (12,) unscaled

            # Full 23-feature vector: synthetic action + analytical biometrics
            synthetic_flat = np.concatenate([action, bio])   # (23,)
            X_rows.append(synthetic_flat)
            y_rows.append(bio)

    X_raw = np.array(X_rows, dtype=np.float32)  # (N, 23)
    y_raw = np.array(y_rows, dtype=np.float32)  # (N, 12)

    print(f"Generated {len(X_rows)} total samples "
          f"({len(processed_features)} real + "
          f"{len(processed_features) * n_variants} synthetic)")

    # Scale using already-fitted processor scalers
    X_scaled = processor.transform_X(
        X_raw.reshape(len(X_raw), 1, 23)
    ).reshape(len(X_raw), 23)

    y_scaled = processor.transform_y(y_raw)

    return X_scaled, y_scaled


# ------------------------------------------------------------------
# Main training pipeline
# ------------------------------------------------------------------

def train_edmd(processor: DataProcessor = None)-> tuple[EDMDModel, DataProcessor]:
    """
    Full EDMD training pipeline:
    1. Fetch all available cached data (expand beyond LSTM training window)
    2. Fit DataProcessor scalers
    3. Generate hybrid dataset (real + synthetic)
    4. Train/val split + fit EDMDModel
    5. Evaluate + sensitivity report
    6. Save model
    """
    print("=" * 60)
    print("  ChronoOpt — EDMD Training Pipeline")
    print("=" * 60)

    # --- 1. Fetch data — use all available cache ---
    # Use a larger window than LSTM training to maximise real distribution coverage
    print(f"\n[1/5] Fetching data...")
    training_end_date = date.fromisoformat(config.TRAINING_END_DATE)

    # Fetch as many days as available (will be limited by cache)
    raw_data = get_historical_metrics(
        config.NUM_DAYS_TO_FETCH_RAW,
        end_date=training_end_date,
    )
    if not raw_data:
        raise RuntimeError("No historical data fetched. Check cache.")
    print(f"      Fetched {len(raw_data)} days.")

    # --- 2. Feature engineering + fit processor ---
    print("\n[2/5] Engineering features and fitting processor...")
    processed_features = [extract_daily_features(d) for d in raw_data]

    if processor is None:
        print("No processor provided — fitting from scratch...")
        processor = DataProcessor()
        processor.prepare_data_for_training(processed_features, config.NUM_DAYS_FOR_STATE)
    else:
        print("Using provided processor — skipping fit.")

    # --- 3. Generate hybrid dataset ---
    print(f"\n[3/5] Generating synthetic dataset "
          f"(n_variants={config.EDMD_SYNTHETIC_VARIANTS} per day)...")
    X_scaled, y_scaled = generate_synthetic_data(
        processed_features,
        processor,
        n_variants=config.EDMD_SYNTHETIC_VARIANTS,
    )
    print(f"      Dataset shape: X={X_scaled.shape}, y={y_scaled.shape}")

    # --- 4. Train/val split + fit ---
    print("\n[4/5] Fitting EDMD...")
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y_scaled,
        test_size=0.15,
        shuffle=True,
        random_state=42,
    )

    edmd = EDMDModel(
        degree=config.EDMD_DEGREE,
        alpha=config.EDMD_ALPHA,
    )
    edmd.fit(X_train, y_train)

    # --- 5. Evaluate ---
    print("\n[5/5] Evaluating...")
    edmd.evaluate(X_val, y_val, processor.model_feature_keys)
    edmd.print_sensitivity(processor.agent_feature_keys, processor.model_feature_keys)

    # --- 6. Save ---
    edmd.save(config.EDMD_MODEL_SAVE_PATH)
    print(f"\nEDMD training complete.")
    plot_edmd_results(edmd, X_val, y_val,processor)
    return edmd, processor


import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def plot_edmd_results(edmd: EDMDModel,
                      X_val: np.ndarray,
                      y_val: np.ndarray,
                      processor: DataProcessor,
                      save_path: str = "src/models/saved_models/edmd_diagnostics.png"):
    """
    Three-panel diagnostic plot:
        1. Predicted vs actual for all 12 features (scatter)
        2. Residual distributions per feature (boxplot)
        3. Action feature total influence (bar chart from K matrix)
    """
    pred_scaled = np.array([edmd.predict(x) for x in X_val])
    
    # Inverse transform for human-readable scale
    pred_real = processor.inverse_transform_y(pred_scaled)
    true_real = processor.inverse_transform_y(y_val)
    
    keys = processor.model_feature_keys
    n    = len(keys)

    fig = plt.figure(figsize=(20, 24))
    gs  = gridspec.GridSpec(3, 1, hspace=0.4)

    # --- Panel 1: Predicted vs Actual (4×3 subgrid) ---
    gs1 = gridspec.GridSpecFromSubplotSpec(4, 3, subplot_spec=gs[0], hspace=0.5, wspace=0.35)
    for i, key in enumerate(keys):
        ax = fig.add_subplot(gs1[i // 3, i % 3])
        ax.scatter(true_real[:, i], pred_real[:, i],
                   alpha=0.4, s=12, color='steelblue')
        mn = min(true_real[:, i].min(), pred_real[:, i].min())
        mx = max(true_real[:, i].max(), pred_real[:, i].max())
        ax.plot([mn, mx], [mn, mx], 'r--', lw=1)
        mae = float(np.mean(np.abs(pred_real[:, i] - true_real[:, i])))
        short = key.replace('_seconds', 's').replace('_', ' ')
        ax.set_title(f"{short}\nMAE={mae:.1f}", fontsize=8)
        ax.set_xlabel("Actual", fontsize=7)
        ax.set_ylabel("Predicted", fontsize=7)
        ax.tick_params(labelsize=7)
    fig.text(0.5, 0.98, "EDMD — Predicted vs Actual (unscaled)",
             ha='center', fontsize=13, fontweight='bold')

    # --- Panel 2: Residuals boxplot ---
    ax2 = fig.add_subplot(gs[1])
    residuals = pred_real - true_real  # (N, 12)
    # Normalize by feature std for comparability
    stds = true_real.std(axis=0) + 1e-8
    norm_res = residuals / stds
    bp = ax2.boxplot(norm_res, patch_artist=True, notch=False,
                     medianprops=dict(color='red', lw=2))
    for patch in bp['boxes']:
        patch.set_facecolor('lightsteelblue')
        patch.set_alpha(0.7)
    ax2.axhline(0, color='gray', lw=1, ls='--')
    ax2.set_xticklabels(
        [k.replace('_seconds', 's').replace('_', '\n') for k in keys],
        fontsize=7
    )
    ax2.set_ylabel("Normalised residual (pred - actual) / std", fontsize=9)
    ax2.set_title("Residual Distribution per Feature (normalised)", fontsize=11)
    ax2.grid(axis='y', alpha=0.3)

    # --- Panel 3: Action feature influence bar chart ---
    ax3 = fig.add_subplot(gs[2])
    W = edmd.regressor.coef_          # (12, n_observables)
    linear_action_w = np.abs(W[:, :11])  # (12, 11) — linear action cols only
    total_influence  = linear_action_w.sum(axis=0)  # (11,)
    agent_keys = processor.agent_feature_keys
    colors = ['#e74c3c' if 'bed' in k or 'wake' in k
              else '#2ecc71' if 'step' in k
              else '#3498db' for k in agent_keys]
    bars = ax3.bar(agent_keys, total_influence, color=colors, edgecolor='white', lw=0.5)
    ax3.set_xticklabels(
        [k.replace('_', '\n').replace('activity\n', '') for k in agent_keys],
        fontsize=8
    )
    ax3.set_ylabel("Total |weight| across all 12 predictions", fontsize=9)
    ax3.set_title("EDMD — Action Feature Influence (linear K matrix terms)\n"
                  "Red=sleep timing  Green=steps  Blue=activity", fontsize=11)
    ax3.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, total_influence):
        ax3.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.0002,
                 f"{val:.4f}", ha='center', fontsize=7)

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Diagnostic plots saved to {save_path}")
    plt.close()

if __name__ == "__main__":
    train_edmd()
