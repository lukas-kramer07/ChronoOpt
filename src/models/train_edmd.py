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
from src.models.plot_utils import plot_model_diagnostics, plot_next_day_prediction
from src.features.utils import generate_synthetic_data
from datetime import datetime, timedelta

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
        # processor.prepare_data_for_training(processed_features, config.NUM_DAYS_FOR_STATE)
    # else:
        # print("Using provided processor — skipping fit.")

    # --- 3. Generate hybrid dataset ---
    print("\n[3/5] Preparing data via processor (seq_len=1)...")
    X_seq, y_scaled = processor.prepare_data_for_training(
        processed_features,
        num_days_in_state=1,  # single-day input for EDMD
    )
    # X_seq shape: (N, 1, 23) → flatten to (N, 23)
    X_scaled = X_seq.reshape(len(X_seq), 23)
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
    W = edmd.regressor.coef_
    plot_model_diagnostics(
        pred_unscaled=processor.inverse_transform_y(
            np.array([edmd.predict(x) for x in X_val])
        ),
        true_unscaled=processor.inverse_transform_y(y_val),
        model_feature_keys=processor.model_feature_keys,
        agent_feature_keys=processor.agent_feature_keys,
        linear_action_weights=np.abs(W[:, :11]),
        model_name="EDMD",
        save_path="src/models/saved_models/edmd_diagnostics.png",
        show=False,
    )
    for i in range(1,10):
        last_x_scaled = X_scaled[-i]                          # (23,)
        pred_scaled   = edmd.predict(last_x_scaled)            # (12,)

        # Reconstruct into feature dict for plotting
        last_date_str = processed_features[-1]['date']
        pred_date_str = (
            datetime.strptime(last_date_str, "%Y-%m-%d") + timedelta(days=1)
        ).strftime("%Y-%m-%d")

        predicted_dict = processor.reconstruct_features_from_flat(
            pred_scaled, date_str=pred_date_str
        )

        plot_next_day_prediction(
            historical_features_dicts=processed_features,
            predicted_features_dict=predicted_dict,
            num_days_for_state=config.NUM_DAYS_FOR_STATE,
            show=True,
        )

    # --- 6. Save ---
    edmd.save(config.EDMD_MODEL_SAVE_PATH)
    print(f"\nEDMD training complete.")
    return edmd, processor

if __name__ == "__main__":
    train_edmd()