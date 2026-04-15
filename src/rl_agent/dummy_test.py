# dumm_test.py
# Verifies that ChronoOptEnv initialises correctly and step() runs end to end.

import numpy as np
import torch
from datetime import date, timedelta

from src.data_ingestion.garmin_parser import get_historical_metrics
from src.features.feature_engineer import extract_daily_features
from src.models.data_processor import DataProcessor
from src.models.prediction_model import PredictionModel
from src.rl_agent.rl_environment import ChronoOptEnv
from src import config

# --- Config ---
NUM_DAYS = config.NUM_DAYS_FOR_STATE + 50  # A few extra days for safety
TRAINING_END_DATE = date(2025, 12, 1)     # Last reliable date
N_STEPS = 35                               # Run past max_steps to verify done flag

def build_dummy_action() -> np.ndarray:
    """
    Returns a plausible unscaled action vector (11 features).
    Order: [total_steps, Strength, Cardio, Yoga, Stretching, OtherActivity, NoActivity,
            bed_hour, bed_minute, wake_hour, wake_minute]
    """
    return np.array([
        8000,   # total_steps
        0,      # activity_Strength
        1,      # activity_Cardio
        0,      # activity_Yoga
        0,      # activity_Stretching
        0,      # activity_OtherActivity
        0,      # activity_NoActivity
        22,     # bed_hour (10 PM)
        0,      # bed_minute
        7,      # wake_hour (7 AM)
        0,      # wake_minute
    ], dtype=np.float32)


def main():
    print("=" * 60)
    print("  ChronoOpt RL Environment — Smoke Test")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # --- 1. Fetch historical data ---
    print(f"\n[1/5] Fetching {NUM_DAYS} days of historical data...")
    raw_data = get_historical_metrics(NUM_DAYS, end_date=TRAINING_END_DATE)
    assert len(raw_data) >= config.NUM_DAYS_FOR_STATE + 1, \
        f"Not enough data: got {len(raw_data)} days, need {config.NUM_DAYS_FOR_STATE + 1}"
    print(f"      Fetched {len(raw_data)} days. OK")

    # --- 2. Feature engineering ---
    print("\n[2/5] Engineering features...")
    processed = [extract_daily_features(d) for d in raw_data]
    print(f"      Processed {len(processed)} days. OK")

    # --- 3. DataProcessor + PredictionModel ---
    print("\n[3/5] Initialising DataProcessor and PredictionModel...")
    processor = DataProcessor()
    X, y = processor.prepare_data_for_training(processed, config.NUM_DAYS_FOR_STATE)
    assert X.shape[0] > 0, "No training samples produced"
    print(f"      X shape: {X.shape}, y shape: {y.shape}. OK")

    model = PredictionModel(
        input_size=processor.input_size,
        hidden_size=config.MODEL_HYPERPARAMETERS['hidden_size'],
        output_size=processor.output_size,
        num_layers=config.MODEL_HYPERPARAMETERS['num_layers']
    )
    print(f"      Model initialised (untrained — predictions will be noisy). OK")

    # --- 4. Initialise environment ---
    print("\n[4/5] Initialising ChronoOptEnv...")

    # Build initial state from last NUM_DAYS_FOR_STATE days of processed features
    last_n_days = processed[-config.NUM_DAYS_FOR_STATE:]
    initial_state_flat = np.array(
        [processor.flatten_features_for_day(d) for d in last_n_days],
        dtype=np.float32
    )  # (NUM_DAYS_FOR_STATE, 23) — unscaled

    env = ChronoOptEnv(
        initial_state_data=initial_state_flat,
        model=model,
        processor=processor,
        device=device
    )
    print("      Environment initialised. OK")

    # --- 5. Run smoke test loop ---
    print(f"\n[5/5] Running {N_STEPS} steps...")
    print(f"      Expected done=True at step {env.max_steps}\n")

    observation, info = env.reset()
    assert observation.shape == (config.NUM_DAYS_FOR_STATE, processor.input_size), \
        f"Unexpected observation shape after reset: {observation.shape}"
    print(f"      reset() observation shape: {observation.shape}. OK")

    action = build_dummy_action()

    for step in range(1, N_STEPS + 1):
        obs, reward, done, truncated, info = env.step(action)

        # Validate types and shapes
        assert isinstance(reward, float), f"Reward is not float: {type(reward)}"
        assert 0.0 <= reward <= 100.0, f"Reward out of bounds: {reward}"
        assert obs.shape == (config.NUM_DAYS_FOR_STATE, processor.input_size), \
            f"Unexpected obs shape at step {step}: {obs.shape}"
        assert isinstance(done, bool), f"done is not bool: {type(done)}"

        print(f"      Step {step:02d} | reward: {reward:.2f} | done: {done} | obs shape: {obs.shape}")

        if done:
            print(f"\n      done=True triggered at step {step}. OK")
            assert step == env.max_steps, \
                f"done triggered at wrong step: {step} (expected {env.max_steps})"
            break

    print("\n" + "=" * 60)
    print("  SMOKE TEST PASSED — rl_environment loop is functional.")
    print("=" * 60)


if __name__ == "__main__":
    main()