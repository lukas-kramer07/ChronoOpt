# src/api/inference.py
# Core inference logic for ChronoOpt.
#
# This module owns two responsibilities:
#   1. Loading and holding all ML artifacts at startup (LSTM, processor, policy)
#   2. Generating a daily recommendation from real Garmin state
#


import numpy as np
import torch
from datetime import date, timedelta
from typing import Optional

from src import config
from src.models.prediction_model import PredictionModel
from src.models.data_processor import DataProcessor
from src.rl_agent.policy_network import PolicyNetwork
from src.rl_agent.rl_environment import ChronoOptEnv
from src.rl_agent.deterministic_environment import DeterministicEnv
from src.data_ingestion.garmin_parser import get_historical_metrics
from src.features.feature_engineer import extract_daily_features
from src.rl_agent.train_agent import build_fitted_processor, build_initial_state

# ---------------------------------------------------------------------------
# Global model state — populated once at startup via load_all_models()
# ---------------------------------------------------------------------------

_lstm: Optional[PredictionModel] = None
_processor: Optional[DataProcessor] = None
_policy: Optional[PolicyNetwork] = None
_device: Optional[torch.device] = None
_policy_source: str = "not_loaded"
_last_fetch_date: Optional[str] = None
_garmin_days_available: int = 0


def load_all_models() -> None:
    """
    Loads the LSTM, fits the DataProcessor, and loads the trained policy.
    Called once at FastAPI startup via the lifespan context manager.
    Expensive (~5-10s from cache) but amortised over the entire app session.
    """
    global _lstm, _processor, _policy, _device, _policy_source

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[inference] Device: {_device}")

    # --- LSTM ---
    print("[inference] Loading LSTM...")
    _lstm = PredictionModel.load(config.LSTM_MODEL_SAVE_PATH, _device)
    print("[inference] LSTM loaded.")

    # --- DataProcessor: replay training pipeline from cache to refit scalers ---
    # This is intentionally the same pipeline used during LSTM training so the
    # scaler parameters are identical. Reads from local JSON cache, no API calls.
    print("[inference] Fitting DataProcessor from cache (this takes ~5s)...")
    _processor, _ = build_fitted_processor()
    print("[inference] DataProcessor fitted.")

    # --- Policy: try trained policy, fall back to deterministic env if missing ---
    print(f"[inference] Loading policy from {config.POLICY_SAVE_PATH}...")
    try:
        _policy = PolicyNetwork.load(config.POLICY_SAVE_PATH, _device)
        _policy_source = "trained_policy"
        print("[inference] Trained policy loaded.")
    except (FileNotFoundError, KeyError) as e:
        print(f"[inference] Policy not found ({e}). Will use DeterministicEnv fallback.")
        _policy = None
        _policy_source = "deterministic_fallback"


def is_ready() -> dict:
    """Returns a health dict reporting load status of each component."""
    return {
        "lstm_loaded": _lstm is not None,
        "policy_loaded": _policy is not None,
        "processor_fitted": _processor is not None and _processor._is_scaler_fitted,
        "last_garmin_fetch_date": _last_fetch_date,
        "garmin_days_available": _garmin_days_available,
        "policy_source": _policy_source,
    }


def _build_real_state() -> tuple[np.ndarray, list, int]:
    """
    Fetches the last NUM_DAYS_FOR_STATE days of real Garmin data and builds
    an unscaled (seq_len, 23) state array.

    Returns:
        state_array: np.ndarray of shape (seq_len, 23), unscaled
        processed_features: list of feature dicts (for baseline extraction)
        days_used: how many real days ended up in the state
    """
    global _last_fetch_date, _garmin_days_available

    raw = get_historical_metrics(config.NUM_DAYS_FOR_STATE + 5)  # fetch extra for safety
    processed = [extract_daily_features(d) for d in raw]

    # Use the most recent NUM_DAYS_FOR_STATE days
    last_n = processed[-config.NUM_DAYS_FOR_STATE:]
    state_array = np.array(
        [_processor.flatten_features_for_day(d) for d in last_n],
        dtype=np.float32
    )  # (seq_len, 23), unscaled

    _garmin_days_available = len(processed)
    _last_fetch_date = last_n[-1]["date"] if last_n else None

    return state_array, last_n, len(last_n)


def _scale_state(state_array: np.ndarray) -> np.ndarray:
    """Scales a (seq_len, 23) unscaled state to the LSTM's input space."""
    seq_len = state_array.shape[0]
    return _processor.transform_X(
        state_array.reshape(1, seq_len, 23)
    )[0]  # (seq_len, 23), scaled


def _predict_sleep_score(env: ChronoOptEnv, state_array: np.ndarray, action: np.ndarray) -> float:
    """
    Runs one step of the environment with the given action and returns the
    predicted sleep score. We manually set env.history to avoid perturbation.
    """
    env.history = state_array.tolist()
    _, reward, _, _, _ = env.step(action)
    return round(float(reward), 2)


def _extract_baseline_action(state_array: np.ndarray) -> np.ndarray:
    """
    Extracts yesterday's agent-controlled features (indices 0–10) from the
    last row of the state array as the baseline 'do what you did yesterday' action.
    """
    return state_array[-1, :11].astype(np.float32)


def get_recommendation() -> dict:
    """
    Main inference function. Builds today's recommendation from real Garmin
    state, computes predicted sleep score for recommendation vs baseline,
    and returns a structured result dict matching RecommendationResponse.
    """
    if _lstm is None or _processor is None:
        raise RuntimeError("Models not loaded. Call load_all_models() first.")

    # --- 1. Build real state from Garmin cache ---
    state_array, processed_features, days_used = _build_real_state()
    scaled_obs = _scale_state(state_array)

    # --- 2. Create env (needed for sleep score prediction via LSTM) ---
    # We use ChronoOptEnv or DeterministicEnv depending on policy source.
    # Either way, we override history manually — no reset() perturbation.
    EnvClass = ChronoOptEnv if _policy_source == "trained_policy" else DeterministicEnv
    env = EnvClass(
        initial_state_data=state_array,
        model=_lstm,
        processor=_processor,
        device=_device,
    )

    # --- 3. Get recommended action ---
    if _policy is not None:
        _policy.eval()
        recommended_action, _, _, _, _ = _policy.get_action(
            scaled_obs, _device, deterministic=True
        )
    else:
        # Fallback: use DeterministicEnv's optimal analytical action
        recommended_action = np.array(
            [9000, 1, 0, 0, 0, 0, 0, 22, 30, 7, 0], dtype=np.float32
        )

    # --- 4. Compute predicted sleep scores ---
    # We need two separate env instances so step() doesn't bleed state between them.
    env_rec = EnvClass(initial_state_data=state_array, model=_lstm,
                       processor=_processor, device=_device)
    env_base = EnvClass(initial_state_data=state_array, model=_lstm,
                        processor=_processor, device=_device)

    recommended_score = _predict_sleep_score(env_rec, state_array, recommended_action)
    baseline_action = _extract_baseline_action(state_array)
    baseline_score = _predict_sleep_score(env_base, state_array, baseline_action)

    # --- 5. Decode action into human-readable fields ---
    ACTIVITY_NAMES = ['Strength', 'Cardio', 'Yoga', 'Stretching', 'OtherActivity', 'NoActivity']
    steps      = int(round(recommended_action[0]))
    activity   = ACTIVITY_NAMES[int(np.argmax(recommended_action[1:7]))]
    bed_hour   = int(recommended_action[7])
    bed_min    = int(recommended_action[8])
    wake_hour  = int(recommended_action[9])
    wake_min   = int(recommended_action[10])

    today = date.today().isoformat()

    return {
        "date": today,
        "recommendation": {
            "target_steps": steps,
            "activity_type": activity,
            "bed_hour": bed_hour,
            "bed_minute": bed_min,
            "wake_hour": wake_hour,
            "wake_minute": wake_min,
        },
        "predicted_scores": {
            "recommended": recommended_score,
            "baseline": baseline_score,
            "delta": round(recommended_score - baseline_score, 2),
        },
        "state_days_used": days_used,
        "policy_source": _policy_source,
    }
