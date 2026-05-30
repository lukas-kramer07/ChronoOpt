# src/api/inference.py
# Core inference logic for ChronoOpt.
#
# This module owns two responsibilities:
#   1. Loading and holding all ML artifacts at startup (LSTM, processor, policy)
#   2. Generating a daily recommendation from real Garmin state
#


from __future__ import annotations
 
import numpy as np
import torch
from dataclasses import dataclass
from datetime import date
from typing import Optional
 
from src import config
from src.models.prediction_model import PredictionModel
from src.models.data_processor import DataProcessor
from src.rl_agent.policy_network import PolicyNetwork
from src.rl_agent.rl_environment import ChronoOptEnv
from src.data_ingestion.garmin_parser import get_historical_metrics
from src.features.feature_engineer import extract_daily_features
from src.rl_agent.train_agent import build_fitted_processor
 
 
# ---------------------------------------------------------------------------
# ModelBundle — the single object that travels through the app
# ---------------------------------------------------------------------------
 
@dataclass
class ModelBundle:
    """
    All ML artifacts needed for inference, loaded once at startup and stored
    on app.state. Passed into endpoint functions via Depends(get_ml_models).
    """
    lstm: PredictionModel
    processor: DataProcessor
    policy: Optional[PolicyNetwork]
    policy_source: str   # "trained_policy" | "deterministic_fallback"
    device: torch.device
 
    @property
    def is_healthy(self) -> bool:
        return self.lstm is not None and self.processor._is_scaler_fitted
 
 
# ---------------------------------------------------------------------------
# Startup: load everything, return a bundle
# ---------------------------------------------------------------------------
 
def load_all_models() -> ModelBundle:
    """
    Loads the LSTM, fits the DataProcessor, and loads the trained policy.
    Called once inside the FastAPI lifespan context; result stored on app.state.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[inference] Device: {device}")
 
    # --- LSTM ---
    print("[inference] Loading LSTM...")
    lstm = PredictionModel.load(config.LSTM_MODEL_SAVE_PATH, device)
    print("[inference] LSTM loaded.")
 
    # --- DataProcessor ---
    # Replays the training pipeline from cache to refit the StandardScalers
    # with identical parameters to those used during LSTM training.
    print("[inference] Fitting DataProcessor from cache (~5s)...")
    processor, _ = build_fitted_processor()
    print("[inference] DataProcessor fitted.")
 
    # --- Policy ---
    print(f"[inference] Loading policy from {config.POLICY_SAVE_PATH}...")
    try:
        policy = PolicyNetwork.load(config.POLICY_SAVE_PATH, device)
        policy_source = "trained_policy"
        print("[inference] Trained policy loaded.")
    except (FileNotFoundError, KeyError) as e:
        print(f"[inference] Policy not found ({e}). Falling back to DeterministicEnv.")
        policy = None
        policy_source = "deterministic_fallback"
 
    return ModelBundle(
        lstm=lstm,
        processor=processor,
        policy=policy,
        policy_source=policy_source,
        device=device,
    )
 
 
# ---------------------------------------------------------------------------
# Inference helpers — all take ModelBundle explicitly, no hidden dependencies
# ---------------------------------------------------------------------------
 
def _build_real_state(models: ModelBundle) -> tuple[np.ndarray, list, int]:
    """
    Fetches the last NUM_DAYS_FOR_STATE days of real Garmin data and builds
    an unscaled (seq_len, 23) state array.

    Returns:
        state_array:        (seq_len, 23) unscaled
        processed_features: list of feature dicts
        days_used:          number of real days in the state
    """
    raw = get_historical_metrics(config.NUM_DAYS_FOR_STATE + 5)
    processed = [extract_daily_features(d) for d in raw]
 
    last_n = processed[-config.NUM_DAYS_FOR_STATE:]
    state_array = np.array(
        [models.processor.flatten_features_for_day(d) for d in last_n],
        dtype=np.float32,
    )
    return state_array, last_n, len(last_n)
 
 
def _scale_state(state_array: np.ndarray, models: ModelBundle) -> np.ndarray:
    """Scales a (seq_len, 23) unscaled state to the LSTM's input space."""
    seq_len = state_array.shape[0]
    return models.processor.transform_X(
        state_array.reshape(1, seq_len, 23)
    )[0]
 
 
def _score_policy_rollout(state_array, scaled_obs, models, n_steps=7) -> float:
    """Policy rollout — policy re-observes and re-acts at each step."""
    env = ChronoOptEnv(initial_state_data=state_array, model=models.lstm,
                       processor=models.processor, device=models.device)
    env.history = state_array.tolist()
    obs = scaled_obs.copy()
    rewards = []
    for _ in range(n_steps):
        action, _, _, _, _ = models.policy.get_action(obs, models.device, deterministic=True)
        obs, reward, _, _, _ = env.step(action)
        rewards.append(reward)
    return round(float(np.mean(rewards)), 2)

def _score_fixed_action(state_array, action, models, n_steps=3) -> float:
    """Baseline — repeat yesterday's action for a short horizon."""
    env = ChronoOptEnv(initial_state_data=state_array, model=models.lstm,
                       processor=models.processor, device=models.device)
    env.history = state_array.tolist()
    env.step(action)  # warmup — discard
    rewards = []
    for _ in range(n_steps):
        _, reward, _, _, _ = env.step(action)
        rewards.append(reward)
    return round(float(np.mean(rewards)), 2)
 
 
# ---------------------------------------------------------------------------
# Main inference function
# ---------------------------------------------------------------------------
 
ACTIVITY_NAMES = ['Strength', 'Cardio', 'Yoga', 'Stretching', 'OtherActivity', 'NoActivity']
 
 
def get_recommendation(models: ModelBundle) -> dict:
    """
    Builds today's recommendation from real Garmin state.
 
    Returns a dict matching the RecommendationResponse Pydantic model shape.
    """
    # 1. Build real state
    state_array, _, days_used = _build_real_state(models)
    scaled_obs = _scale_state(state_array, models)
 
    # 2. Get recommended action from policy (or hardcoded fallback)
    if models.policy is not None:
        models.policy.eval()
        recommended_action, _, _, _, _ = models.policy.get_action(
            scaled_obs, models.device, deterministic=True
        )
    else:
        recommended_action = np.array(
            [900000, 1, 0, 0, 0, 0, 0, 22, 30, 7, 0], dtype=np.float32
        )
 
    # 3. Predicted sleep scores: recommended vs baseline (repeat yesterday)
    days = 0
    model =""
    baseline_action = state_array[-1, :11].astype(np.float32)
    if models.policy is not None:
        recommended_score = _score_policy_rollout(state_array, scaled_obs, models,n_steps=7)
        days = 7
        model = "rollout"
    else:
        recommended_score = _score_fixed_action(state_array, recommended_action, models,n_steps=3)
        days = 3
        model = "no-rollout"
    baseline_score = _score_fixed_action(state_array, baseline_action, models)
 
    # 4. Decode action → human-readable fields
    steps     = int(round(recommended_action[0]))
    activity  = ACTIVITY_NAMES[int(np.argmax(recommended_action[1:7]))]
    bed_hour  = int(recommended_action[7])
    bed_min   = int(recommended_action[8])
    wake_hour = int(recommended_action[9])
    wake_min  = int(recommended_action[10])
 
    return {
        "date": date.today().isoformat(),
        "recommendation": {
            "target_steps":  steps,
            "activity_type": activity,
            "bed_hour":      bed_hour,
            "bed_minute":    bed_min,
            "wake_hour":     wake_hour,
            "wake_minute":   wake_min,
        },
        "predicted_scores": {
            "recommended": recommended_score,
            "baseline":    baseline_score,
            "delta":       round(recommended_score - baseline_score, 2),
            "days":        days,
            "model":       model,
        },
        "state_days_used": days_used,
        "policy_source":   models.policy_source,
    }
