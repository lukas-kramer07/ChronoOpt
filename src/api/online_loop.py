# src/api/online_loop.py
# Nightly background task registered via APScheduler in FastAPI lifespan.
#
# What it does each night:
#   1. Fetch yesterday's real Garmin data
#   2. Compute actual sleep score reward
#   3. Build real transition (state, recommendation, reward)
#   4. Single online policy gradient step
#   5. Every EDMD_REFIT_INTERVAL days: refit EDMD on all real data
#   6. Persist updated model weights



import logging
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from datetime import date, timedelta, datetime
from typing import Optional
from src.api import database

from src import config
from src.data_ingestion.garmin_parser import get_historical_metrics
from src.features.feature_engineer import extract_daily_features
from src.features.utils import calculate_sleep_score_proxy
from src.models.edmd_model import EDMDModel


logger = logging.getLogger(__name__)

# How often to refit EDMD (in days)
EDMD_REFIT_INTERVAL = 7


# ------------------------------------------------------------------
# Core update functions
# ------------------------------------------------------------------

def compute_real_reward(garmin_day: dict) -> float:
    """
    Computes the actual sleep score reward from a real Garmin day dict.
    Falls back to 0.0 if sleep data is missing.

    Args:
        garmin_day: Raw Garmin metrics dict for one day.

    Returns:
        float: Sleep score proxy in [0, 100], power-law sharpened (k=2).
    """
    features = extract_daily_features(garmin_day)
    sleep_metrics = features.get('sleep_metrics', {})
    raw_score = calculate_sleep_score_proxy(sleep_metrics)
    # Same power law as training environment
    return float(((raw_score / 100.0) ** 2) * 100.0)


def build_real_state(processed_features: list,
                     processor,
                     seq_len: int = 10) -> Optional[np.ndarray]:
    """
    Builds a scaled (seq_len, 23) observation from the last seq_len real days.

    Args:
        processed_features: List of processed feature dicts, oldest→newest.
        processor:          Fitted DataProcessor.
        seq_len:            Number of history days (must match policy input).

    Returns:
        np.ndarray: Scaled observation of shape (seq_len, 23), or None if
                    insufficient data.
    """
    if len(processed_features) < seq_len:
        logger.warning(f"Only {len(processed_features)} days available, need {seq_len}.")
        return None

    last_n = processed_features[-seq_len:]
    state_unscaled = np.array(
        [processor.flatten_features_for_day(d) for d in last_n],
        dtype=np.float32,
    )  # (seq_len, 23)

    state_scaled = processor.transform_X(
        state_unscaled.reshape(1, seq_len, 23)
    )[0]  # (seq_len, 23)

    return state_scaled


def update_policy(
    policy,
    device: torch.device,
    state: np.ndarray,
    reward: float,
    lr: float = 1e-5,
) -> dict:
    """
    Single online policy gradient step (A2C-style).

    Re-runs the policy on the current state, computes advantage from the
    real reward, and does one gradient update.

    Uses a very small LR (default 1e-5 vs training 5e-5) to avoid
    overwriting the offline-trained prior too aggressively.

    Args:
        policy:  PolicyNetwork instance (will be mutated).
        device:  Torch device.
        state:   Scaled real observation, shape (seq_len, 23).
        reward:  Actual sleep score reward from Garmin data.
        lr:      Online learning rate.

    Returns:
        dict: {'policy_loss', 'value_loss', 'advantage', 'value_estimate'}
    """
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    policy.train()
    x = torch.tensor(state, dtype=torch.float32).flatten().unsqueeze(0).to(device)

    continuous_out, activity_probs, value = policy.forward(x)
    value = value.squeeze()

    # Sample action under current policy to get log_prob
    std = torch.full_like(continuous_out, 0.5)
    dist_cont = torch.distributions.Normal(continuous_out, std)
    cont_sample = dist_cont.sample()
    log_prob_cont = dist_cont.log_prob(cont_sample).sum(dim=-1)

    dist_act = torch.distributions.Categorical(probs=activity_probs)
    act_idx = dist_act.sample()
    log_prob_act = dist_act.log_prob(act_idx)

    log_prob = log_prob_cont + log_prob_act

    # Advantage: actual reward minus value estimate
    reward_tensor = torch.tensor(reward, dtype=torch.float32).to(device)
    advantage = reward_tensor - value.detach()

    # Policy gradient loss + value loss
    policy_loss = -log_prob * advantage
    value_loss  = F.mse_loss(value, reward_tensor)
    loss        = policy_loss + 0.5 * value_loss

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
    optimizer.step()

    policy.eval()

    return {
        'policy_loss':     policy_loss.item(),
        'value_loss':      value_loss.item(),
        'advantage':       advantage.item(),
        'value_estimate':  value.item(),
    }


def refit_edmd(models, processed_features: list) -> dict:
    """
    Refits EDMDModel on all available real data and updates models.edmd in-place.

    Args:
        models:             ModelBundle from app.state.models.
        processed_features: All available processed feature dicts.

    Returns:
        dict: {'val_mse', 'n_days_used'}
    """
    logger.info(f"Refitting EDMD on {len(processed_features)} days...")

    from sklearn.model_selection import train_test_split

    # Build (X, y) pairs with seq_len=1
    X_seq, y_scaled = models.processor.prepare_data_for_training(
        processed_features,
        num_days_in_state=1,
    )
    if X_seq.shape[0] < 20:
        logger.warning("Too few samples to refit EDMD. Skipping.")
        return {'val_mse': None, 'n_days_used': len(processed_features)}

    X_scaled = X_seq.reshape(len(X_seq), 23)

    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y_scaled,
        test_size=0.15,
        shuffle=True,
        random_state=42,
    )

    new_edmd = EDMDModel(
        degree=config.EDMD_DEGREE,
        alpha=config.EDMD_ALPHA,
    )
    new_edmd.fit(X_train, y_train)

    # Evaluate
    pred_val = np.array([new_edmd.predict(x) for x in X_val])
    val_mse  = float(np.mean((pred_val - y_val) ** 2))
    logger.info(f"EDMD refit complete. Val MSE (scaled): {val_mse:.4f}")

    # Update in-place, env will pick up new model on next inference
    models.edmd = new_edmd
    new_edmd.save(config.EDMD_MODEL_SAVE_PATH)
    models.policy.save(config.POLICY_SAVE_PATH)

    return {'val_mse': val_mse, 'n_days_used': len(processed_features)}


# ------------------------------------------------------------------
# Main nightly job
# ------------------------------------------------------------------

async def run_nightly_loop(app_state) -> dict:
    """
    Nightly background task. Called by APScheduler, runs inside FastAPI process.

    Workflow:
        1. Fetch last NUM_DAYS_FOR_STATE + 1 days of real Garmin data
        2. Compute actual sleep score for yesterday
        3. Build observation state (last NUM_DAYS_FOR_STATE days before yesterday)
        4. Single online policy gradient step
        5. If today is a refit day: refit EDMD on all available data
        6. Save updated policy weights

    Args:
        app_state: FastAPI app.state (must have .models: ModelBundle)

    Returns:
        dict: Loop run summary for logging/debugging.
    """
    models   = app_state.models
    today    = date.today()
    yesterday = today - timedelta(days=1)

    result = {
        'run_date':       today.isoformat(),
        'status':         'ok',
        'reward':         None,
        'policy_update':  None,
        'edmd_refit':     None,
        'error':          None,
    }

    try:
        # --- 1. Fetch real data ---
        # Need seq_len days for state + 1 more for yesterday's reward
        n_fetch = config.NUM_DAYS_FOR_STATE + 3
        logger.info(f"Nightly loop: fetching {n_fetch} days up to {yesterday}...")

        raw_data = get_historical_metrics(n_fetch, end_date=yesterday)
        if not raw_data or len(raw_data) < config.NUM_DAYS_FOR_STATE + 1:
            raise RuntimeError(
                f"Insufficient data: got {len(raw_data) if raw_data else 0} days, "
                f"need {config.NUM_DAYS_FOR_STATE + 1}."
            )

        processed = [extract_daily_features(d) for d in raw_data]

        # --- 2. Compute actual reward (yesterday's sleep score) ---
        reward = compute_real_reward(raw_data[-1])
        result['reward'] = reward
        logger.info(f"Yesterday's actual sleep reward: {reward:.2f}")

        database.upsert_outcome({
            'date':         yesterday.isoformat(),
            'actual_score': round(reward, 2),  # store 0-100, not sharpened reward
            'actual_steps': None,
            'actual_activity': None,
            'actual_bed_hour': None,
            'actual_bed_minute': None,
            'actual_wake_hour': None,
            'actual_wake_minute': None,
            'followed_recommendation': None,
            'notes': 'auto-filled from Garmin',
})

        # --- 3. Build state from the NUM_DAYS_FOR_STATE days before yesterday ---
        # i.e., the observation the policy would have seen before yesterday's recommendation
        state_features = processed[:-1]  # everything except yesterday
        state = build_real_state(state_features, models.processor, config.NUM_DAYS_FOR_STATE)
        if state is None:
            raise RuntimeError("Could not build valid state from real data.")

        # --- 4. Online policy update ---
        update_info = update_policy(
            policy=models.policy,
            device=models.device,
            state=state,
            reward=reward,
            lr=config.ONLINE_LEARNING_RATE,
        )
        result['policy_update'] = update_info
        logger.info(
            f"Policy updated. advantage={update_info['advantage']:.3f}, "
            f"value_est={update_info['value_estimate']:.2f}"
        )

        # --- 5. EDMD refit (every EDMD_REFIT_INTERVAL days) --- // Currently turned off
        day_of_year = today.timetuple().tm_yday
        if day_of_year % EDMD_REFIT_INTERVAL == 0 and False:
            logger.info("EDMD refit day — fetching full history...")
            full_raw = get_historical_metrics(
                config.NUM_DAYS_TO_FETCH_RAW,
                end_date=yesterday,
            )
            full_processed = [extract_daily_features(d) for d in full_raw]
            refit_info = refit_edmd(models, full_processed)
            result['edmd_refit'] = refit_info
        else:
            days_until = EDMD_REFIT_INTERVAL - (day_of_year % EDMD_REFIT_INTERVAL)
            logger.info(f"EDMD refit in {days_until} days.")

        # --- 6. Save policy ---
        models.policy.save(config.POLICY_SAVE_PATH)
        logger.info("Policy weights saved.")

    except Exception as e:
        logger.error(f"Nightly loop failed: {e}", exc_info=True)
        result['status'] = 'error'
        result['error']  = str(e)

    return result


# ------------------------------------------------------------------
# APScheduler registration helper
# ------------------------------------------------------------------

def register_nightly_job(scheduler, app_state):
    """
    Registers the nightly loop with APScheduler.

    Args:
        scheduler:  AsyncIOScheduler instance.
        app_state:  FastAPI app.state with .models already populated.
    """
    scheduler.add_job(
        run_nightly_loop,
        trigger='cron',
        hour=3,           # 3 AM — Garmin sync should be complete by then
        minute=0,
        kwargs={'app_state': app_state},
        id='nightly_online_loop',
        replace_existing=True,
        misfire_grace_time=3600,  # if server was down, run within 1h of scheduled time
    )
    logger.info("Nightly online loop registered for 03:00 daily.")