# src/features/utils.py
# This module contains utility functions for feature engineering.
import numpy as np
from src.models.data_processor import DataProcessor

def calculate_sleep_score_proxy(sleep_data: dict) -> float:
    """
    Calculates a simplified sleep score proxy (0-100) based on Garmin sleep data.
    This version attempts to better align with Garmin's known factors and emphasis,
    incorporating user feedback on specific metric ratings.

    Args:
        sleep_data (dict): A dictionary containing raw sleep metrics from Garmin.

    Returns:
        float: A calculated sleep score proxy between 0 and 100.
    """
    if not sleep_data:
        return 50.0 # Default/neutral score if no sleep data

    # Extract relevant fields, providing default values if keys are missing
    daily_sleep_dto = sleep_data
    total_sleep_seconds = daily_sleep_dto.get('total_sleep_seconds', 0)
    deep_sleep_seconds = daily_sleep_dto.get('deep_sleep_seconds', 0)
    rem_sleep_seconds = daily_sleep_dto.get('rem_sleep_seconds', 0)
    awake_sleep_seconds = daily_sleep_dto.get('awake_sleep_seconds', 0)

    restless_moments_count = sleep_data.get('restless_moments_count', 0)
    resting_heart_rate = sleep_data.get('resting_heart_rate', 0) or sleep_data.get('sleep_resting_heart_rate',0) # From sleepData
    avg_sleep_stress = daily_sleep_dto.get('avg_sleep_stress', 0) # Average stress during sleep

    score = 0.0

    # --- Scoring Components (Adjusted Point Allocations) ---

    # 1. Total Sleep Duration (Max 35 points)
    total_sleep_hours = total_sleep_seconds / 3600
    if 7.5 <= total_sleep_hours <= 9.5: 
        score += 35.0
    elif 6.0 <= total_sleep_hours < 7.0 or 9.5 < total_sleep_hours <= 10.5:
        score += 25.0 # Decent but not ideal
    elif total_sleep_hours > 0:
        score += 12.0 # Some sleep, but poor duration

    # 2. Sleep Quality - Awake Time (Max 10 points awarded for low awake time, penalties for high)
    awake_minutes = awake_sleep_seconds / 60
    if awake_minutes <= 15:
        score += 10.0 # Excellent (very little awake time)
    elif 15 < awake_minutes <= 45: 
        score += 5.0
    elif awake_minutes > 45: # Only penalize for excessive awake time
        score -= 7.0

    # 3. Sleep Quality - Restlessness (Max 10 points awarded for low restlessness, penalties for high)
    if restless_moments_count <= 30: # Very low restlessness
        score += 10.0
    elif 30 < restless_moments_count <= 70: 
        score += 5.0
    elif restless_moments_count > 70:
        score -= 7.0

    # 4. Deep Sleep Ratio (Max 15 points)
    total_asleep_seconds = total_sleep_seconds - awake_sleep_seconds
    if total_asleep_seconds > 0:
        deep_ratio = deep_sleep_seconds / total_asleep_seconds
        if 0.15 <= deep_ratio <= 0.25: # Ideal range
            score += 15.0
        elif 0.10 <= deep_ratio < 0.15 or 0.25 < deep_ratio <= 0.30: 
            score += 7.0 # Reduced from 8.0 to fine-tune
    # Else no points if no total asleep time

    # 5. REM Sleep Ratio (Max 15 points)
    if total_asleep_seconds > 0:
        rem_ratio = rem_sleep_seconds / total_asleep_seconds
        if 0.20 <= rem_ratio <= 0.25: # Ideal range
            score += 15.0
        elif 0.15 <= rem_ratio < 0.20 or 0.25 < rem_ratio <= 0.30: 
            score += 7.0 # Reduced from 8.0 to fine-tune
    # Else no points if no total asleep time

    # 6. Overnight Recovery - Resting Heart Rate (Max 15 points)
    if resting_heart_rate > 0:
        if resting_heart_rate <= 50: # Excellent RHR
            score += 15.0
        elif resting_heart_rate <= 60: # Good RHR
            score += 10.0
        elif resting_heart_rate <= 70: # Fair RHR
            score += 5.0

    # 7. Overnight Recovery - Average Sleep Stress (Max 10 points)
    if avg_sleep_stress >= 0 and avg_sleep_stress <= 100:
        # Scale 0-100 stress to 10-0 points. Lower stress = higher points.
        score += (100 - avg_sleep_stress) * (10.0 / 100.0)

    # Ensure final score is within 0-100 range
    return max(0.0, min(100.0, score))


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
