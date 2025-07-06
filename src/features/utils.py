# src/features/utils.py
# This module contains utility functions for feature engineering.

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
    resting_heart_rate = sleep_data.get('resting_heart_rate', 0) # From sleepData
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
