# src/features/utils.py
# This module contains utility functions for feature engineering.

def calculate_sleep_score_proxy(sleep_data: dict) -> float:
    """
    Calculates a simplified sleep score proxy (0-100) based on Garmin sleep data.
    This is a heuristic and can be refined later to better match Garmin's proprietary score.

    Args:
        sleep_data (dict): A dictionary containing raw sleep metrics from Garmin.

    Returns:
        float: A calculated sleep score proxy between 0 and 100.
    """
    if not sleep_data:
        return 50.0 # Default/neutral score if no sleep data

    # Extract relevant fields, providing default values if keys are missing
    # All primary sleep duration metrics are nested under 'dailySleepDTO'
    daily_sleep_dto = sleep_data.get('dailySleepDTO', {})
    total_sleep_seconds = daily_sleep_dto.get('sleepTimeSeconds', 0)
    deep_sleep_seconds = daily_sleep_dto.get('deepSleepSeconds', 0)
    rem_sleep_seconds = daily_sleep_dto.get('remSleepSeconds', 0) 
    awake_sleep_seconds = daily_sleep_dto.get('awakeSleepSeconds', 0)

    # Restless moments and resting heart rate are direct keys in sleep_data
    restless_moments_count = sleep_data.get('restlessMomentsCount', 0)
    resting_heart_rate = sleep_data.get('restingHeartRate', 0)

    score = 0.0

    # 1. Total Sleep Duration (e.g., 7-9 hours is ideal, contributes up to 30 points)
    total_sleep_hours = total_sleep_seconds / 3600
    if 7.0 <= total_sleep_hours <= 9.0:
        score += 30.0
    elif 6.0 <= total_sleep_hours < 7.0 or 9.0 < total_sleep_hours <= 10.0:
        score += 15.0
    elif total_sleep_hours > 0: # Some sleep but outside ideal range
        score += 5.0

    # 2. Deep and REM Sleep Ratio (aim for ~15-25% deep, ~20-25% REM of total sleep)
    # These ratios are typically calculated against total *asleep* time
    total_asleep_seconds = total_sleep_seconds - awake_sleep_seconds
    if total_asleep_seconds > 0:
        deep_ratio = deep_sleep_seconds / total_asleep_seconds
        rem_ratio = rem_sleep_seconds / total_asleep_seconds

        # Deep sleep contribution (up to 25 points)
        if 0.15 <= deep_ratio <= 0.25:
            score += 25.0
        elif 0.10 <= deep_ratio < 0.15 or 0.25 < deep_ratio <= 0.30:
            score += 10.0

        # REM sleep contribution (up to 25 points)
        if 0.20 <= rem_ratio <= 0.25:
            score += 25.0
        elif 0.15 <= rem_ratio < 0.20 or 0.25 < rem_ratio <= 0.30:
            score += 10.0
    else:
        # If no non-awake sleep, these ratios are meaningless, no points added
        pass

    # 3. Restlessness (lower is better, contributes up to 10 points)
    # This is a heuristic: fewer restless moments mean higher score
    if restless_moments_count <= 5:
        score += 10.0
    elif 5 < restless_moments_count <= 10:
        score += 5.0
    # No points for very high restlessness

    # 4. Resting Heart Rate (lower is generally better, contributes up to 10 points)
    # This is a very simplified heuristic, assumes lower is better.
    # A personalized baseline would be better here.
    if resting_heart_rate > 0: # Ensure data exists
        if resting_heart_rate <= 55: # Excellent
            score += 10.0
        elif 55 < resting_heart_rate <= 65: # Good
            score += 5.0
        # No points for higher RHR, or even negative for very high RHR in a more advanced model

    # Ensure score is within 0-100 range
    return max(0.0, min(100.0, score))

