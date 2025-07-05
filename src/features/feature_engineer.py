# src/features/feature_engineer.py
# This module is responsible for extracting and transforming raw Garmin metrics
# into a standardized set of daily features, and then compiling these into
# time-series state vectors for the prediction model.

from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

# Import the utility functions, including our sleep score proxy calculator
from src.features.utils import calculate_sleep_score_proxy

def extract_daily_features(raw_daily_metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extracts and standardizes key daily features from the raw Garmin metrics.
    Handles missing data by returning default/N/A values.

    Args:
        raw_daily_metrics (Dict[str, Any]): A dictionary containing raw metrics for a single day,
                                            as returned by garmin_parser.get_daily_metrics.

    Returns:
        Dict[str, Any]: A standardized dictionary of daily features.
    """
    features = {
        'date': raw_daily_metrics.get('date'),
        'total_steps': 0,
        'avg_heart_rate': 0.0, # Will try to populate
        'resting_heart_rate': 0.0,
        'avg_respiration_rate': 0.0,
        'avg_stress': 0.0,
        'body_battery_end_value': 0.0, # Using 'end_value' or 'most_recent_value' as the daily summary
        'activity_type_flags': { # One-hot encoded or similar for activity types
            'Strength': 0,
            'Cardio': 0,
            'Yoga': 0,
            'Stretching': 0,
            'OtherActivity': 0, # Catch-all for recognized activities not explicitly flagged
            'NoActivity': 1 # Default to NoActivity if no activities found
        },
        'sleep_metrics': {
            'total_sleep_seconds': 0.0,
            'deep_sleep_seconds' : 0.0,
            'rem_sleep_seconds' : 0.0,
            'awake_sleep_seconds': 0.0, #ironic, isn't it
            'restless_moments_count': 0.0,
            'avg_sleep_stress': 0.0,
            'resting_heart_rate':0.0,
        },
        'wake_time_gmt': 'N/A', # Will be timestamp string (GMT)
        'bed_time_gmt': 'N/A',  # Will be timestamp string (GMT)
    }

    # --- Process Sleep Data ---
    sleep_data = raw_daily_metrics.get('sleepData', {})
    if sleep_data:
        # Extract dailySleepDTO
        daily_sleep_dto = sleep_data.get('dailySleepDTO', {})

        features['sleep_metrics']['total_sleep_seconds'] = daily_sleep_dto.get('sleepTimeSeconds', 0)
        features['sleep_metrics']['deep_sleep_seconds']= daily_sleep_dto.get('deepSleepSeconds', 0)
        features['sleep_metrics']['rem_sleep_seconds'] = daily_sleep_dto.get('remSleepSeconds', 0)
        features['sleep_metrics']['awake_sleep_seconds'] = daily_sleep_dto.get('awakeSleepSeconds', 0)

        features['sleep_metrics']['restless_moments_count'] = sleep_data.get('restlessMomentsCount', 0)
        features['sleep_metrics']['resting_heart_rate'] = sleep_data.get('restingHeartRate', 0) # From sleepData
        features['sleep_metrics']['avg_sleep_stress'] = daily_sleep_dto.get('avgSleepStress', 0) # Average stress during sleep


        features['bed_time_gmt'] = daily_sleep_dto.get('sleepStartTimestampGMT', 'N/A')
        features['wake_time_gmt'] = daily_sleep_dto.get('sleepEndTimestampGMT', 'N/A')

    # --- Process Daily Summary Data (Primary Source for many aggregates) ---
    daily_summary_data = raw_daily_metrics.get('dailySummaryData', {})
    if daily_summary_data:
        features['total_steps'] = daily_summary_data.get('totalSteps', 0)
        features['resting_heart_rate'] = daily_summary_data.get('restingHeartRate', features['resting_heart_rate']) # Prioritize daily summary for resting HR
        features['avg_stress'] = daily_summary_data.get('averageStressLevel', 0.0)
        features['body_battery_end_value'] = daily_summary_data.get('bodyBatteryMostRecentValue', 0.0)
        features['avg_respiration_rate'] = daily_summary_data.get('avgSleepRespirationValue', daily_summary_data.get('avgWakingRespirationValue', 0.0))

        # Refined logic for avg_heart_rate:
        # Try to calculate avg_heart_rate from minAvgHeartRate and maxAvgHeartRate from dailySummaryData
        min_avg_hr = daily_summary_data.get('minAvgHeartRate', 0)
        max_avg_hr = daily_summary_data.get('maxAvgHeartRate', 0)
        if min_avg_hr > 0 and max_avg_hr > 0:
            features['avg_heart_rate'] = (min_avg_hr + max_avg_hr) / 2.0
        else:
            # If min/max avg HR not available, try the general 'averageHeartRate' key
            features['avg_heart_rate'] = daily_summary_data.get('averageHeartRate', features['avg_heart_rate'])


    # --- Process Heart Rate Data (Fallback/Detail for avg HR) ---
    # Only try this if avg_heart_rate wasn't found in daily summary
    heart_rate_data = raw_daily_metrics.get('heartRateData', {})
    if features['avg_heart_rate'] == 0.0 and heart_rate_data and heart_rate_data.get('heartRate'):
        features['avg_heart_rate'] = heart_rate_data['heartRate'].get('avg', 0.0)

    # --- Process Respiration Data (Fallback if not in daily summary) ---
    if features['avg_respiration_rate'] == 0.0 and raw_daily_metrics.get('respirationData'): # Only if not set by daily summary
        features['avg_respiration_rate'] = raw_daily_metrics['respirationData'].get('avgSleepRespirationValue', raw_daily_metrics['respirationData'].get('avgWakingRespirationValue', 0.0))

    # --- Process Stress Data (Fallback if not in daily summary) ---
    if features['avg_stress'] == 0.0 and raw_daily_metrics.get('stressData'): # Only if not set by daily summary
        features['avg_stress'] = raw_daily_metrics['stressData'].get('avgStressLevel', 0.0)

    # --- Process Body Battery Data (Fallback if not in daily summary, or for specific 'total') ---
    body_battery_data = raw_daily_metrics.get('bodyBatteryData', [])
    if features['body_battery_end_value'] == 0.0 and isinstance(body_battery_data, list) and len(body_battery_data) > 0:
        # If daily summary didn't provide it, try to get 'total' from the first item of the list
        features['body_battery_end_value'] = body_battery_data[0].get('total', 0.0)


    # --- Process Activity Data ---
    activity_data = raw_daily_metrics.get('activityData', [])
    if activity_data:
        # Reset all activity flags to 0 first
        for key in features['activity_type_flags']:
            features['activity_type_flags'][key] = 0
        features['activity_type_flags']['NoActivity'] = 0 # Assume activity exists

        for activity in activity_data:
            activity_type = activity.get('activityType', {}).get('typeKey', '').lower() # 'typeKey' is common for activity type
            if 'strength_training' in activity_type:
                features['activity_type_flags']['Strength'] = 1
            elif 'running' in activity_type or 'cycling' in activity_type or 'swimming' in activity_type or 'cardio' in activity_type:
                features['activity_type_flags']['Cardio'] = 1
            elif 'yoga' in activity_type:
                features['activity_type_flags']['Yoga'] = 1
            elif 'stretching' in activity_type:
                features['activity_type_flags']['Stretching'] = 1
            else:
                # If it's a recognized activity but not one of our specific flags
                features['activity_type_flags']['OtherActivity'] = 1

        # If no specific activity types were matched, but there were activities,
        # we might want to set 'NoActivity' to 0 and leave other flags as 0.
        # For simplicity, if any activity exists, 'NoActivity' is 0.
        if any(features['activity_type_flags'].values()): # If any flag is 1
             features['activity_type_flags']['NoActivity'] = 0
        else: # If no specific activities matched, but activityData was not empty
            features['activity_type_flags']['NoActivity'] = 1 # Revert to NoActivity if no specific type matched

    return features

def create_state_vectors(historical_daily_features: List[Dict[str, Any]], num_days_in_state: int) -> List[Dict[str, Any]]:
    """
    Creates time-series state vectors from historical daily features.
    Each state vector represents 'num_days_in_state' consecutive days of features.

    Args:
        historical_daily_features (List[Dict[str, Any]]): A list of standardized daily feature dictionaries,
                                                           sorted from oldest to newest.
        num_days_in_state (int): The number of past days to include in each state vector (your 'x').

    Returns:
        List[Dict[str, Any]]: A list of state vectors. Each state vector is a dictionary
                              containing 'date_end' (the date of the last day in the sequence)
                              and 'features' (a list of dictionaries, one for each day in the sequence).
    """
    state_vectors = []
    if len(historical_daily_features) < num_days_in_state:
        print(f"Warning: Not enough historical data ({len(historical_daily_features)} days) to create "
              f"state vectors of {num_days_in_state} days. Skipping state vector creation.")
        return []

    for i in range(len(historical_daily_features) - num_days_in_state + 1):
        # A state vector consists of 'num_days_in_state' consecutive days
        current_state_sequence = historical_daily_features[i : i + num_days_in_state]

        # The 'date_end' for the state vector is the date of the last day in the sequence
        date_end = current_state_sequence[-1]['date']

        # We'll include all extracted features for each day in the sequence
        state_vectors.append({
            'date_end': date_end,
            'features': current_state_sequence
        })

    return state_vectors

# --- Main execution for testing purposes ---
if __name__ == "__main__":
    # This block is for testing feature_engineer.py independently.
    # In a real scenario, it would be called by main.py or other modules.

    # Adjust NUM_DAYS_TO_FETCH_RAW if you want to test with more/less data
    # NUM_DAYS_FOR_STATE determines the length of each state vector (your 'x')
    NUM_DAYS_FOR_STATE = 7 # Example: use 7 days for the state vector (your 'x')
    NUM_DAYS_TO_FETCH_RAW = NUM_DAYS_FOR_STATE + 2 # Fetch a few extra days to ensure enough data for states

    from src.data_ingestion.garmin_parser import get_historical_metrics

    print("--- Running feature_engineer.py in standalone test mode ---")

    print(f"\nCollecting historical data for the last {NUM_DAYS_TO_FETCH_RAW} days...")
    raw_historical_data = get_historical_metrics(NUM_DAYS_TO_FETCH_RAW)

    if raw_historical_data:
        print(f"\n--- Processing {len(raw_historical_data)} days of raw data into features ---")
        processed_features = []
        for day_raw_data in raw_historical_data:
            daily_features = extract_daily_features(day_raw_data)
            processed_features.append(daily_features)
            # Print a simplified view of the extracted features for each day
            print(f"  Date: {daily_features['date']}, Sleep Metrics: {daily_features['sleep_metrics']}, "
                  f"Steps: {daily_features['total_steps']}, Avg HR: {daily_features['avg_heart_rate']:.1f}, "
                  f"Resting HR: {daily_features['resting_heart_rate']:.1f}, "
                  f"Avg Respiration: {daily_features['avg_respiration_rate']:.1f}, "
                  f"Avg Stress: {daily_features['avg_stress']:.1f}, BB End: {daily_features['body_battery_end_value']:.1f}, "
                  f"Activities: {daily_features['activity_type_flags']}")

        print(f"\n--- Creating state vectors of {NUM_DAYS_FOR_STATE} days ---")
        state_vectors = create_state_vectors(processed_features, NUM_DAYS_FOR_STATE)

        if state_vectors:
            print(f"Successfully created {len(state_vectors)} state vector(s).")
            # Print the last state vector as an example
            print(f"\n--- Example Last State Vector (Ending {state_vectors[-1]['date_end']}) ---")
            for i, day_feature in enumerate(state_vectors[-1]['features']):
                print(f"  Day {i+1} ({day_feature['date']}):")
                print(f"    Total Steps: {day_feature['total_steps']}"
                f"    Avg HR: {day_feature['avg_heart_rate']:.1f}"
                f"    Resting HR: {day_feature['resting_heart_rate']:.1f}"
                f"    Avg Respiration: {day_feature['avg_respiration_rate']:.1f}"
                f"    Avg Stress: {day_feature['avg_stress']:.1f}"
                f"    Body Battery End: {day_feature['body_battery_end_value']:.1f}"
                f"    Activities: {day_feature['activity_type_flags']}"
                f"    Bed Time GMT: {day_feature['bed_time_gmt']}"
                f"    Wake Time GMT: {day_feature['wake_time_gmt']}"
                f"    Sleep Metrics: {day_feature['sleep_metrics']}")
        else:
            print("No state vectors could be created. Check if enough historical data was fetched.")
    else:
        print("No raw historical data fetched. Cannot proceed with feature engineering.")
