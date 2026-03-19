# src/features/feature_engineer.py
# This module is responsible for extracting and transforming raw Garmin metrics
# into a standardized set of daily features, and then compiling these into
# time-series state vectors for the prediction model.

from datetime import datetime, timedelta
from typing import List, Dict, Any

def extract_daily_features(raw_daily_metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extracts and standardizes key daily features from the raw Garmin metrics.
    Handles missing data by returning default/N/A values, explicitly converting
    None to 0/0.0 for numerical fields.

    Args:
        raw_daily_metrics (Dict[str, Any]): A dictionary containing raw metrics for a single day,
                                            as returned by garmin_parser.get_daily_metrics.

    Returns:
        Dict[str, Any]: A standardized dictionary of daily features.
    """
    features = {
        'date': raw_daily_metrics.get('date'),
        'total_steps': 0,
        'avg_heart_rate': 0.0,
        'resting_heart_rate': 0.0,
        'avg_respiration_rate': 0.0,
        'avg_stress': 0.0,
        'body_battery_end_value': 0.0,
        'activity_type_flags': {
            'Strength': 0,
            'Cardio': 0,
            'Yoga': 0,
            'Stretching': 0,
            'OtherActivity': 0,
            'NoActivity': 1
        },
        'sleep_metrics': {
            'total_sleep_seconds': 0.0,
            'deep_sleep_seconds' : 0.0,
            'rem_sleep_seconds' : 0.0,
            'awake_sleep_seconds': 0.0,
            'restless_moments_count': 0.0,
            'avg_sleep_stress': 0.0,
            'resting_heart_rate':0.0,
        },
        'wake_time_gmt': 'N/A',
        'bed_time_gmt': 'N/A',
    }

    # --- Process Sleep Data ---
    sleep_data = raw_daily_metrics.get('sleepData', {})
    if sleep_data:
        daily_sleep_dto = sleep_data.get('dailySleepDTO', {})

        # Explicitly handle None for sleep metrics
        features['sleep_metrics']['total_sleep_seconds'] = float(daily_sleep_dto.get('sleepTimeSeconds') or 0.0)
        features['sleep_metrics']['deep_sleep_seconds']= float(daily_sleep_dto.get('deepSleepSeconds') or 0.0)
        features['sleep_metrics']['rem_sleep_seconds'] = float(daily_sleep_dto.get('remSleepSeconds') or 0.0)
        features['sleep_metrics']['awake_sleep_seconds'] = float(daily_sleep_dto.get('awakeSleepSeconds') or 0.0)

        features['sleep_metrics']['restless_moments_count'] = float(sleep_data.get('restlessMomentsCount') or 0.0)
        features['sleep_metrics']['resting_heart_rate'] = float(sleep_data.get('restingHeartRate') or 0.0)
        features['sleep_metrics']['avg_sleep_stress'] = float(daily_sleep_dto.get('avgSleepStress') or 0.0)

        features['bed_time_gmt'] = daily_sleep_dto.get('sleepStartTimestampGMT', 'N/A')
        features['wake_time_gmt'] = daily_sleep_dto.get('sleepEndTimestampGMT', 'N/A')

    # --- Process Daily Summary Data (Primary Source for many aggregates) ---
    daily_summary_data = raw_daily_metrics.get('dailySummaryData', {})
    if daily_summary_data:
        features['total_steps'] = int(daily_summary_data.get('totalSteps') or 0)
        features['resting_heart_rate'] = float(daily_summary_data.get('restingHeartRate') or features['resting_heart_rate'])
        features['avg_stress'] = float(daily_summary_data.get('averageStressLevel') or 0.0)
        features['body_battery_end_value'] = float(daily_summary_data.get('bodyBatteryMostRecentValue') or 0.0)
        features['avg_respiration_rate'] = float(daily_summary_data.get('avgSleepRespirationValue') or daily_summary_data.get('avgWakingRespirationValue') or 0.0)

        min_avg_hr = daily_summary_data.get('minAvgHeartRate')
        max_avg_hr = daily_summary_data.get('maxAvgHeartRate')

        # Robustly calculate avg_heart_rate
        if min_avg_hr is not None and max_avg_hr is not None and min_avg_hr > 0 and max_avg_hr > 0:
            features['avg_heart_rate'] = (float(min_avg_hr) + float(max_avg_hr)) / 2.0
        else:
            features['avg_heart_rate'] = float(daily_summary_data.get('averageHeartRate') or features['avg_heart_rate'])


    # --- Process Heart Rate Data (Fallback/Detail for avg HR) ---
    heart_rate_data = raw_daily_metrics.get('heartRateData', {})
    if features['avg_heart_rate'] == 0.0 and heart_rate_data and heart_rate_data.get('heartRate'):
        features['avg_heart_rate'] = float(heart_rate_data['heartRate'].get('avg') or 0.0)

    # --- Process Respiration Data (Fallback if not in daily summary) ---
    if features['avg_respiration_rate'] == 0.0 and raw_daily_metrics.get('respirationData'):
        respiration_data = raw_daily_metrics['respirationData']
        features['avg_respiration_rate'] = float(respiration_data.get('avgSleepRespirationValue') or respiration_data.get('avgWakingRespirationValue') or 0.0)

    # --- Process Stress Data (Fallback if not in daily summary) ---
    if features['avg_stress'] == 0.0 and raw_daily_metrics.get('stressData'):
        stress_data = raw_daily_metrics['stressData']
        features['avg_stress'] = float(stress_data.get('avgStressLevel') or 0.0)

    # --- Process Body Battery Data (Fallback if not in daily summary, or for specific 'total') ---
    body_battery_data = raw_daily_metrics.get('bodyBatteryData', [])
    if features['body_battery_end_value'] == 0.0 and isinstance(body_battery_data, list) and len(body_battery_data) > 0:
        features['body_battery_end_value'] = float(body_battery_data[0].get('total') or 0.0)


    # --- Process Activity Data ---
    activity_data = raw_daily_metrics.get('activityData', [])
    if activity_data:
        for key in features['activity_type_flags']:
            features['activity_type_flags'][key] = 0
        features['activity_type_flags']['NoActivity'] = 0

        activity_found = False
        for activity in activity_data:
            activity_type = activity.get('activityType', {}).get('typeKey', '').lower()
            if 'strength_training' in activity_type:
                features['activity_type_flags']['Strength'] = 1
                activity_found = True
            elif 'running' in activity_type or 'cycling' in activity_type or 'swimming' in activity_type or 'cardio' in activity_type:
                features['activity_type_flags']['Cardio'] = 1
                activity_found = True
            elif 'yoga' in activity_type:
                features['activity_type_flags']['Yoga'] = 1
                activity_found = True
            elif 'stretching' in activity_type:
                features['activity_type_flags']['Stretching'] = 1
                activity_found = True
            else:
                features['activity_type_flags']['OtherActivity'] = 1
                activity_found = True

        if not activity_found:
            features['activity_type_flags']['NoActivity'] = 1

    return features

# --- Main execution for testing purposes ---
if __name__ == "__main__":
    # This block is for testing feature_engineer.py independently.
    # In a real scenario, it would be called by main.py or other modules.

    from src.data_ingestion.garmin_parser import get_historical_metrics

    print("--- Running feature_engineer.py in standalone test mode ---")
    NUM_DAYS_FOR_STATE = 7 # Example: use 7 days for the state vector (your 'x')
    NUM_DAYS_TO_FETCH_RAW = NUM_DAYS_FOR_STATE + 2 # Fetch a few extra days to ensure enough data for states

    # Determine the end date for data fetching (yesterday) for consistent testing
    end_date_for_test = datetime.now().date() - timedelta(days=1)
    raw_historical_data = get_historical_metrics(NUM_DAYS_TO_FETCH_RAW, end_date=end_date_for_test)


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
    else:
        print("No raw historical data fetched. Cannot proceed with feature engineering.")