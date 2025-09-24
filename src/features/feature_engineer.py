# src/features/feature_engineer.py
# This module is responsible for extracting and transforming raw Garmin metrics
# into a standardized set of daily features, and then compiling these into
# time-series state vectors for the prediction model.

from datetime import datetime, timedelta
from typing import List, Dict, Any
import copy 

class FeatureEngineer:
    def __init__(self):
        # standard feature set
        self.features_dict = {
            'date': None,
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

    def extract_daily_features(self, raw_daily_metrics: Dict[str, Any]) -> Dict[str, Any]:
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
        features = copy.deepcopy(self.features_dict)

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

    def align_sleep_data(self, daily_features: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Shifts sleep metrics to align with the previous day's actions.
        This ensures the sleep from the night of Day X -> Day X+1 is associated with Day X's routine.

        Args:
            daily_features (List[Dict[str, Any]]): A list of standardized daily feature dictionaries.

        Returns:
            List[Dict[str, Any]]: A new list of features with sleep data correctly shifted.
        """
        if len(daily_features) < 2:
            print("Warning: Not enough data to align sleep metrics. Returning original list.")
            return daily_features

        aligned_features = copy.deepcopy(daily_features)

        for i in range(len(aligned_features) - 1):
            # Take the sleep metrics from the next day's entry and assign them to the current day
            next_day_sleep_metrics = aligned_features[i + 1]['sleep_metrics']
            next_day_bed_time = aligned_features[i + 1]['bed_time_gmt']
            next_day_wake_time = aligned_features[i + 1]['wake_time_gmt']

            # Update the current day's entry with the next day's sleep data
            aligned_features[i]['sleep_metrics'] = next_day_sleep_metrics
            aligned_features[i]['bed_time_gmt'] = next_day_bed_time
            aligned_features[i]['wake_time_gmt'] = next_day_wake_time

        # The last day in the list won't have sleep metrics from the next day, so we leave them as they are
        # They will be placeholders and should be excluded from training sets if needed.

        return aligned_features


    def create_state_vectors(self, historical_daily_features: List[Dict[str, Any]], num_days_in_state: int) -> List[Dict[str, Any]]:
        """
        Creates time-series state vectors from historical daily features.
        Each state vector represents 'num_days_in_state' consecutive days of features.

        Args:
            historical_daily_features (List[Dict[str, Any]]): A list of standardized daily feature dictionaries,
                                                            sorted from oldest to newest.
            num_days_in_state (int): The number of past days to include in each state vector.

        Returns:
            List[Dict[str, Any]]: A list of state vectors. Each state vector is a dictionary
                                containing 'date_end' (the date of the last day in the sequence)
                                and 'features' (a list of dictionaries, one for each day in the sequence).
        """
        # First, align the sleep data to ensure causal consistency
        aligned_features = self.align_sleep_data(historical_daily_features)
        state_vectors = []
        if len(aligned_features) < num_days_in_state + 1:
            print(f"Warning: Not enough historical data ({len(aligned_features)} days) to create "
                    f"state vectors of {num_days_in_state} days. Skipping state vector creation.")
            return []

        # iterate up to the second-to-last day to ensure the final state vector has complete data
        for i in range(len(aligned_features) - num_days_in_state):
            # A state vector consists of 'num_days_in_state' consecutive days
            current_state_sequence = aligned_features[i : i + num_days_in_state]

            # The 'date_end' for the state vector is the date of the last day in the sequence
            date_end = current_state_sequence[-1]['date']

            state_vectors.append({
                'date_end': date_end,
                'features': current_state_sequence
            })

        return state_vectors