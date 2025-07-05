# src/models/mock_prediction_model.py
# This file provides a mock prediction model for initial integration testing.
# It simply "predicts" the next day's features by returning the last day's features from the input sequence.

from typing import List, Dict, Any
import numpy as np
from datetime import datetime,timedelta

class MockPredictionModel:
    """
    A mock prediction model that simulates predicting the next day's features.
    For testing purposes, it returns the features of the last day in the input sequence.
    """
    def __init__(self, feature_names: List[str], sleep_metrics_keys: List[str]):
        """
        Initializes the mock model.
        Args:
            feature_names (List[str]): Ordered list of all top-level feature names.
            sleep_metrics_keys (List[str]): Ordered list of keys within the 'sleep_metrics' dictionary.
        """
        self.feature_names = feature_names
        self.sleep_metrics_keys = sleep_metrics_keys

    def predict(self, state_vector: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Simulates predicting the next day's features.
        Returns the features of the last day in the input state_vector.

        Args:
            state_vector (List[Dict[str, Any]]): A list of dictionaries, where each dictionary
                                                 represents a day's features in the state sequence.
                                                 This is the 'features' list from create_state_vectors in src/features/feature_engineer.py.

        Returns:
            Dict[str, Any]: A dictionary representing the predicted features for the next day.
                            This will be a copy of the last day in the state_vector.
        """
        if not state_vector:
            print("Warning: Empty state vector provided to MockPredictionModel. Returning empty dict.")
            return {}

        # The mock prediction is simply the last day's features
        predicted_features = state_vector[-1].copy()

        # Update the date for the predicted day (for realism in simulation)
        # Assuming date is 'YYYY-MM-DD'
        current_date_str = predicted_features['date']
        current_date = datetime.strptime(current_date_str, "%Y-%m-%d").date()
        next_date = current_date + timedelta(days=1)
        predicted_features['date'] = next_date.strftime("%Y-%m-%d")

        print(f"MockPredictionModel predicted features for {predicted_features['date']} (copy of last input day).")
        return predicted_features

    def train(self, X_train: Any, y_train: Any):
        """
        Mock training function. Does nothing.
        """
        print("MockPredictionModel: No training performed.")

    def evaluate(self, X_test: Any, y_test: Any):
        """
        Mock evaluation function. Does nothing.
        """
        print("MockPredictionModel: No evaluation performed.")

# Example usage (for internal testing of the mock model)
if __name__ == "__main__":
    from datetime import date

    # Define dummy feature names and sleep metrics keys for initialization
    dummy_feature_names = [
        'total_steps', 'avg_heart_rate', 'resting_heart_rate',
        'avg_respiration_rate', 'avg_stress', 'body_battery_end_value',
        'activity_type_flags', 'sleep_metrics', 'wake_time_gmt', 'bed_time_gmt'
    ]
    dummy_sleep_metrics_keys = [
        'total_sleep_seconds', 'deep_sleep_seconds', 'rem_sleep_seconds',
        'awake_sleep_seconds', 'restless_moments_count', 'avg_sleep_stress',
        'resting_heart_rate'
    ]

    mock_model = MockPredictionModel(dummy_feature_names, dummy_sleep_metrics_keys)

    # Create a dummy state vector (e.g., 3 days of data)
    dummy_state_vector = []
    for i in range(3):
        day_features = {
            'date': (date.today() - timedelta(days=2-i)).strftime("%Y-%m-%d"),
            'total_steps': 5000 + i*100,
            'avg_heart_rate': 70.0 + i,
            'resting_heart_rate': 50.0 - i,
            'avg_respiration_rate': 14.0,
            'avg_stress': 20.0 + i*5,
            'body_battery_end_value': 60.0 + i*5,
            'activity_type_flags': {'Strength': 0, 'Cardio': 1, 'Yoga': 0, 'Stretching': 0, 'OtherActivity': 0, 'NoActivity': 0},
            'sleep_metrics': {
                'total_sleep_seconds': 28800 + i*60, # 8 hours + some
                'deep_sleep_seconds': 3600 + i*30,
                'rem_sleep_seconds': 5400 + i*30,
                'awake_sleep_seconds': 1800 - i*10,
                'restless_moments_count': 30 - i*5,
                'avg_sleep_stress': 10.0 + i*2,
                'resting_heart_rate': 48.0 - i
            },
            'wake_time_gmt': f"GMT_WAKE_{i}",
            'bed_time_gmt': f"GMT_BED_{i}"
        }
        dummy_state_vector.append(day_features)

    print("\nDummy State Vector Input:")
    for day in dummy_state_vector:
        print(day)

    predicted_day_features = mock_model.predict(dummy_state_vector)

    print("\nMock Predicted Day Features:")
    print(predicted_day_features)
