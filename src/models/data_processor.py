# src/models/data_processor.py
# This module handles the transformation of daily feature dictionaries
# into numerical tensors suitable for PyTorch models.

from typing import List, Dict, Any, Tuple
import numpy as np
from datetime import datetime
from src import config

class DataProcessor:
    """
    Handles the conversion of structured daily feature dictionaries into
    flat numerical arrays (for model input) and back (for interpretation).
    It also manages feature scaling (e.g., normalization) if needed.
    """
    def __init__(self):
        # Define the order of numerical features for a single day.
        # This order MUST be consistent across all data processing.
        self.biometrical_keys = config.BIOMETRIC_KEYS
        self.action_keys = config.ACTION_KEYS
        self.numerical_feature_keys = self.action_keys+self.biometrical_keys
        self.input_size = len(self.numerical_feature_keys)
        self.output_size = len(self.biometrical_keys)
        print(f"DataProcessor initialized. Total numerical features per day: {self.input_size}")

    def _convert_timestamp_to_time_features(self, timestamp_gmt: str) -> Tuple[int, int]:
        """
        Converts a GMT timestamp string (e.g., '1751235180000') to (hour, minute).
        Returns (0, 0) if conversion fails or timestamp is 'N/A'.
        """
        if timestamp_gmt == 'N/A':
            return 0, 0
        try:
            # Timestamps from Garmin are often milliseconds since epoch
            timestamp_ms = int(timestamp_gmt)
            dt_object = datetime.fromtimestamp(timestamp_ms / 1000) # Convert ms to seconds
            return dt_object.hour, dt_object.minute
        except (ValueError, TypeError):
            print(f"Warning: Could not parse timestamp {timestamp_gmt}. Returning (0, 0).")
            return 0, 0
        
    def flatten_features_for_day(self, daily_features: Dict[str, Any]) -> np.ndarray:
        """
        Converts a single day's structured feature dictionary into a flat numerical NumPy array.
        Ensures consistent order as defined in self.numerical_feature_keys.
        """
        flat_features = []
        activity_set = {'activity_Strength', 'activity_Cardio', 'activity_Yoga', 'activity_Stretching', 'activity_OtherActivity', 'activity_NoActivity',}
        sleep_set = {'total_sleep_seconds','deep_sleep_seconds', 'rem_sleep_seconds', 'awake_sleep_seconds', 'restless_moments_count', 'avg_sleep_stress', 'sleep_resting_heart_rate'}
        
        # Process time features
        bed_hour, bed_minute = self._convert_timestamp_to_time_features(daily_features.get('bed_time_gmt', 'N/A'))
        wake_hour, wake_minute = self._convert_timestamp_to_time_features(daily_features.get('wake_time_gmt', 'N/A'))
        time_dict= {'bed_time_gmt_hour':bed_hour, 'bed_time_gmt_minute':bed_minute, 'wake_time_gmt_hour':wake_hour, 'wake_time_gmt_minute':wake_minute}


        for key in self.numerical_feature_keys:
            if key in sleep_set:
                sleep_metrics = daily_features.get('sleep_metrics', {})
                flat_features.append(sleep_metrics.get(key,0.0))
                continue
            if key in activity_set:
                activity_flags = daily_features.get('activity_type_flags', {})
                flat_features.append(activity_flags.get(key,0))
                continue
            if key in time_dict:
                flat_features.append(time_dict.get(key))
                continue
            flat_features.append(daily_features.get(key,0.0))

        return np.array(flat_features, dtype=np.float32)

    def prepare_data_for_training(self, state_vectors: List[List[Dict[str, Any]]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepares the state vector data into (X, y) pairs for model training.

        Args:
            state_vectors (List[List[Dict[str, Any]]]): A list of state_vectors, containing the data for a set number of days (config.NUM_DAYS_FOR_STATE + 1).

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                X (np.ndarray): Input sequences. Shape (num_samples, num_days_in_state, num_features_per_day).
                y (np.ndarray): Target features for the next day. Shape (num_samples, num_features_per_day).
        """
        X, y = [], []

        # Ensure we have enough data for a full state vector and a target day
        if not state_vectors or len(state_vectors[0]['features']) != config.NUM_DAYS_FOR_INPUT + 1:
            print(f"Warning: Data size mismatch. Expected vectors of length {config.NUM_DAYS_FOR_INPUT + 1} (input + target). Got {len(state_vectors)} vectors with length {len(state_vectors[0]['features']) if state_vectors else 0}.")
            return np.array([]), np.array([])

        flattened_vectors = np.array([[self.flatten_features_for_day(day) for day in v['features']] for v in state_vectors], dtype=np.float32)

        # Separate into X (input sequence) and y (target features)
        action_length = len(self.action_keys)
        for v in flattened_vectors:
            # X is the entire sequence except for the final day
            X.append(v[:-1, :])
            # y is the biometric data from the final day
            y.append(v[-1, action_length:])
            
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


    def reconstruct_features_from_flat(self, flat_features: np.ndarray, date_str: str = "N/A") -> Dict[str, Any]:
        """
        Reconstructs a structured feature dictionary from a flat numerical NumPy array.
        This is useful for converting model predictions back into a readable format -> reconstruct only biometric data and not action data

        Args:
            flat_features (np.ndarray): A 1D NumPy array of numerical features.
            date_str (str): The date string to assign to the reconstructed features.

        Returns:
            Dict[str, Any]: A structured dictionary of features.
        """
        reconstructed_features = {
            'date': date_str,
        }
        sleep_set = {'total_sleep_seconds','deep_sleep_seconds', 'rem_sleep_seconds', 'awake_sleep_seconds', 'restless_moments_count', 'avg_sleep_stress', 'sleep_resting_heart_rate'}

        # Ensure the input array matches the expected size
        if len(flat_features) != self.output_size:
            print(f"Error: Flat features size mismatch. Expected {self.output_size}, got {len(flat_features)}.")
            return reconstructed_features

        for idx,key in enumerate(self.biometrical_keys):
            if key in sleep_set:
                reconstructed_features['sleep_metrics'][key] = flat_features[idx]
            else:
                reconstructed_features[key] = flat_features[idx]

        return reconstructed_features
    
# Example usage (for internal testing of DataProcessor)
if __name__ == "__main__":
    from src.features.feature_engineer import extract_daily_features, create_state_vectors
    from src.data_ingestion.garmin_parser import get_historical_metrics

    print("--- Running DataProcessor in standalone test mode ---")
    NUM_DAYS_TO_FETCH_RAW = config.NUM_DAYS_FOR_INPUT+5 # Need extra day for target

    raw_historical_data = get_historical_metrics(NUM_DAYS_TO_FETCH_RAW)

    if raw_historical_data:
        processed_features = []
        for day_raw_data in raw_historical_data:
            daily_features = extract_daily_features(day_raw_data)
            processed_features.append(daily_features)

        state_vectors = create_state_vectors(processed_features,config.NUM_DAYS_FOR_INPUT)
        processor = DataProcessor()
        X_train_np, y_train_np = processor.prepare_data_for_training(state_vectors)

        print(f"\nShape of X_train (inputs): {X_train_np.shape}")
        print(f"Shape of y_train (targets): {y_train_np.shape}")

        if X_train_np.shape[0] > 0:
            print("\n--- Example Flattened Input for Day 1 of first state vector ---")
            print(X_train_np[0, 0, :]) # First day of first sequence

            print("\n--- Example Flattened Target for first state vector ---")
            print(y_train_np[0, :]) # Target for the first sequence

            # Test reconstruction
            print("\n--- Reconstructed Target Features (first sample) ---")
            reconstructed_target = processor.reconstruct_features_from_flat(y_train_np[0, :], date_str="Predicted Date")
            print(reconstructed_target)

            # Verify the number of features matches the expected output_size
            print(f"\nNumber of features per day (expected by model): {processor.output_size}")
            if X_train_np.shape[2] == processor.output_size:
                print("Feature count matches DataProcessor's expected output_size. Good to go!")
            else:
                print("ERROR: Feature count mismatch. Check DataProcessor's numerical_feature_keys.")
        else:
            print("Not enough data to create training samples. Check NUM_DAYS_TO_FETCH_RAW and NUM_DAYS_FOR_STATE.")
    else:
        print("No raw historical data fetched. Cannot proceed with data preparation.")
