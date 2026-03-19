# src/models/data_processor.py
# This module handles the transformation of daily feature dictionaries
# into numerical tensors suitable for PyTorch models, including data scaling.

from typing import List, Dict, Any, Tuple
import numpy as np
from datetime import datetime,timedelta
from sklearn.preprocessing import StandardScaler # Changed from MinMaxScaler

class DataProcessor:
    """
    Handles the conversion of structured daily feature dictionaries into
    flat numerical arrays (for model input) and back (for interpretation).
    It also manages feature scaling (StandardScaler normalization).
    """
    def __init__(self):
        # Define the order of numerical features for a single day.
        # This order MUST be consistent across all data processing.
        self.numerical_feature_keys = [
            'total_steps',
            'avg_heart_rate',
            'resting_heart_rate',
            'avg_respiration_rate',
            'avg_stress',
            'body_battery_end_value',
            # Sleep Metrics (flattened)
            'total_sleep_seconds',
            'deep_sleep_seconds',
            'rem_sleep_seconds',
            'awake_sleep_seconds',
            'restless_moments_count',
            'avg_sleep_stress',
            'sleep_resting_heart_rate', # Note: 'resting_heart_rate' from sleep_metrics
            # Activity Type Flags (one-hot encoded)
            'activity_Strength',
            'activity_Cardio',
            'activity_Yoga',
            'activity_Stretching',
            'activity_OtherActivity',
            'activity_NoActivity',
            # Time features (numerical representation of bed/wake times)
            'bed_time_gmt_hour',
            'bed_time_gmt_minute',
            'wake_time_gmt_hour',
            'wake_time_gmt_minute',
        ]
        self.output_size = len(self.numerical_feature_keys)
        self.scaler = StandardScaler() # Changed to StandardScaler
        self._is_scaler_fitted = False
        print(f"DataProcessor initialized. Total numerical features per day: {self.output_size}")

    def create_state_vectors(self,historical_daily_features: List[Dict[str, Any]], num_days_in_state: int) -> List[Dict[str, Any]]:
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
            return 0, 0

    def flatten_features_for_day(self, daily_features: Dict[str, Any]) -> np.ndarray:
        """
        Converts a single day's structured feature dictionary into a flat numerical NumPy array.
        Ensures consistent order as defined in self.numerical_feature_keys.
        """
        flat_features = []

        # Process top-level numerical features
        flat_features.append(daily_features.get('total_steps', 0))
        flat_features.append(daily_features.get('avg_heart_rate', 0.0))
        flat_features.append(daily_features.get('resting_heart_rate', 0.0))
        flat_features.append(daily_features.get('avg_respiration_rate', 0.0))
        flat_features.append(daily_features.get('avg_stress', 0.0))
        flat_features.append(daily_features.get('body_battery_end_value', 0.0))

        # Process sleep_metrics
        sleep_metrics = daily_features.get('sleep_metrics', {})
        flat_features.append(sleep_metrics.get('total_sleep_seconds', 0.0))
        flat_features.append(sleep_metrics.get('deep_sleep_seconds', 0.0))
        flat_features.append(sleep_metrics.get('rem_sleep_seconds', 0.0))
        flat_features.append(sleep_metrics.get('awake_sleep_seconds', 0.0))
        flat_features.append(sleep_metrics.get('restless_moments_count', 0.0))
        flat_features.append(sleep_metrics.get('avg_sleep_stress', 0.0))
        flat_features.append(sleep_metrics.get('resting_heart_rate', 0.0)) # From sleep_metrics

        # Process activity_type_flags (one-hot encoded)
        activity_flags = daily_features.get('activity_type_flags', {})
        flat_features.append(float(activity_flags.get('Strength', 0)))
        flat_features.append(float(activity_flags.get('Cardio', 0)))
        flat_features.append(float(activity_flags.get('Yoga', 0)))
        flat_features.append(float(activity_flags.get('Stretching', 0)))
        flat_features.append(float(activity_flags.get('OtherActivity', 0)))
        flat_features.append(float(activity_flags.get('NoActivity', 0)))

        # Process time features
        bed_hour, bed_minute = self._convert_timestamp_to_time_features(daily_features.get('bed_time_gmt', 'N/A'))
        wake_hour, wake_minute = self._convert_timestamp_to_time_features(daily_features.get('wake_time_gmt', 'N/A'))
        flat_features.append(float(bed_hour))
        flat_features.append(float(bed_minute))
        flat_features.append(float(wake_hour))
        flat_features.append(float(wake_minute))

        # Check for NaN values before returning
        np_flat_features = np.array(flat_features, dtype=np.float32)
        if np.isnan(np_flat_features).any():
            print(f"Warning: NaN detected in flattened features for date {daily_features.get('date')}. Replacing with 0.")
            np_flat_features[np.isnan(np_flat_features)] = 0.0 # Replace NaNs with 0
        return np_flat_features

    def fit_scaler(self, data_to_fit: np.ndarray):
        """
        Fits the StandardScaler on the provided data.
        This should be called once on the training data before scaling.
        Args:
            data_to_fit (np.ndarray): A 2D array (num_samples, num_features) to fit the scaler on.
        """
        if data_to_fit.size == 0:
            print("Warning: Attempted to fit scaler on empty data. Scaler not fitted.")
            return

        # Reshape data_to_fit to 2D if it's 3D (e.g., (num_samples, seq_len, num_features))
        # StandardScaler expects 2D input (n_samples, n_features)
        if data_to_fit.ndim == 3:
            num_samples, seq_len, num_features = data_to_fit.shape
            reshaped_data = data_to_fit.reshape(num_samples * seq_len, num_features)
        else: # Assumes 2D (num_samples, num_features)
            reshaped_data = data_to_fit

        print(f"Fitting scaler on data of shape {reshaped_data.shape}...")
        self.scaler.fit(reshaped_data)
        self._is_scaler_fitted = True
        print("Scaler fitted successfully.")

    def transform_data(self, data_to_transform: np.ndarray) -> np.ndarray:
        """
        Transforms the provided data using the fitted StandardScaler.
        Args:
            data_to_transform (np.ndarray): A 2D or 3D array to transform.
        Returns:
            np.ndarray: The scaled data.
        """
        if not self._is_scaler_fitted:
            print("Error: Scaler not fitted. Cannot transform data. Returning original data.")
            return data_to_transform

        original_ndim = data_to_transform.ndim
        original_shape = data_to_transform.shape

        if original_ndim == 3:
            num_samples, seq_len, num_features = original_shape
            reshaped_data = data_to_transform.reshape(num_samples * seq_len, num_features)
            transformed_data = self.scaler.transform(reshaped_data)
            return transformed_data.reshape(original_shape)
        elif original_ndim == 2:
            return self.scaler.transform(data_to_transform)
        else:
            print(f"Warning: Unsupported data dimension for transform: {original_ndim}. Returning original data.")
            return data_to_transform

    def inverse_transform_data(self, transformed_data: np.ndarray) -> np.ndarray:
        """
        Inverse transforms the data using the fitted StandardScaler.
        Args:
            transformed_data (np.ndarray): A 2D or 3D array of scaled data.
        Returns:
            np.ndarray: The original scale data.
        """
        if not self._is_scaler_fitted:
            print("Error: Scaler not fitted. Cannot inverse transform data. Returning original data.")
            return transformed_data

        original_ndim = transformed_data.ndim
        original_shape = transformed_data.shape

        if original_ndim == 3:
            num_samples, seq_len, num_features = original_shape
            reshaped_data = transformed_data.reshape(num_samples * seq_len, num_features)
            inverse_transformed_data = self.scaler.inverse_transform(reshaped_data)
            return inverse_transformed_data.reshape(original_shape)
        elif original_ndim == 2:
            return self.scaler.inverse_transform(transformed_data)
        else:
            print(f"Warning: Unsupported data dimension for inverse transform: {original_ndim}. Returning original data.")
            return transformed_data

    def prepare_data_for_training(self, historical_daily_features: List[Dict[str, Any]], num_days_in_state: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepares the historical data into (X, y) pairs for model training.
        Applies scaling after flattening.

        Args:
            historical_daily_features (List[Dict[str, Any]]): A list of standardized daily feature
                                                              dictionaries, sorted oldest to newest.
            num_days_in_state (int): The number of past days to include in each state vector (X).

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                X (np.ndarray): Input sequences. Shape (num_samples, num_days_in_state, num_features_per_day).
                y (np.ndarray): Target features for the next day. Shape (num_samples, num_features_per_day).
        """
        X_flat_unscaled, y_flat_unscaled = [], []

        if len(historical_daily_features) < num_days_in_state + 1: # Need X days for input, and 1 day for target
            print(f"Warning: Not enough historical data ({len(historical_daily_features)} days) to create "
                  f"training samples for {num_days_in_state} days in state. Need at least {num_days_in_state + 1} days.")
            return np.array([]), np.array([])

        for i in range(len(historical_daily_features) - num_days_in_state):
            # Input sequence (X days)
            current_state_sequence_dicts = historical_daily_features[i : i + num_days_in_state]
            current_state_sequence_flat = [self.flatten_features_for_day(d) for d in current_state_sequence_dicts]
            X_flat_unscaled.append(current_state_sequence_flat)

            # Target (next day's features)
            next_day_features_dict = historical_daily_features[i + num_days_in_state]
            next_day_features_flat = self.flatten_features_for_day(next_day_features_dict)
            y_flat_unscaled.append(next_day_features_flat)

        X_np_unscaled = np.array(X_flat_unscaled, dtype=np.float32)
        y_np_unscaled = np.array(y_flat_unscaled, dtype=np.float32)

        # Fit scaler on the entire X_np_unscaled data (reshaped to 2D for fitting)
        # It's important to fit the scaler on the *entire* dataset's range of values.
        # We fit on X_np_unscaled because it represents the input space.
        # The features in y_np_unscaled are the same type of features, just shifted.
        if X_np_unscaled.size > 0:
            self.fit_scaler(X_np_unscaled)

            # Transform both X and y using the fitted scaler
            X_scaled = self.transform_data(X_np_unscaled)
            y_scaled = self.transform_data(y_np_unscaled) # Apply same scaling to targets
            print("Data scaled successfully.")
            return X_scaled, y_scaled
        else:
            return X_np_unscaled, y_np_unscaled # Return empty if no data


    def reconstruct_features_from_flat(self, flat_features: np.ndarray, date_str: str = "N/A") -> Dict[str, Any]:
        """
        Reconstructs a structured feature dictionary from a flat numerical NumPy array.
        This is useful for converting model predictions back into a readable format.
        Applies inverse scaling before reconstruction.

        Args:
            flat_features (np.ndarray): A 1D NumPy array of numerical features (scaled).
            date_str (str): The date string to assign to the reconstructed features.

        Returns:
            Dict[str, Any]: A structured dictionary of features.
        """
        if not self._is_scaler_fitted:
            print("Error: Scaler not fitted. Cannot inverse transform for reconstruction. Returning unscaled.")
            # If scaler not fitted, assume input is unscaled and proceed
            unscaled_features = flat_features.copy()
        else:
            # Inverse transform the flat features before reconstructing
            # Need to reshape 1D array to 2D (1 sample, num_features) for inverse_transform
            unscaled_features = self.inverse_transform_data(flat_features.reshape(1, -1))[0] # Get the first (and only) row

        reconstructed_features = {
            'date': date_str,
            'total_steps': 0,
            'avg_heart_rate': 0.0,
            'resting_heart_rate': 0.0,
            'avg_respiration_rate': 0.0,
            'avg_stress': 0.0,
            'body_battery_end_value': 0.0,
            'activity_type_flags': {
                'Strength': 0, 'Cardio': 0, 'Yoga': 0, 'Stretching': 0,
                'OtherActivity': 0, 'NoActivity': 0
            },
            'sleep_metrics': {
                'total_sleep_seconds': 0.0, 'deep_sleep_seconds': 0.0,
                'rem_sleep_seconds': 0.0, 'awake_sleep_seconds': 0.0,
                'restless_moments_count': 0.0, 'avg_sleep_stress': 0.0,
                'resting_heart_rate': 0.0
            },
            'wake_time_gmt': 'N/A',
            'bed_time_gmt': 'N/A',
        }

        # Ensure the input array matches the expected size
        if len(unscaled_features) != self.output_size:
            print(f"Error: Flat features size mismatch. Expected {self.output_size}, got {len(unscaled_features)}.")
            return reconstructed_features

        # Map flat features back to structured dictionary
        idx = 0
        reconstructed_features['total_steps'] = int(round(np.clip(unscaled_features[idx], 0, None))); idx += 1 # Clip at 0 for steps
        reconstructed_features['avg_heart_rate'] = float(np.clip(unscaled_features[idx], 0, None)); idx += 1
        reconstructed_features['resting_heart_rate'] = float(np.clip(unscaled_features[idx], 0, None)); idx += 1
        reconstructed_features['avg_respiration_rate'] = float(np.clip(unscaled_features[idx], 0, None)); idx += 1
        reconstructed_features['avg_stress'] = float(np.clip(unscaled_features[idx], 0, 100)); idx += 1 # Stress 0-100
        reconstructed_features['body_battery_end_value'] = float(np.clip(unscaled_features[idx], 0, 100)); idx += 1 # Body Battery 0-100

        # Sleep Metrics
        reconstructed_features['sleep_metrics']['total_sleep_seconds'] = float(np.clip(unscaled_features[idx], 0, None)); idx += 1
        reconstructed_features['sleep_metrics']['deep_sleep_seconds'] = float(np.clip(unscaled_features[idx], 0, None)); idx += 1
        reconstructed_features['sleep_metrics']['rem_sleep_seconds'] = float(np.clip(unscaled_features[idx], 0, None)); idx += 1
        reconstructed_features['sleep_metrics']['awake_sleep_seconds'] = float(np.clip(unscaled_features[idx], 0, None)); idx += 1
        reconstructed_features['sleep_metrics']['restless_moments_count'] = float(np.clip(unscaled_features[idx], 0, None)); idx += 1
        reconstructed_features['sleep_metrics']['avg_sleep_stress'] = float(np.clip(unscaled_features[idx], 0, 100)); idx += 1 # Sleep Stress 0-100
        reconstructed_features['sleep_metrics']['resting_heart_rate'] = float(np.clip(unscaled_features[idx], 0, None)); idx += 1

        # Activity Type Flags
        activity_keys = ['Strength', 'Cardio', 'Yoga', 'Stretching', 'OtherActivity'] # Exclude NoActivity for direct prediction
        predicted_activities_flags = {}
        for key in activity_keys:
            # Clip to [0,1] and round to nearest integer (0 or 1)
            predicted_activities_flags[key] = int(round(np.clip(unscaled_features[idx], 0, 1))); idx += 1

        if any(predicted_activities_flags.values()):
            reconstructed_features['activity_type_flags']['NoActivity'] = 0
        else:
            reconstructed_features['activity_type_flags']['NoActivity'] = 1

        # Assign the predicted specific activities
        for key in activity_keys:
            reconstructed_features['activity_type_flags'][key] = predicted_activities_flags[key]

        # Time features as numerical values
        bed_hour, bed_minute = int(flat_features[idx]), int(flat_features[idx+1]); idx += 2
        wake_hour, wake_minute = int(flat_features[idx]), int(flat_features[idx+1]); idx += 2
        reconstructed_features['bed_time_gmt'] = f"{bed_hour:02d}:{bed_minute:02d}" # Example string representation
        reconstructed_features['wake_time_gmt'] = f"{wake_hour:02d}:{wake_minute:02d}"

        return reconstructed_features

# Example usage (for internal testing of DataProcessor)
if __name__ == "__main__":
    from src.features.feature_engineer import extract_daily_features # Import here for standalone test
    from src.data_ingestion.garmin_parser import get_historical_metrics # Import here for standalone test
    from datetime import date # Import date for consistent testing

    print("--- Running DataProcessor in standalone test mode ---")
    NUM_DAYS_FOR_STATE = 7
    NUM_DAYS_TO_FETCH_RAW = NUM_DAYS_FOR_STATE + 5 # Need extra day for target

    # Determine the end date for data fetching (yesterday) for consistent testing
    end_date_for_test = date.today() - timedelta(days=1)
    raw_historical_data = get_historical_metrics(NUM_DAYS_TO_FETCH_RAW, end_date=end_date_for_test)

    if raw_historical_data:
        processed_features = []
        for day_raw_data in raw_historical_data:
            daily_features = extract_daily_features(day_raw_data)
            processed_features.append(daily_features)

        processor = DataProcessor()
        X_train_np, y_train_np = processor.prepare_data_for_training(processed_features, NUM_DAYS_FOR_STATE)

        print(f"\nShape of X_train (inputs): {X_train_np.shape}")
        print(f"Shape of y_train (targets): {y_train_np.shape}")

        if X_train_np.shape[0] > 0:
            print("\n--- Example Flattened Input for Day 1 of first state vector (Scaled) ---")
            print(X_train_np[0, 0, :]) # First day of first sequence

            print("\n--- Example Flattened Target for first state vector (Scaled) ---")
            print(y_train_np[0, :]) # Target for the first sequence

            # Test reconstruction (inverse transform)
            print("\n--- Reconstructed Target Features (first sample, Unscaled) ---")
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
