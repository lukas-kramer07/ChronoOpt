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

        # Agent-controlled features (11) — input to environment, decided by agent
        self.agent_feature_keys = [
            'total_steps',
            'activity_Strength', 'activity_Cardio', 'activity_Yoga',
            'activity_Stretching', 'activity_OtherActivity', 'activity_NoActivity',
            'bed_time_gmt_hour', 'bed_time_gmt_minute',
            'wake_time_gmt_hour', 'wake_time_gmt_minute',
        ]

        # Model-predicted features (12) — LSTM predicts these
        self.model_feature_keys = [
            'avg_heart_rate', 'resting_heart_rate', 'avg_respiration_rate',
            'avg_stress', 'body_battery_end_value',
            'total_sleep_seconds', 'deep_sleep_seconds', 'rem_sleep_seconds',
            'awake_sleep_seconds', 'restless_moments_count',
            'avg_sleep_stress', 'sleep_resting_heart_rate',
        ]

        # Full feature vector = agent + model (consistent order)
        self.numerical_feature_keys = self.agent_feature_keys + self.model_feature_keys

        self.input_size = len(self.numerical_feature_keys)   # 23 — LSTM input width
        self.output_size = len(self.model_feature_keys)       # 12 — LSTM output width

        # Separate scalers for input (23 features) and output (12 features)
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self._is_scaler_fitted = False
        print(f"DataProcessor initialized. Input features per day: {self.input_size}, Output (model-predicted) features: {self.output_size}")

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

        # --- AGENT FEATURES FIRST (indices 0-10) ---

        # Steps
        flat_features.append(float(daily_features.get('total_steps', 0)))

        # Activity flags
        activity_flags = daily_features.get('activity_type_flags', {})
        flat_features.append(float(activity_flags.get('Strength', 0)))
        flat_features.append(float(activity_flags.get('Cardio', 0)))
        flat_features.append(float(activity_flags.get('Yoga', 0)))
        flat_features.append(float(activity_flags.get('Stretching', 0)))
        flat_features.append(float(activity_flags.get('OtherActivity', 0)))
        flat_features.append(float(activity_flags.get('NoActivity', 0)))

        # Bed and wake times
        bed_hour, bed_minute = self._convert_timestamp_to_time_features(daily_features.get('bed_time_gmt', 'N/A'))
        wake_hour, wake_minute = self._convert_timestamp_to_time_features(daily_features.get('wake_time_gmt', 'N/A'))
        flat_features.append(float(bed_hour))
        flat_features.append(float(bed_minute))
        flat_features.append(float(wake_hour))
        flat_features.append(float(wake_minute))

        # --- MODEL FEATURES SECOND (indices 11-22) ---

        flat_features.append(daily_features.get('avg_heart_rate', 0.0))
        flat_features.append(daily_features.get('resting_heart_rate', 0.0))
        flat_features.append(daily_features.get('avg_respiration_rate', 0.0))
        flat_features.append(daily_features.get('avg_stress', 0.0))
        flat_features.append(daily_features.get('body_battery_end_value', 0.0))

        sleep_metrics = daily_features.get('sleep_metrics', {})
        flat_features.append(sleep_metrics.get('total_sleep_seconds', 0.0))
        flat_features.append(sleep_metrics.get('deep_sleep_seconds', 0.0))
        flat_features.append(sleep_metrics.get('rem_sleep_seconds', 0.0))
        flat_features.append(sleep_metrics.get('awake_sleep_seconds', 0.0))
        flat_features.append(sleep_metrics.get('restless_moments_count', 0.0))
        flat_features.append(sleep_metrics.get('avg_sleep_stress', 0.0))
        flat_features.append(sleep_metrics.get('resting_heart_rate', 0.0))

        # Check for NaN values before returning
        np_flat_features = np.array(flat_features, dtype=np.float32)
        if np.isnan(np_flat_features).any():
            print(f"Warning: NaN detected in flattened features for date {daily_features.get('date')}. Replacing with 0.")
            np_flat_features[np.isnan(np_flat_features)] = 0.0 # Replace NaNs with 0
        return np_flat_features

    def fit_scaler(self, X_data: np.ndarray, y_data: np.ndarray):
        """
        Fits separate scalers for X (23 features) and y (12 model features).
        Args:
            X_data (np.ndarray): 3D array of shape (num_samples, seq_len, 23)
            y_data (np.ndarray): 2D array of shape (num_samples, 12)
        """
        if X_data.size == 0 or y_data.size == 0:
            print("Warning: Attempted to fit scaler on empty data. Scalers not fitted.")
            return

        # Reshape X to 2D for fitting
        num_samples, seq_len, num_features = X_data.shape
        X_reshaped = X_data.reshape(num_samples * seq_len, num_features)

        print(f"Fitting scaler_X on data of shape {X_reshaped.shape}...")
        self.scaler_X.fit(X_reshaped)

        print(f"Fitting scaler_y on data of shape {y_data.shape}...")
        self.scaler_y.fit(y_data)

        self._is_scaler_fitted = True
        print("Scalers fitted successfully.")

    def transform_X(self, X: np.ndarray) -> np.ndarray:
        """
        Scales X input data (23 features) using scaler_X.
        Args:
            X (np.ndarray): 3D array (num_samples, seq_len, 23) or 2D (num_samples, 23)
        Returns:
            np.ndarray: Scaled array of same shape.
        """
        if not self._is_scaler_fitted:
            print("Error: Scalers not fitted. Returning original data.")
            return X

        original_shape = X.shape
        if X.ndim == 3:
            num_samples, seq_len, num_features = original_shape
            return self.scaler_X.transform(
                X.reshape(num_samples * seq_len, num_features)
            ).reshape(original_shape)
        elif X.ndim == 2:
            return self.scaler_X.transform(X)
        else:
            print(f"Warning: Unsupported dimension {X.ndim}. Returning original.")
            return X

    def transform_y(self, y: np.ndarray) -> np.ndarray:
        """
        Scales y target data (12 model features) using scaler_y.
        Args:
            y (np.ndarray): 2D array (num_samples, 12)
        Returns:
            np.ndarray: Scaled array of same shape.
        """
        if not self._is_scaler_fitted:
            print("Error: Scalers not fitted. Returning original data.")
            return y
        return self.scaler_y.transform(y)
        
    def inverse_transform_X(self, X: np.ndarray) -> np.ndarray:
        """Inverse scales X data (23 features) using scaler_X."""
        if not self._is_scaler_fitted:
            print("Error: Scalers not fitted. Returning original data.")
            return X

        original_shape = X.shape
        if X.ndim == 3:
            num_samples, seq_len, num_features = original_shape
            return self.scaler_X.inverse_transform(
                X.reshape(num_samples * seq_len, num_features)
            ).reshape(original_shape)
        elif X.ndim == 2:
            return self.scaler_X.inverse_transform(X)
        else:
            print(f"Warning: Unsupported dimension {X.ndim}. Returning original.")
            return X

    def inverse_transform_y(self, y: np.ndarray) -> np.ndarray:
        """Inverse scales y data (12 model features) using scaler_y."""
        if not self._is_scaler_fitted:
            print("Error: Scalers not fitted. Returning original data.")
            return y
        return self.scaler_y.inverse_transform(y)

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
                y (np.ndarray): Model-predicted target features for the next day. Shape (num_samples, 12).
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
            y_flat_unscaled.append(next_day_features_flat[len(self.agent_feature_keys):])

        X_np_unscaled = np.array(X_flat_unscaled, dtype=np.float32)
        y_np_unscaled = np.array(y_flat_unscaled, dtype=np.float32)

        # Fit scaler on the entire X_np_unscaled data (reshaped to 2D for fitting)
        # It's important to fit the scaler on the *entire* dataset's range of values.
        # We fit on X_np_unscaled because it represents the input space.
        # The features in y_np_unscaled are the same type of features, just shifted.
        if X_np_unscaled.size > 0:
            self.fit_scaler(X_np_unscaled,y_np_unscaled)

            # Transform both X and y using the fitted scaler
            X_scaled = self.transform_X(X_np_unscaled)
            y_scaled = self.transform_y(y_np_unscaled)
            print("Data scaled successfully.")
            return X_scaled, y_scaled
        else:
            return X_np_unscaled, y_np_unscaled # Return empty if no data

    def reconstruct_features_from_flat(self, flat_features: np.ndarray,
                                    date_str: str = "N/A") -> Dict[str, Any]:
        """
        Reconstructs a structured feature dictionary from a flat numerical array.
        Automatically detects scope based on input length:
            - 12 features → model-only reconstruction (LSTM output), uses scaler_y
            - 23 features → full reconstruction (agent + model), uses scaler_X

        Args:
            flat_features (np.ndarray): 1D array of length 12 or 23.
            date_str (str): Date string. Only populated in the 23-feature case.

        Returns:
            Dict[str, Any]: Structured feature dictionary.
        """
        n = len(flat_features)

        if n == self.output_size:  # 12 features — model output only
            if self._is_scaler_fitted:
                unscaled = self.inverse_transform_y(flat_features.reshape(1, -1))[0]
            else:
                unscaled = flat_features.copy()
            idx = 0

        elif n == self.input_size:  # 23 features — full vector
            if self._is_scaler_fitted:
                unscaled = self.inverse_transform_X(flat_features.reshape(1, -1))[0]
            else:
                unscaled = flat_features.copy()
            idx = len(self.agent_feature_keys)  # Model features start at index 11

        else:
            raise ValueError(f"Unexpected input length: {n}. Expected {self.output_size} or {self.input_size}.")

        # --- Build model result dict (always present, indices 11-22 or 0-11) ---
        result = {'date': date_str, 'sleep_metrics': {}}
        result['avg_heart_rate'] = float(np.clip(unscaled[idx], 0, None)); idx += 1
        result['resting_heart_rate'] = float(np.clip(unscaled[idx], 0, None)); idx += 1
        result['avg_respiration_rate'] = float(np.clip(unscaled[idx], 0, None)); idx += 1
        result['avg_stress'] = float(np.clip(unscaled[idx], 0, 100)); idx += 1
        result['body_battery_end_value'] = float(np.clip(unscaled[idx], 0, 100)); idx += 1
        result['sleep_metrics']['total_sleep_seconds'] = float(np.clip(unscaled[idx], 0, None)); idx += 1
        result['sleep_metrics']['deep_sleep_seconds'] = float(np.clip(unscaled[idx], 0, None)); idx += 1
        result['sleep_metrics']['rem_sleep_seconds'] = float(np.clip(unscaled[idx], 0, None)); idx += 1
        result['sleep_metrics']['awake_sleep_seconds'] = float(np.clip(unscaled[idx], 0, None)); idx += 1
        result['sleep_metrics']['restless_moments_count'] = float(np.clip(unscaled[idx], 0, None)); idx += 1
        result['sleep_metrics']['avg_sleep_stress'] = float(np.clip(unscaled[idx], 0, 100)); idx += 1
        result['sleep_metrics']['resting_heart_rate'] = float(np.clip(unscaled[idx], 0, None)); idx += 1

        # --- Append agent fields if full 23-feature reconstruction ---
        if n == self.input_size:
            agent_idx = 0
            result['total_steps'] = int(round(np.clip(unscaled[agent_idx], 0, None))); agent_idx += 1
            activity_keys = ['Strength', 'Cardio', 'Yoga', 'Stretching', 'OtherActivity', 'NoActivity']
            result['activity_type_flags'] = {}
            for key in activity_keys:
                result['activity_type_flags'][key] = int(round(np.clip(unscaled[agent_idx], 0, 1))); agent_idx += 1
            result['bed_time_gmt'] = f"{int(unscaled[agent_idx]):02d}:{int(unscaled[agent_idx+1]):02d}"; agent_idx += 2
            result['wake_time_gmt'] = f"{int(unscaled[agent_idx]):02d}:{int(unscaled[agent_idx+1]):02d}"

        return result

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

           # Verify feature counts
            print(f"\nInput features per day (LSTM input): {processor.input_size}")
            print(f"Output features per day (LSTM output): {processor.output_size}")
            if X_train_np.shape[2] == processor.input_size and y_train_np.shape[1] == processor.output_size:
                print("Feature counts correct. X has 23, y has 12. Good to go!")
            else:
                print(f"ERROR: Feature count mismatch. X shape: {X_train_np.shape}, y shape: {y_train_np.shape}")
        else:
            print("Not enough data to create training samples. Check NUM_DAYS_TO_FETCH_RAW and NUM_DAYS_FOR_STATE.")
    else:
        print("No raw historical data fetched. Cannot proceed with data preparation.")