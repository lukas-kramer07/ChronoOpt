# src/main.py
# This is the main orchestration script for the ChronoOpt system.
# It handles data ingestion, feature engineering, and training/evaluation
# of the prediction model.

import numpy as np
import torch
import os
from datetime import datetime, timedelta, date
import matplotlib.pyplot as plt # Import matplotlib for plotting
from typing import List,Dict,Any

# Import modules from our project
from src.data_ingestion.garmin_parser import get_historical_metrics
from src.features.feature_engineer import extract_daily_features
from src.models.data_processor import DataProcessor
from src.models.prediction_model import PredictionModel
from src import config # Our configuration file
from src.features.utils import calculate_sleep_score_proxy # For calculating reward later

def plot_predictions(historical_features_dicts: List[Dict[str, Any]],
                     predicted_features_dict: Dict[str, Any],
                     num_days_for_state: int):
    """
    Plots key predicted metrics against historical data.

    Args:
        historical_features_dicts (List[Dict[str, Any]]): List of historical processed feature dictionaries.
        predicted_features_dict (Dict[str, Any]): The single predicted next-day feature dictionary.
        num_days_for_state (int): The number of days used for the state vector.
    """
    print("\n--- Generating Prediction Plots ---")

    # Select 5 different features to plot for clarity
    plot_keys = [
        'total_steps',
        'avg_heart_rate',
        'avg_stress',
        'body_battery_end_value',
        'sleep_metrics.total_sleep_seconds', # Nested key for sleep duration
    ]

    # Prepare data for plotting
    dates = []
    historical_values = {key: [] for key in plot_keys}
    predicted_values = {key: None for key in plot_keys} # Store single predicted value
    
    if len(historical_features_dicts) < num_days_for_state + 1:
        print("Not enough historical data to generate meaningful plots.")
        return

    relevant_historical_data = historical_features_dicts[-(num_days_for_state + 1):]

    for i, day_dict in enumerate(relevant_historical_data):
        dates.append(day_dict['date'])
        for key in plot_keys:
            if '.' in key: # Handle nested keys like 'sleep_metrics.total_sleep_seconds'
                main_key, sub_key = key.split('.')
                val = day_dict.get(main_key, {}).get(sub_key, np.nan)
            else:
                val = day_dict.get(key, np.nan)
            historical_values[key].append(val)

    # Add the predicted date to the dates list
    predicted_date = predicted_features_dict['date']
    dates.append(predicted_date)

    # Extract predicted values
    for key in plot_keys:
        if '.' in key:
            main_key, sub_key = key.split('.')
            predicted_values[key] = predicted_features_dict.get(main_key, {}).get(sub_key, np.nan)
        else:
            predicted_values[key] = predicted_features_dict.get(key, np.nan)

    # Create plots
    num_plots = len(plot_keys)
    fig, axes = plt.subplots(num_plots, 1, figsize=(12, 4 * num_plots), sharex=True)
    if num_plots == 1: # Ensure axes is iterable even for a single plot
        axes = [axes]

    x_indices = np.arange(len(dates)) # Numerical indices for plotting

    for i, key in enumerate(plot_keys):
        ax = axes[i]
        ax.plot(x_indices[:num_days_for_state], historical_values[key][:num_days_for_state], marker='o', linestyle='-', color='blue', label='Historical Actual')
        ax.plot(x_indices[num_days_for_state], historical_values[key][num_days_for_state], marker='o', color='green', markersize=8, label='Actual Next Day')
        ax.plot(x_indices[-1], predicted_values[key], marker='x', color='red', markersize=10, label='Predicted Next Day')

        ax.set_title(f'{key.replace("_", " ").title()}')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True)

    plt.xticks(x_indices, dates, rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def run_prediction_pipeline():
    """
    Orchestrates the data pipeline for the prediction model:
    1. Fetches historical raw data.
    2. Engineers features for each day.
    3. Prepares data into (X, y) format for the PyTorch model.
    4. Initializes and trains the PredictionModel.
    5. Evaluates the trained model (with per-metric MAE).
    6. Performs an example prediction.
    """
    print("--- Starting ChronoOpt Prediction Pipeline ---")

    # --- 1. Data Ingestion ---
    print(f"\nCollecting historical data for the last {config.NUM_DAYS_TO_FETCH_RAW} days...")
    raw_historical_data = get_historical_metrics(config.NUM_DAYS_TO_FETCH_RAW, end_date=date(2025,12,24))

    if not raw_historical_data:
        print("Error: No raw historical data fetched. Exiting pipeline.")
        return

    # --- 2. Feature Engineering ---
    print(f"\nProcessing {len(raw_historical_data)} days of raw data into features...")
    processed_features = []
    for day_raw_data in raw_historical_data:
        daily_features = extract_daily_features(day_raw_data)
        processed_features.append(daily_features)
    print(f"Successfully processed {len(processed_features)} days of features.")

    # --- 3. Data Preparation for Model ---
    print("\nPreparing data for the prediction model...")
    data_processor = DataProcessor()

    # X: (num_samples, sequence_length, num_features_per_day)
    # y: (num_samples, num_features_per_day)
    X_data, y_data = data_processor.prepare_data_for_training(
        processed_features, config.NUM_DAYS_FOR_STATE
    )

    if X_data.shape[0] == 0:
        print("Error: Not enough data to create training samples. Exiting pipeline.")
        return

    print(f"Data prepared: X_data shape {X_data.shape}, y_data shape {y_data.shape}")

    # Dynamically set input_size and output_size for the model
    model_input_size = data_processor.input_size    
    model_output_size = data_processor.output_size 

    # Update model hyperparameters with dynamically determined sizes
    model_params = config.MODEL_HYPERPARAMETERS.copy()
    model_params['input_size'] = model_input_size
    model_params['output_size'] = model_output_size

    # --- 4. Model Initialization ---
    print("\nInitializing Prediction Model...")
    model = PredictionModel(
        input_size=model_params['input_size'],
        hidden_size=model_params['hidden_size'],
        output_size=model_params['output_size'],
        num_layers=model_params['num_layers']
    )

    # --- 5. Model Training ---
    print("\nTraining Prediction Model...")
    model.train_model(
        X_data, y_data,
        epochs=model_params['epochs'],
        batch_size=model_params['batch_size'],
        learning_rate=model_params['learning_rate'],
        validation_split=model_params['validation_split'],
        patience=model_params['patience'],
        lr_scheduler_factor=model_params['lr_scheduler_factor'],
        lr_scheduler_patience=model_params['lr_scheduler_patience']
    )

    # --- 6. Model Evaluation (on a subset of the data, typically the validation split) ---
    print("\nEvaluating Prediction Model...")
    num_test_samples = max(1, int(X_data.shape[0] * 0.1)) # Use 10% of data as test
    X_test_scaled = X_data[-num_test_samples:]
    y_test_scaled = y_data[-num_test_samples:]

    if X_test_scaled.shape[0] > 0:
        # Pass the feature names from the data_processor for MAE calculation
        evaluation_metrics = model.evaluate_model(X_test_scaled, y_test_scaled, data_processor.model_feature_keys)
        print(f"Prediction Model Evaluation Results: Overall MSE: {evaluation_metrics['overall_mse']:.4f}")
        # Individual MAEs are printed within evaluate_model function
    else:
        print("Not enough data to perform a separate evaluation.")


    # --- Example Prediction & Sleep Score Calculation ---
    print("\n--- Example Prediction for the next day ---")
    if X_data.shape[0] > 0:
        # Take the last state vector from X_data to predict the very next day
        last_state_vector_scaled = X_data[-1].reshape(1, config.NUM_DAYS_FOR_STATE, model_input_size)
        predicted_flat_features_scaled = model.predict(last_state_vector_scaled)

        # Reconstruct the predicted features into a structured dictionary (inverse transform first)
        last_processed_date_str = processed_features[-1]['date']
        last_processed_date = datetime.strptime(last_processed_date_str, "%Y-%m-%d").date()
        predicted_date = (last_processed_date + timedelta(days=1)).strftime("%Y-%m-%d")

        predicted_structured_features = data_processor.reconstruct_features_from_flat(
            predicted_flat_features_scaled[0], date_str=predicted_date
        )
        print(f"Predicted features for {predicted_date}:")
        print(f"  Avg HR: {predicted_structured_features['avg_heart_rate']:.1f}")
        print(f"  Avg Stress: {predicted_structured_features['avg_stress']:.1f}")
        print(f"  Body Battery End: {predicted_structured_features['body_battery_end_value']:.1f}")
        print(f"  Sleep Metrics: {predicted_structured_features['sleep_metrics']}")

        # Calculate the sleep score proxy from the predicted sleep metrics
        predicted_sleep_score = calculate_sleep_score_proxy(predicted_structured_features['sleep_metrics'])
        print(f"\nPredicted Sleep Score Proxy for {predicted_date}: {predicted_sleep_score:.2f}")

        # --- Plotting Predictions ---
        # We need the historical features (unscaled) and the predicted features (unscaled)
        plot_predictions(processed_features, predicted_structured_features, config.NUM_DAYS_FOR_STATE)

    else:
        print("Not enough data to make an example prediction.")

    print("\n--- ChronoOpt Prediction Pipeline Complete ---")


if __name__ == "__main__":
    os.makedirs('src/models/saved_models', exist_ok=True)
    run_prediction_pipeline()
