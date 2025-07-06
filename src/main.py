# src/main.py
# This is the main orchestration script for the ChronoOpt system.
# It handles data ingestion, feature engineering, and training/evaluation
# of the prediction model.

from datetime import datetime,timedelta
import numpy as np
import torch
import os # For saving models

# Import modules from our project
from src.data_ingestion.garmin_parser import get_historical_metrics
from src.features.feature_engineer import extract_daily_features
from src.models.data_processor import DataProcessor
from src.models.prediction_model import PredictionModel
from src import config # Our configuration file
from src.features.utils import calculate_sleep_score_proxy # For calculating reward later

def run_prediction_pipeline():
    """
    Orchestrates the data pipeline for the prediction model:
    1. Fetches historical raw data.
    2. Engineers features for each day.
    3. Prepares data into (X, y) format for the PyTorch model.
    4. Initializes and trains the PredictionModel.
    5. Evaluates the trained model.
    """
    print("--- Starting ChronoOpt Prediction Pipeline ---")

    # --- 1. Data Ingestion ---
    print(f"\nCollecting historical data for the last {config.NUM_DAYS_TO_FETCH_RAW} days...")
    raw_historical_data = get_historical_metrics(config.NUM_DAYS_TO_FETCH_RAW)

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
    model_input_size = data_processor.output_size # Number of features per day
    model_output_size = data_processor.output_size # Predicting all features for the next day

    # Update model hyperparameters with dynamically determined sizes
    model_params = config.MODEL_HYPERPARAMETERS.copy()
    model_params['input_size'] = model_input_size
    model_params['output_size'] = model_output_size

    # --- 4. Model Initialization ---
    print("\nInitializing Prediction Model...")
    # Pass the determined input_size and output_size to the model constructor
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
        patience=model_params['patience']
    )

    # --- 6. Model Evaluation (on a subset of the data, typically the validation split) ---
    print("\nEvaluating Prediction Model...")
    # For evaluation, we can reuse the test split from the train_model function
    num_test_samples = max(1, int(X_data.shape[0] * 0.1)) # Use 10% of data as test
    X_test = X_data[-num_test_samples:]
    y_test = y_data[-num_test_samples:]

    if X_test.shape[0] > 0:
        evaluation_metrics = model.evaluate_model(X_test, y_test)
        print(f"Prediction Model Evaluation Results: {evaluation_metrics}")
    else:
        print("Not enough data to perform a separate evaluation.")


    # --- Example Prediction & Sleep Score Calculation ---
    print("\n--- Example Prediction for the next day ---")
    if X_data.shape[0] > 0:
        # Take the last state vector from X_data to predict the very next day
        last_state_vector = X_data[-1].reshape(1, config.NUM_DAYS_FOR_STATE, model_input_size)
        predicted_flat_features = model.predict(last_state_vector)

        # Reconstruct the predicted features into a structured dictionary
        # We need to determine the date for this predicted day.
        # Get the date of the last day in the last state vector
        last_day_date_str = processed_features[-1]['date']
        last_day_date = datetime.strptime(last_day_date_str, "%Y-%m-%d").date()
        predicted_date = (last_day_date + timedelta(days=1)).strftime("%Y-%m-%d")

        predicted_structured_features = data_processor.reconstruct_features_from_flat(
            predicted_flat_features[0], date_str=predicted_date
        )
        print(f"Predicted features for {predicted_date}:")
        # Print a subset for readability
        print(f"  Total Steps: {predicted_structured_features['total_steps']:.0f}")
        print(f"  Avg HR: {predicted_structured_features['avg_heart_rate']:.1f}")
        print(f"  Avg Stress: {predicted_structured_features['avg_stress']:.1f}")
        print(f"  Body Battery End: {predicted_structured_features['body_battery_end_value']:.1f}")
        print(f"  Activities: {predicted_structured_features['activity_type_flags']}")
        print(f"  Sleep Metrics: {predicted_structured_features['sleep_metrics']}")

        # Calculate the sleep score proxy from the predicted sleep metrics
        predicted_sleep_score = calculate_sleep_score_proxy(predicted_structured_features['sleep_metrics'])
        print(f"\nPredicted Sleep Score Proxy for {predicted_date}: {predicted_sleep_score:.2f}")
    else:
        print("Not enough data to make an example prediction.")

    print("\n--- ChronoOpt Prediction Pipeline Complete ---")


if __name__ == "__main__":
    # Ensure the 'saved_models' directory exists
    os.makedirs('src/models/saved_models', exist_ok=True)
    run_prediction_pipeline()
