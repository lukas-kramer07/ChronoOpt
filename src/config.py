# src/config.py
# This file stores global configuration parameters for the ChronoOpt system.
# All hyperparameters for models and agents are defined directly here.


# --- Data Ingestion & Feature Engineering ---
# Number of historical days to fetch from Garmin Connect.
# Ensure this is at least NUM_DAYS_FOR_STATE + 1 for training data generation.
NUM_DAYS_TO_FETCH_RAW = 200 

# Number of past days to include in each state vector (X in the RL context).
NUM_DAYS_FOR_STATE = 7 

# --- Prediction Model Hyperparameters ---
# These parameters are for the PyTorch PredictionModel (LSTM).
MODEL_HYPERPARAMETERS = {
    'input_size': None, # This will be determined dynamically by DataProcessor.output_size
    'hidden_size': 64,  # Number of features in the LSTM's hidden state
    'output_size': None, # This will be determined dynamically by DataProcessor.output_size
    'num_layers': 2,    # Number of stacked LSTM layers
    'epochs': 1_000_000,      # Number of training epochs
    'batch_size': 32,   # Batch size for training
    'learning_rate': 0.01, # Learning rate for the optimizer
    'validation_split': 0.2, # Fraction of data for validation
    'patience': 500,     # Early stopping patience
}

ACTION_KEYS = [ 
    'bed_time_gmt_hour', 
    'bed_time_gmt_minute', 
    'wake_time_gmt_hour', 
    'wake_time_gmt_minute',
    'total_steps', 
    'activity_Strength', 
    'activity_Cardio', 
    'activity_Yoga', 
    'activity_Stretching', 
    'activity_OtherActivity', 
    'activity_NoActivity',
]

BIOMETRIC_KEYS = [
    'avg_heart_rate',
    'resting_heart_rate',
    'avg_respiration_rate', 
    'avg_stress', 
    'body_battery_end_value', 
    'total_sleep_seconds',
    'deep_sleep_seconds', 
    'rem_sleep_seconds', 
    'awake_sleep_seconds', 
    'restless_moments_count', 
    'avg_sleep_stress', 
    'sleep_resting_heart_rate']

# --- Reinforcement Learning ---
RL_AGENT_PARAMETERS = {
    'gamma': 0.99, # Discount factor
    'epsilon_start': 1.0,
    'epsilon_end': 0.01,
    'epsilon_decay': 0.995,
    'buffer_size': 10000,
    'learning_rate': 0.0005,
    'target_update_frequency': 10,
}

# --- Other Global Settings ---
# Seed for reproducibility
RANDOM_SEED = 42
