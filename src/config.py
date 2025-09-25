# src/config.py
# This file stores global configuration parameters for the ChronoOpt system.
# All hyperparameters for models and agents are defined directly here.


# --- Data Ingestion & Feature Engineering ---
# Number of historical days to fetch from Garmin Connect.
# Ensure this is at least NUM_DAYS_FOR_STATE + 1 for training data generation.
NUM_DAYS_TO_FETCH_RAW = 200 

# Number of past days to include in each state vector (X in the RL context).
NUM_DAYS_FOR_INPUT= 7

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

# the standard features to extract from the raw data
STANDARD_FEATURES = {
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
    'bed_time_gmt': 'N/A'}