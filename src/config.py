# src/config.py
# This file stores global configuration parameters for the ChronoOpt system.
# All hyperparameters for models and agents are defined directly here.


# --- Data Ingestion & Feature Engineering ---
# Number of historical days to fetch from Garmin Connect.
# Ensure this is at least NUM_DAYS_FOR_STATE + 1 for training data generation.
NUM_DAYS_TO_FETCH_RAW = 450 

# Number of past days to include in each state vector (X in the RL context).
NUM_DAYS_FOR_STATE = 10 

LSTM_MODEL_SAVE_PATH = "src/models/saved_models/lstm_chronoopt.pt"
TRAINING_END_DATE = "2025-12-01"

# --- Prediction Model Hyperparameters ---
# These parameters are for the PyTorch PredictionModel (LSTM).
MODEL_HYPERPARAMETERS = {
    'input_size': None, # This will be determined dynamically by DataProcessor.output_size
    'hidden_size': 32,  #=
    'output_size': None, # This will be determined dynamically by DataProcessor.output_size
    'num_layers': 1,   
    'epochs': 200,      
    'batch_size': 64,   
    'learning_rate': 0.002, 
    'validation_split': 0.1, 
    'patience': 10,    
    'lr_scheduler_factor':  0.5, 
    'lr_scheduler_patience': 10, 
}

# --- EDMD ---
EDMD_MODEL_SAVE_PATH      = "src/models/saved_models/edmd_chronoopt.pkl"
EDMD_DEGREE               = 1      # polynomial lifting degree
EDMD_ALPHA                = 100.0     # Ridge regularization
EDMD_SYNTHETIC_VARIANTS   = 0      # synthetic action variants per real day
USE_EDMD_ENV              = True  
USE_PREDICTION_CONSTRAINTS  = True

# --- Reinforcement Learning ---
USE_DETERMINISTIC_ENV = False

RL_AGENT_PARAMETERS = {
    'gamma': 0.99, # Discount factor
    'epsilon_start': 1.0,
    'epsilon_end': 0.01,
    'epsilon_decay': 0.995,
    'buffer_size': 10000,
    'learning_rate': 0.0003,
    'target_update_frequency': 10,
}

PPO_HYPERPARAMETERS = {
    'lr': 1.5e-5,
    'gamma': 0.99,
    'lam': 0.95,
    'clip_eps': 0.2,
    'c1': 0.5,
    'c2': 0.05,
    'n_steps': 2048,
    'k_epochs': 4,
    'batch_size': 64,
    'max_grad_norm': 0.5,
    'total_steps': 1_500_000,
    'log_interval': 10,
}

POLICY_SAVE_PATH = "src/rl_agent/saved_policies/reinforce_policy.pt"

RL_TRAIN_NUM_EPISODES = 500
RL_TRAIN_MAX_STEPS = 5
RL_TRAIN_LOG_INTERVAL = 10

# --- Other Global Settings ---
# Seed for reproducibility
RANDOM_SEED = 42
