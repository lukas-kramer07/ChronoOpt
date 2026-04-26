# src/rl_agent/train_agent.py
# Wiring script for training the REINFORCE agent against ChronoOptEnv
# using the pre-trained LSTM prediction model.
#
# Run from project root:
#   python3 -m src.rl_agent.train_agent

import torch
import numpy as np
from datetime import date

from src import config
from src.models.prediction_model import PredictionModel
from src.models.data_processor import DataProcessor
from src.features.feature_engineer import extract_daily_features
from src.data_ingestion.garmin_parser import get_historical_metrics
from src.rl_agent.rl_environment import ChronoOptEnv
from src.rl_agent.policy_network import PolicyNetwork
from src.rl_agent.agent import ReinforceAgent
from src.rl_agent.deterministic_environment import DeterministicEnv


def load_trained_lstm(device: torch.device) -> PredictionModel:
    """Loads the pre-trained LSTM from disk."""
    return PredictionModel.load(config.LSTM_MODEL_SAVE_PATH, device)


def build_fitted_processor() -> tuple[DataProcessor, list]:
    """
    Replays the data pipeline to fit the processor scalers and returns
    the fitted processor alongside the full list of processed feature dicts.
    All data is served from local cache — no Garmin API calls.
    """
    training_end_date = date.fromisoformat(config.LSTM_TRAINING_END_DATE)
    raw_data = get_historical_metrics(config.NUM_DAYS_TO_FETCH_RAW, end_date=training_end_date)
    if not raw_data:
        raise RuntimeError("Failed to fetch historical data. Check cache.")

    processed_features = [extract_daily_features(d) for d in raw_data]

    processor = DataProcessor()
    processor.prepare_data_for_training(processed_features, config.NUM_DAYS_FOR_STATE)
    # Side effect: scalers are now fitted.

    return processor, processed_features


def build_initial_state(processed_features: list, processor: DataProcessor) -> np.ndarray:
    """
    Builds the initial (seq_len, 23) unscaled state array for ChronoOptEnv
    from the last NUM_DAYS_FOR_STATE days of the training window.

    Args:
        processed_features: Full list of processed feature dicts, oldest to newest.
        processor: Fitted DataProcessor.

    Returns:
        np.ndarray: Shape (NUM_DAYS_FOR_STATE, 23), unscaled.
    """
    last_n = processed_features[-config.NUM_DAYS_FOR_STATE:]
    state = np.array(
        [processor.flatten_features_for_day(d) for d in last_n],
        dtype=np.float32
    )
    return state


def run_rl_training():
    """
    Full RL training pipeline:
    1. Load trained LSTM
    2. Fit DataProcessor (replay from cache)
    3. Build initial environment state
    4. Instantiate ChronoOptEnv, PolicyNetwork, ReinforceAgent
    5. Train agent
    6. Save policy
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- 1. Load LSTM ---
    print("\nLoading trained LSTM...")
    model = load_trained_lstm(device)

    # --- 2. Fit processor ---
    print("\nFitting DataProcessor from cache...")
    processor, processed_features = build_fitted_processor()

    # --- 3. Build initial state ---
    print("\nBuilding initial environment state...")
    initial_state = build_initial_state(processed_features, processor)
    print(f"Initial state shape: {initial_state.shape}")

    # --- 4. Instantiate environment ---
    if config.USE_DETERMINISTIC_ENV:
        env = DeterministicEnv(
            initial_state_data=initial_state,
            model=model,
            processor=processor,
            device=device,
        )
    else:
        env = ChronoOptEnv(
            initial_state_data=initial_state,
            model=model,
            processor=processor,
            device=device,
        )

    # --- 5. Instantiate policy and agent ---
    input_size = config.NUM_DAYS_FOR_STATE * 23
    policy_net = PolicyNetwork(
        input_size=input_size,
        hidden_size=32,
        num_hidden_layers=1,
        dropout_rate=0.0,
    )
    policy_net.to(device)

    agent = ReinforceAgent(
        policy_network=policy_net,
        lr=config.RL_AGENT_PARAMETERS['learning_rate'],
        gamma=config.RL_AGENT_PARAMETERS['gamma'],
        device=device,
    )

    # --- 6. Train ---
    print(f"\nStarting REINFORCE training — {config.RL_TRAIN_NUM_EPISODES} episodes...")
    episode_rewards = agent.train(
        env=env,
        num_episodes=config.RL_TRAIN_NUM_EPISODES,
        max_steps=config.RL_TRAIN_MAX_STEPS,
        log_interval=config.RL_TRAIN_LOG_INTERVAL,
    )

    # --- 7. Save policy ---
    policy_net.save(config.POLICY_SAVE_PATH)

    print(f"\nTraining complete.")
    print(f"Final {config.RL_TRAIN_LOG_INTERVAL}-episode avg reward: "
          f"{np.mean(episode_rewards[-config.RL_TRAIN_LOG_INTERVAL:]):.2f}")


if __name__ == "__main__":
    run_rl_training()