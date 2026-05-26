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
from src.rl_agent.ppo_agent import PPOAgent
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
        hidden_size=64,
        num_hidden_layers=1,
        dropout_rate=0.0,
    )
    policy_net.to(device)

    ppo_params = config.PPO_HYPERPARAMETERS
    agent = PPOAgent(
        policy_network=policy_net,
        lr=ppo_params['lr'],
        gamma=ppo_params['gamma'],
        lam=ppo_params['lam'],
        clip_eps=ppo_params['clip_eps'],
        c1=ppo_params['c1'],
        c2=ppo_params['c2'],
        n_steps=ppo_params['n_steps'],
        k_epochs=ppo_params['k_epochs'],
        batch_size=ppo_params['batch_size'],
        max_grad_norm=ppo_params['max_grad_norm'],
        device=device,
    )


    evaluate_reward_range(env,processor,device)
    # --- 6. Train ---
    print(f"\nStarting REINFORCE training — {config.RL_TRAIN_NUM_EPISODES} episodes...")
    rollout_rewards = agent.train(
        env=env,
        total_steps=ppo_params['total_steps'],
        log_interval=ppo_params['log_interval'],
    )

    # --- 7. Save policy ---
    policy_net.save(config.POLICY_SAVE_PATH)

    print(f"\nTraining complete.")
    print(f"Final {config.RL_TRAIN_LOG_INTERVAL}-episode avg reward: "
          f"{np.mean(rollout_rewards[-config.RL_TRAIN_LOG_INTERVAL:]):.2f}")
    # --- Evaluate trained policy ---
    evaluate_policy(policy_net, env, processor, device, num_days=30)

# Run 10 episodes with a random policy and 10 with a greedy optimal policy
# to verify the reward spread is meaningful

def evaluate_reward_range(env, processor, device):
    """Check reward range between worst and best possible actions."""
    
    # Worst action: no steps, no activity, bad sleep schedule
    worst_action = np.array([0, 0, 0, 0, 0, 0, 1,  # NoActivity
                              6, 0, 12, 0], dtype=np.float32)  # bed 6am, wake 12am
    
    # Best action: good steps, vigorous activity, good sleep schedule  
    best_action = np.array([9000, 1, 0, 0, 0, 0, 0,  # Strength
                             22, 30, 7, 0], dtype=np.float32)  # bed 22:30, wake 7:00

    obs, _ = env.reset()
    
    worst_rewards = []
    best_rewards = []
    
    for _ in range(30):
        _, r_worst, _, _, _ = env.step(worst_action)
        worst_rewards.append(r_worst)
    
    obs, _ = env.reset()
    for _ in range(30):
        _, r_best, _, _, _ = env.step(best_action)
        best_rewards.append(r_best)
    
    print(f"Worst action avg reward per step: {np.mean(worst_rewards):.2f}")
    print(f"Best action avg reward per step:  {np.mean(best_rewards):.2f}")
    print(f"Reward gap: {np.mean(best_rewards) - np.mean(worst_rewards):.2f}")

def evaluate_policy(policy_net: PolicyNetwork,
                    env: ChronoOptEnv,
                    processor: DataProcessor,
                    device: torch.device,
                    num_days: int = 30):
    """
    Runs the trained policy deterministically for num_days and prints
    a human-readable schedule of recommended daily behaviors.
    """
    from src.rl_agent.rl_environment import ChronoOptEnv
    from datetime import date, timedelta

    ACTIVITY_NAMES = ['Strength', 'Cardio', 'Yoga', 'Stretching', 'OtherActivity', 'NoActivity']

    print("\n" + "="*65)
    print("CHRONOOPT — RECOMMENDED SCHEDULE (30-DAY SIMULATION)")
    print("="*65)
    print(f"{'Day':<5} {'Date':<12} {'Steps':>7} {'Bed':>6} {'Wake':>6} {'Activity':<14} {'Score':>6}")
    print("-"*65)

    obs, _ = env.reset()
    total_score = 0.0
    start_date = date.today()

    policy_net.eval()
    for day in range(num_days):
        action, _, _, _, _ = policy_net.get_action(obs, device, deterministic=True)
        obs, reward, done, _, _ = env.step(action)
        total_score += reward

        steps      = int(action[0])
        activity   = ACTIVITY_NAMES[int(np.argmax(action[1:7]))]
        bed_hour   = int(action[7])
        bed_min    = int(action[8])
        wake_hour  = int(action[9])
        wake_min   = int(action[10])
        day_date   = (start_date + timedelta(days=day)).strftime("%Y-%m-%d")

        # Wrap bed hour for display
        bed_display  = f"{bed_hour % 24:02d}:{bed_min:02d}"
        wake_display = f"{wake_hour:02d}:{wake_min:02d}"

        print(f"{day+1:<5} {day_date:<12} {steps:>7,} {bed_display:>6} {wake_display:>6} "
              f"{activity:<14} {reward:>6.1f}")

        if done:
            break

    print("-"*65)
    print(f"{'':5} {'30-day avg sleep score:':>40} {total_score/num_days:>6.1f}")
    print("="*65)

if __name__ == "__main__":
    run_rl_training()