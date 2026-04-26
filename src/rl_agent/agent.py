# src/rl_agent/agent.py

import torch
import torch.optim as optim
import numpy as np
from typing import List

from src.rl_agent.policy_network import PolicyNetwork
from src.rl_agent.rl_environment import ChronoOptEnv


class ReinforceAgent:
    """
    REINFORCE (Monte Carlo Policy Gradient) agent for ChronoOpt.

    Maintains a policy network, collects episode rollouts from ChronoOptEnv,
    and updates the policy using discounted returns.
    """

    def __init__(self,
                 policy_network: PolicyNetwork,
                 lr: float = 1e-3,
                 gamma: float = 0.99,
                 device: torch.device = torch.device("cpu")):
        """
        Args:
            policy_network (PolicyNetwork): The two-headed MLP policy.
            lr (float): Learning rate for Adam optimizer.
            gamma (float): Discount factor for returns.
            device (torch.device): Device to run on.
        """
        self.policy = policy_network
        self.gamma = gamma
        self.device = device

        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        # Episode buffers — cleared after each update
        self._episode_log_probs: List[torch.Tensor] = []
        self._episode_rewards: List[float] = []

    # ------------------------------------------------------------------
    # Per-step interface
    # ------------------------------------------------------------------

    def select_action(self, observation: np.ndarray) -> np.ndarray:
        """
        Selects an action given the current observation.
        Stores the log probability for the policy update.

        Args:
            observation (np.ndarray): Scaled observation array, shape (seq_len, 23).

        Returns:
            np.ndarray: Unscaled action vector, shape (11,).
        """
        self.policy.eval()
        action, log_prob = self.policy.get_action(observation, self.device, deterministic=False)
        self.policy.train()
        self._episode_log_probs.append(log_prob)
        return action

    def store_reward(self, reward: float):
        """Stores the reward received after the last action."""
        self._episode_rewards.append(reward)

    # ------------------------------------------------------------------
    # Policy update
    # ------------------------------------------------------------------

    def update_policy(self) -> float:
        """
        Performs a REINFORCE policy gradient update using the stored episode.

        Returns:
            float: The scalar loss value for logging.
        """
        # --- Compute discounted returns ---
        G = 0.0
        returns = []
        for r in reversed(self._episode_rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        returns_tensor = torch.tensor(returns, dtype=torch.float32).to(self.device)

        # --- Normalize returns (variance reduction) ---
        if returns_tensor.std() > 1e-8:
            returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-8)

        # --- Compute loss ---
        loss = sum(-log_prob * G_t 
                   for log_prob, G_t in zip(self._episode_log_probs, returns_tensor))

        # --- Backprop ---
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.optimizer.step()

        scalar_loss = loss.item()

        # --- Clear episode buffers ---
        self._episode_log_probs = []
        self._episode_rewards = []

        return scalar_loss

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self,
              env: ChronoOptEnv,
              num_episodes: int = 500,
              max_steps: int = 30,
              log_interval: int = 10) -> List[float]:
        """
        Runs the full REINFORCE training loop.

        Args:
            env (ChronoOptEnv): The environment to train in.
            num_episodes (int): Total number of episodes to run.
            max_steps (int): Max steps per episode (should match env.max_steps).
            log_interval (int): Print summary every N episodes.

        Returns:
            List[float]: Total undiscounted reward per episode, for plotting.
        """
        episode_rewards = []

        for episode in range(1, num_episodes + 1):
            observation, _ = env.reset()
            total_reward = 0.0

            for _ in range(max_steps):
                action = self.select_action(observation)
                observation, reward, done, _, _ = env.step(action)
                self.store_reward(reward)
                total_reward += reward

                if done:
                    break
            maximum,minimum = max(self._episode_rewards),min(self._episode_rewards)
            loss = self.update_policy()
            episode_rewards.append(total_reward)

            if episode % log_interval == 0:
                avg_reward = np.mean(episode_rewards[-log_interval:])
                print(f"Episode {episode:4d} | "
                      f"Total Reward: {total_reward:7.2f} | "
                      f"Avg({log_interval}): {avg_reward:7.2f} | "
                      f"Loss: {loss:8.4f} | "
                      f"Max_Reward: {maximum} | Min_Reward: {minimum}")

        return episode_rewards


# ------------------------------------------------------------------
# Standalone test
# ------------------------------------------------------------------

if __name__ == "__main__":
    import torch
    import numpy as np
    from src import config
    from src.rl_agent.policy_network import PolicyNetwork
    from src.models.data_processor import DataProcessor
    from src.models.prediction_model import PredictionModel
    from src.rl_agent.rl_environment import ChronoOptEnv

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Minimal dummy setup ---
    seq_len = config.NUM_DAYS_FOR_STATE
    num_features = 23

    # Dummy processor (unfitted scaler — env will handle scaling)
    processor = DataProcessor()

    # Dummy LSTM
    model = PredictionModel(
        input_size=num_features,
        hidden_size=32,
        output_size=12,
        num_layers=2
    )

    # Dummy initial state (unscaled)
    initial_state = np.random.rand(seq_len, num_features).astype(np.float32)
    initial_state[:, 0] *= 10000  # total_steps range
    initial_state[:, 7] = 22.0    # bed_hour
    initial_state[:, 9] = 7.0     # wake_hour

    env = ChronoOptEnv(
        initial_state_data=initial_state,
        model=model,
        processor=processor,
        device=device
    )

    # --- Agent ---
    input_size = seq_len * num_features
    policy_net = PolicyNetwork(input_size=input_size, hidden_size=256, num_hidden_layers=2, dropout_rate=0.1)
    policy_net.to(device)

    agent = ReinforceAgent(policy_network=policy_net, lr=1e-3, gamma=0.99, device=device)

    # --- Short training run ---
    print("\nRunning 5-episode smoke test...")
    rewards = agent.train(env, num_episodes=5, max_steps=30, log_interval=1)

    print(f"\nSmoke test complete. Episode rewards: {[f'{r:.1f}' for r in rewards]}")
    print("agent.py standalone test PASSED.")