# src/rl_agent/ppo_agent.py
# PPO agent for ChronoOpt.
#
# Architecture:
#   - RolloutBuffer: stores N steps of experience, computes GAE advantages
#   - PPOAgent: collects rollouts, runs PPO update epochs

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import List, Tuple

from src.rl_agent.policy_network import PolicyNetwork
from src.rl_agent.rl_environment import ChronoOptEnv


class RolloutBuffer:
    """
    Stores a fixed-length rollout of experience and computes
    GAE advantages + discounted returns for PPO updates.
    """

    def __init__(self, n_steps: int, obs_shape: tuple, device: torch.device):
        """
        Args:
            n_steps:    Number of steps to collect per rollout.
            obs_shape:  Shape of a single observation (seq_len, 23).
            device:     Torch device.
        """
        self.n_steps = n_steps
        self.obs_shape = obs_shape
        self.device = device
        self.clear()


    def clear(self):
        """Resets all buffers."""
        self.observations = []
        self.continuous_samples = []
        self.activity_indices = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        self._ptr = 0

    def store(self,
              obs: np.ndarray,
              continuous_sample: torch.Tensor,
              activity_idx: torch.Tensor,
              action: np.ndarray,
              log_prob: torch.Tensor,
              reward: float,
              value: torch.Tensor,
              done: bool):
        """Stores one step of experience."""
        self.observations.append(obs.flatten())
        self.continuous_samples.append(continuous_sample.cpu().flatten())
        self.activity_indices.append(activity_idx.cpu())
        self.actions.append(action)
        self.log_probs.append(log_prob.detach().cpu())
        self.rewards.append(reward)
        self.values.append(value.detach().cpu())
        self.dones.append(done)
        self._ptr += 1

    def compute_advantages(self, last_value: float, gamma: float, lam: float):
        """
        Computes GAE advantages and discounted returns in-place.

        Args:
            last_value: V(s_{T+1}) — value estimate for the state after
                        the last collected step. 0.0 if terminal.
            gamma:      Discount factor.
            lam:        GAE lambda.
        """
        rewards = np.array(self.rewards, dtype=np.float32)
        values = np.array([v.item() for v in self.values], dtype=np.float32)
        dones = np.array(self.dones, dtype=np.float32)

        advantages = np.zeros_like(rewards)
        last_gae = 0.0

        for t in reversed(range(len(rewards))):
            next_value = last_value if t == len(rewards) - 1 else values[t + 1]
            next_non_terminal = 1.0 - (dones[t + 1] if t < len(rewards) - 1 else 1.0)
            delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
            last_gae = delta + gamma * lam * next_non_terminal * last_gae
            advantages[t] = last_gae

        self.advantages = advantages
        self.returns = advantages + values  # G_t = A_t + V(s_t)

    def get_tensors(self) -> dict:
        """Converts stored lists to tensors for PPO update."""
        return {
            'observations': torch.tensor(
                np.array(self.observations), dtype=torch.float32).to(self.device),
            'continuous_samples': torch.stack(self.continuous_samples).to(self.device),
            'activity_indices': torch.stack(self.activity_indices).to(self.device),
            'log_probs': torch.stack(self.log_probs).to(self.device),
            'advantages': torch.tensor(self.advantages, dtype=torch.float32).to(self.device),
            'returns': torch.tensor(self.returns, dtype=torch.float32).to(self.device),
        }

    def get_minibatches(self, batch_size: int) -> List[dict]:
        """
        Yields shuffled minibatches from the rollout buffer.

        Args:
            batch_size: Number of samples per minibatch.

        Returns:
            List of minibatch dicts with same keys as get_tensors().
        """
        tensors = self.get_tensors()
        n = len(self.rewards)
        indices = np.random.permutation(n)
        minibatches = []

        for start in range(0, n, batch_size):
            batch_idx = torch.tensor(
                indices[start:start + batch_size], dtype=torch.long)
            minibatches.append({k: v[batch_idx] for k, v in tensors.items()})

        return minibatches

    def __len__(self):
        return self._ptr


class PPOAgent:
    """
    PPO agent for ChronoOpt.

    Collects rollouts from ChronoOptEnv and updates the policy network
    using the clipped PPO objective with GAE advantage estimation.
    """

    def __init__(self,
                 policy_network: PolicyNetwork,
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 lam: float = 0.95,
                 clip_eps: float = 0.2,
                 c1: float = 0.5,
                 c2: float = 0.01,
                 n_steps: int = 2048,
                 k_epochs: int = 4,
                 batch_size: int = 64,
                 max_grad_norm: float = 0.5,
                 device: torch.device = torch.device("cpu")):
        """
        Args:
            policy_network: Two-headed MLP with value head.
            lr:             Adam learning rate.
            gamma:          Discount factor.
            lam:            GAE lambda.
            clip_eps:       PPO clipping epsilon.
            c1:             Value loss coefficient.
            c2:             Entropy bonus coefficient.
            n_steps:        Steps collected per rollout.
            k_epochs:       PPO update epochs per rollout.
            batch_size:     Minibatch size for PPO updates.
            max_grad_norm:  Gradient clipping norm.
            device:         Torch device.
        """
        self.policy = policy_network
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.c1 = c1
        self.c2 = c2
        self.n_steps = n_steps
        self.k_epochs = k_epochs
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        self.device = device

        self.optimizer = optim.Adam(policy_network.parameters(), lr=lr)

        # Observation shape inferred at first rollout
        self._obs_shape = None
        self._buffer = None

        self._reward_mean = 0.0
        self._reward_var = 1.0
        self._reward_count = 0

    def _init_buffer(self, obs_shape: tuple):
        self._obs_shape = obs_shape
        self._buffer = RolloutBuffer(self.n_steps, obs_shape, self.device)

    def _update(self) -> Tuple[float, float, float]:
        """
        Runs K epochs of PPO updates on the collected rollout.

        Returns:
            Tuple of (mean policy loss, mean value loss, mean entropy).
        """
        
        # Normalize advantages
        adv = self._buffer.advantages
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        self._buffer.advantages = adv

        # Normalize returns for stable value learning
        ret = self._buffer.returns
        returns_mean = ret.mean()
        returns_std = ret.std() + 1e-8
        self._buffer.returns = (ret - returns_mean) / returns_std

        policy_losses, value_losses, entropies = [], [], []

        self.policy.train()
        for _ in range(self.k_epochs):
            for batch in self._buffer.get_minibatches(self.batch_size):
                new_log_probs, entropy, new_values = self.policy.evaluate_actions(
                    batch['observations'],
                    batch['continuous_samples'],
                    batch['activity_indices']
                )



                # PPO ratio
                log_ratio = new_log_probs - batch['log_probs']
                ratio = torch.exp(torch.clamp(log_ratio, min=-10.0, max=10.0))

                # Clipped surrogate loss
                advantages = batch['advantages']
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(new_values, batch['returns'])

                # Entropy bonus
                entropy_loss = -entropy.mean()

                loss = policy_loss + self.c1 * value_loss + self.c2 * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(-entropy_loss.item())
        
        self._buffer.clear()
        return (np.mean(policy_losses),
                np.mean(value_losses),
                np.mean(entropies))

    def train(self,
              env: ChronoOptEnv,
              total_steps: int = 500_000,
              log_interval: int = 10) -> List[float]:
        """
        Full PPO training loop.

        Args:
            env:           The environment to train in.
            total_steps:   Total environment steps to collect.
            log_interval:  Print summary every N rollouts.

        Returns:
            List of mean episode rewards per rollout, for plotting.
        """
        obs, _ = env.reset()

        if self._buffer is None:
            self._init_buffer(obs.shape)

        rollout_rewards = []
        episode_reward = 0.0
        episode_rewards_this_rollout = []
        rollout_num = 0
        steps_collected = 0

        print(f"Starting PPO training — {total_steps} total steps, "
              f"{self.n_steps} steps per rollout...")

        while steps_collected < total_steps:
            # --- Collect rollout ---
            self.policy.eval()
            for _ in range(self.n_steps):
                action, log_prob, value, cont_sample, act_idx = \
                    self.policy.get_action(obs, self.device, deterministic=False)

                next_obs, reward, done, truncated, _ = env.step(action)

                # Normalize reward with running statistics
                self._reward_count += 1
                old_mean = self._reward_mean
                self._reward_mean += (reward - self._reward_mean) / self._reward_count
                self._reward_var += (reward - old_mean) * (reward - self._reward_mean)
                reward_std = np.sqrt(self._reward_var / self._reward_count) + 1e-8
                normalized_reward = reward / reward_std  # don't subtract mean — preserve sign of good/bad

                self._buffer.store(
                    obs, cont_sample, act_idx,
                    action, log_prob, normalized_reward, value, done
                )

                episode_reward += reward  # log raw reward for display
                obs = next_obs
                steps_collected += 1

                if done or truncated:
                    episode_rewards_this_rollout.append(episode_reward)
                    episode_reward = 0.0
                    obs, _ = env.reset()

            # Bootstrap value for last state
            with torch.no_grad():
                x = torch.tensor(obs, dtype=torch.float32).flatten().unsqueeze(0).to(self.device)
                _, _, last_value = self.policy.forward(x)
                last_value = last_value.item() if not (done or truncated) else 0.0

            self._buffer.compute_advantages(last_value, self.gamma, self.lam)

            # --- PPO update ---
            policy_loss, value_loss, entropy = self._update()
            rollout_num += 1

            mean_ep_reward = (np.mean(episode_rewards_this_rollout)
                              if episode_rewards_this_rollout else float('nan'))
            rollout_rewards.append(mean_ep_reward)
            episode_rewards_this_rollout = []

            if rollout_num % log_interval == 0:
                print(f"Rollout {rollout_num:4d} | "
                      f"Steps: {steps_collected:7d} | "
                      f"Mean Ep Reward: {mean_ep_reward:7.2f} | "
                      f"Policy Loss: {policy_loss:8.4f} | "
                      f"Value Loss: {value_loss:8.4f} | "
                      f"Entropy: {entropy:6.4f}")

        return rollout_rewards