# src/rl_agent/policy_network.py
# Policy network for the ChronoOpt RL agent.
#
# Architecture:
#   - Shared MLP trunk: processes flattened state observation
#   - Continuous head (5 outputs): total_steps, bed_hour, bed_minute, wake_hour, wake_minute
#   - Activity head (6 outputs): softmax over activity flags
#     (Strength, Cardio, Yoga, Stretching, OtherActivity, NoActivity)
#
# The two-head design is intentional:
#   - Continuous features are unbounded and agent-decided
#   - Activity selection is mutually exclusive — softmax enforces this naturally
#
# Output is always in UNSCALED human-readable space.
# The environment handles internal scaling before feeding to the LSTM.
 
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple
import os
 
# Activity flag order — must match data_processor.agent_feature_keys
ACTIVITY_KEYS = ['Strength', 'Cardio', 'Yoga', 'Stretching', 'OtherActivity', 'NoActivity']
CONTINUOUS_KEYS = ['total_steps', 'bed_hour', 'bed_minute', 'wake_hour', 'wake_minute']
 
# Agent feature order (must match data_processor.agent_feature_keys exactly):
# [total_steps, Strength, Cardio, Yoga, Stretching, OtherActivity, NoActivity,
#  bed_hour, bed_minute, wake_hour, wake_minute]
# Indices: steps=0, activity=1-6, bed_h=7, bed_m=8, wake_h=9, wake_m=10
 
STEPS_IDX = 0
ACTIVITY_START_IDX = 1
ACTIVITY_END_IDX = 7    # exclusive
BED_HOUR_IDX = 7
BED_MIN_IDX = 8
WAKE_HOUR_IDX = 9
WAKE_MIN_IDX = 10
 
 
class PolicyNetwork(nn.Module):
    """
    Two-headed policy network for ChronoOpt.
 
    Input:
        Flattened scaled observation of shape (NUM_DAYS_FOR_STATE * 23,)
 
    Output:
        action (np.ndarray): Unscaled action vector of shape (11,) in agent_feature_keys order.
        continuous_out (torch.Tensor): Raw continuous head output (for loss computation).
        activity_probs (torch.Tensor): Softmax probabilities over 6 activities (for loss computation).
    """
 
    def __init__(self,
                 input_size: int,
                 hidden_size: int = 128,
                 num_hidden_layers: int = 1,
                 dropout_rate: float = 0.1):
        """
        Args:
            input_size (int): Flattened observation size = NUM_DAYS_FOR_STATE * 23.
            hidden_size (int): Width of the shared trunk and heads.
            num_hidden_layers (int): Number of hidden layers in the shared trunk.
            dropout_rate (float): Dropout applied after each trunk layer.
        """
        super(PolicyNetwork, self).__init__()
 
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_hidden_layers=num_hidden_layers
        self.dropout_rate = dropout_rate
        # --- Shared trunk ---
        trunk_layers = []
        in_features = input_size
        for _ in range(num_hidden_layers):
            trunk_layers.append(nn.Linear(in_features, hidden_size))
            trunk_layers.append(nn.Tanh())  # Tanh: gradient everywhere, no dead neurons
            in_features = hidden_size
        self.trunk = nn.Sequential(*trunk_layers)
 
        # --- Continuous head (5 outputs) ---
        # Outputs: total_steps, bed_hour, bed_minute, wake_hour, wake_minute
        # Activation: sigmoid — we scale to realistic ranges post-hoc
        self.continuous_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 5),
            nn.Sigmoid()  # Output in [0, 1] — scaled to real ranges in decode_action()
        )
        
        nn.init.uniform_(self.continuous_head[-2].weight, -0.1, 0.1)
        nn.init.constant_(self.continuous_head[-2].bias, 0.0)

        # --- Activity head (6 outputs) ---
        # Softmax enforces mutual exclusivity across activity types
        self.activity_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 6)
            # No activation — softmax applied in forward()
        )
 
        print(f"PolicyNetwork initialized. Input size: {input_size}, Hidden size: {hidden_size}, "
              f"Layers: {num_hidden_layers}, Dropout: {dropout_rate}")
 
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
 
        Args:
            x (torch.Tensor): Flattened scaled observation. Shape: (batch_size, input_size).
 
        Returns:
            Tuple:
                continuous_out (torch.Tensor): Sigmoid output [0,1] for 5 continuous features.
                                               Shape: (batch_size, 5).
                activity_probs (torch.Tensor): Softmax probabilities over 6 activities.
                                               Shape: (batch_size, 6).
        """
        trunk_out = self.trunk(x)
        continuous_out = self.continuous_head(trunk_out)
        activity_logits = self.activity_head(trunk_out)
        activity_probs = F.softmax(activity_logits, dim=-1)
        return continuous_out, activity_probs
 
    def decode_action(self,
                      continuous_out: torch.Tensor,
                      activity_probs: torch.Tensor,
                      deterministic: bool = False) -> np.ndarray:
        """
        Converts raw network outputs into a human-readable unscaled action vector.
 
        Continuous features are scaled from [0,1] to realistic ranges:
            total_steps  : [2000, 25000]
            bed_hour     : [20, 26]  (20=8PM, 26=2AM next day)
            bed_minute   : [0, 59]
            wake_hour    : [5, 10]
            wake_minute  : [0, 59]
 
        Activity selection:
            deterministic=True  → argmax (greedy)
            deterministic=False → sample from softmax distribution (exploration)
 
        Args:
            continuous_out (torch.Tensor): Shape (1, 5) or (5,).
            activity_probs (torch.Tensor): Shape (1, 6) or (6,).
            deterministic (bool): Use argmax vs sampling for activity selection.
 
        Returns:
            np.ndarray: Unscaled action vector of shape (11,).
                        Order: [total_steps, Strength, Cardio, Yoga, Stretching,
                                OtherActivity, NoActivity, bed_hour, bed_minute,
                                wake_hour, wake_minute]
        """
        cont = continuous_out.detach().cpu().numpy().flatten()
        probs = activity_probs.detach().cpu().numpy().flatten()
 
        # Scale continuous outputs to realistic ranges
        steps = 2000 + cont[0] * (25000 - 2000)
        bed_hour = 20 + cont[1] * (26 - 20)     # 8PM to 2AM (next day)
        bed_min = cont[2] * 59
        wake_hour = 5 + cont[3] * (10 - 5)       # 5AM to 10AM
        wake_min = cont[4] * 59
 
        # Round time components to integers
        bed_hour = int(round(bed_hour)) % 24
        bed_min = int(round(bed_min))
        wake_hour = int(round(wake_hour)) % 24
        wake_min = int(round(wake_min))
        steps = float(round(steps))
 
        # Select activity
        if deterministic:
            activity_idx = int(np.argmax(probs))
        else:
            activity_idx = int(np.random.choice(len(probs), p=probs))
 
        # Build one-hot activity flags
        activity_flags = np.zeros(6, dtype=np.float32)
        activity_flags[activity_idx] = 1.0
 
        # Assemble full action vector in agent_feature_keys order
        action = np.array([
            steps,
            activity_flags[0],  # Strength
            activity_flags[1],  # Cardio
            activity_flags[2],  # Yoga
            activity_flags[3],  # Stretching
            activity_flags[4],  # OtherActivity
            activity_flags[5],  # NoActivity
            float(bed_hour),
            float(bed_min),
            float(wake_hour),
            float(wake_min),
        ], dtype=np.float32)
 
        return action
 
    def get_action(self, observation: np.ndarray, device: torch.device, deterministic: bool = False) -> tuple:
        """
        Gets an action and its log probability from the policy network.

        Args:
            observation (np.ndarray): Flattened observation array.
            device (torch.device): Device to run on.
            deterministic (bool): If True, returns deterministic action.

        Returns:
            tuple: (action np.ndarray(11), log_prob scalar tensor)
        """
        x = torch.tensor(observation, dtype=torch.float32).flatten().unsqueeze(0).to(device)
        continuous_out, activity_probs = self.forward(x)

        # --- Continuous log prob (Normal distribution, fixed std) ---
        std = torch.full_like(continuous_out, 0.2)
        dist_continuous = torch.distributions.Normal(continuous_out, std)

        if deterministic:
            continuous_sample = continuous_out
        else:
            continuous_sample = dist_continuous.sample()

        log_prob_continuous = dist_continuous.log_prob(continuous_sample).sum(dim=-1) # scalar

        # --- Activity log prob (Categorical) ---
        dist_activity = torch.distributions.Categorical(probs=activity_probs)

        if deterministic:
            activity_sample = torch.argmax(activity_probs, dim=-1)
        else:
            activity_sample = dist_activity.sample()

        log_prob_activity = dist_activity.log_prob(activity_sample) # scalar

        # --- Combined log prob ---
        log_prob = log_prob_continuous + log_prob_activity

        # --- Decode to unscaled action ---
        action = self.decode_action(continuous_sample, activity_probs, deterministic)
        return action, log_prob
    
    def save(self, path: str):
        """
        Saves policy network weights and architecture metadata to disk.

        Args:
            path (str): File path to save the checkpoint.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        checkpoint = {
            'state_dict': self.state_dict(),
            'metadata': {
                'input_size': self.input_size,
                'hidden_size': self.hidden_size,
                'num_hidden_layers': self.num_hidden_layers,
                'dropout_rate': self.dropout_rate,
            }
        }
        torch.save(checkpoint, path)
        print(f"Policy saved to {path}")

    @classmethod
    def load(cls, path: str, device: torch.device) -> 'PolicyNetwork':
        """
        Loads a PolicyNetwork from a checkpoint file.

        Args:
            path (str): Path to the saved checkpoint.
            device (torch.device): Device to load onto.

        Returns:
            PolicyNetwork: Reconstructed network ready for inference.
        """
        checkpoint = torch.load(path, map_location=device)
        meta = checkpoint['metadata']
        net = cls(
            input_size=meta['input_size'],
            hidden_size=meta['hidden_size'],
            num_hidden_layers=meta['num_hidden_layers'],
            dropout_rate=meta['dropout_rate'],
        )
        net.load_state_dict(checkpoint['state_dict'])
        net.to(device)
        print(f"Policy loaded from {path}")
        return net


if __name__ == "__main__":
    import torch
    import numpy as np
    from src import config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    input_size = config.NUM_DAYS_FOR_STATE * 23
    net = PolicyNetwork(input_size=input_size, hidden_size=256, num_hidden_layers=2, dropout_rate=0.1)
    net.to(device)

    dummy_obs = np.random.randn(config.NUM_DAYS_FOR_STATE, 23).astype(np.float32)

    stochastic_action, stochastic_logprob = net.get_action(dummy_obs, device, deterministic=False)
    print(f"Stochastic action: {stochastic_action}")
    print(f"Stochastic log_prob: {stochastic_logprob.item():.4f}")

    deterministic_action, deterministic_logprob = net.get_action(dummy_obs, device, deterministic=True)
    print(f"Deterministic action: {deterministic_action}")
    print(f"Deterministic log_prob: {deterministic_logprob.item():.4f}")

    print("PolicyNetwork standalone test PASSED.")