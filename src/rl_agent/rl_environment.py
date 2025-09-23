# src/rl_agent/RL_environment.py
# This module defines the reinforcement learning environment for our
# ChronoOpt agent. It will manage the state, actions, rewards, and
# transitions based on the LSTM model's predictions.

import numpy as np
import torch
from typing import Tuple, Dict, Any

# TODO: Import the BioMetricPredictor and DataProcessor classes once finalized.
# from src.models.saved_models import BioMetricPredictor
# from src.models.data_processor import DataProcessor
# from src.features.feature_engineer import extract_daily_features, create_state_vectors
from src.features.utils import calculate_sleep_score_proxy

class ChronoOptEnv:
    """
    A custom reinforcement learning environment for optimizing daily routines
    to improve a user's biometric health.

    The environment uses a pre-trained LSTM model to predict the next day's
    biometrics based on the current state and the agent's actions.

    Currently, the state/decision space looks as follows:
    
    Agent-Controlled Features (11): total_steps, activity_Strength, activity_Cardio, activity_Yoga, activity_Stretching, 
    activity_OtherActivity, activity_NoActivity, bed_time_gmt_hour, bed_time_gmt_minute, wake_time_gmt_hour, wake_time_gmt_minute
    
    Model-Predicted Features (12): avg_heart_rate, resting_heart_rate, avg_respiration_rate, avg_stress, body_battery_end_value, total_sleep_seconds,
    deep_sleep_seconds, rem_sleep_seconds, awake_sleep_seconds, restless_moments_count, avg_sleep_stress, sleep_resting_heart_rate

    The agents decisions are rewarded based on the calculated sleep score, the lstm predicts based on the agents decisions. The agent however, is meant to adapt to the individual user
    and this environment is a pre-training for that purpose. Hence, absolute accuracy is not necessary.
    """
    def __init__(self,
                 initial_state_data: np.ndarray,
                 model: 'BioMetricPredictor',
                 processor: 'DataProcessor',
                 device: torch.device = torch.device("cpu")):
        """
        Initializes the environment.

        Args:
            initial_state_data (np.ndarray): A NumPy array representing the initial
                                             historical state (e.g., 7 days of data).
                                             Shape: (sequence_length, num_features).
            model (BioMetricPredictor): The pre-trained LSTM prediction model.
            processor (DataProcessor): The data processor for scaling and feature handling.
            device (torch.device): The device (CPU or CUDA) to run the model on.
        """
        if initial_state_data.ndim != 2:
            raise ValueError("initial_state_data must be a 2D NumPy array.")
        
        self.device = device
        self.model = model.to(self.device)
        self.model.eval() # Set model to evaluation mode
        self.processor = processor

        # The state is a queue or a fixed-length array of past daily feature vectors.
        # We use a deque or a similar structure to efficiently manage the state history.
        self.history = initial_state_data.tolist() # Convert to list for easy appending/popping

        # Total features for one day = 11 (agent-controlled) + 12 (model-predicted) = 23
        self.num_total_features = 23
        self.num_agent_features = 11
        self.num_model_features = 12

        # TODO: define observation space
        self.observation_space = None
        
        # Define the action space. This is a vector of 11 values corresponding to the
        # agent-controlled features.
        self.action_space = self.num_agent_features

        # Agent-Controlled Features:
        self.agent_features = [
            'total_steps', 'activity_Strength', 'activity_Cardio', 'activity_Yoga',
            'activity_Stretching', 'activity_OtherActivity', 'activity_NoActivity',
            'bed_time_gmt_hour', 'bed_time_gmt_minute', 'wake_time_gmt_hour',
            'wake_time_gmt_minute'
        ]

        # Model-Predicted Features:
        self.model_features = [
            'avg_heart_rate', 'resting_heart_rate', 'avg_respiration_rate',
            'avg_stress', 'body_battery_end_value', 'total_sleep_seconds',
            'deep_sleep_seconds', 'rem_sleep_seconds', 'awake_sleep_seconds',
            'restless_moments_count', 'avg_sleep_stress', 'sleep_resting_heart_rate'
        ]
        
        print("ChronoOpt environment initialized.")
        print(f"Initial state history length: {len(self.history)} days.")
        print(f"Action space size: {self.action_space}")
        
    def reset(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Resets the environment to its initial state.

        Returns:
            Tuple[np.ndarray, Dict[str, Any]]:
                observation (np.ndarray): The initial observation for the agent.
                info (Dict[str, Any]): Additional info about the state.
        """
        # TODO: Implement reset logic, e.g., re-initialize history
        self.history = self.initial_state_data.tolist()
        # Get the current observation (the last N days from the history)
        observation = np.array(self.history, dtype=np.float32)
        info = {"message": "Environment reset."}
        return observation, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Performs one step in the environment given a vector of actions.

        Args:
            action (np.ndarray): A 1D NumPy array representing the agent's
                                 chosen values for the 11 agent-controlled features.
                                 Shape: (11,).
        Returns:
            Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
                observation (np.ndarray): The new state after the action.
                reward (float): The reward received from the action.
                terminated (bool): Whether the episode has ended (prematurely, for instance due to too high stress or low sleep scores).
                truncated (bool): Whether the episode was truncated due to a time limit.
                info (Dict[str, Any]): Additional info about the step.
       """
        if action.shape[0] != self.num_agent_features:
            raise ValueError(f"Action must be a 1D array of size {self.num_agent_features}.")

        # 1. Get the current state (last N days of physiological data)
        current_state = np.array(self.history, dtype=np.float32)
        
        # 2. combine the state with the chosen actions
        model_input_np = self.processor.combine_state_action(current_state, action)
        model_input = torch.from_numpy(model_input_np).unsqueeze(0).to(self.device)
        
        # 3. Predict the next day's physiological metrics
        with torch.no_grad():
            predicted_output = self.model(model_input)
        
        # 4. Calculate the reward based on the predicted metrics
        reward = self.calculate_reward(predicted_output)
        
        # 5. Update the environment's state history
        new_day_features = np.concatenate((action, predicted_output.detach().cpu().numpy().flatten()))
        
        self.history.popleft()
        self.history.append(new_day_features.tolist())

        # 6. Get the new observation and determine if the episode is terminated
        observation = np.array(self.history, dtype=np.float32)
        terminated = False # The episode ends after a fixed number of steps
        truncated = False
        info = {"predicted_metrics": new_day_features.tolist()}
        
        return observation, reward, terminated, truncated, info

    def calculate_reward(self, predicted_metrics: torch.Tensor) -> float:
        """
        Calculate the reward (0-100) based on the sleep-proxy score. Can be expanded later.

        Args: 
            predicted_metrics (torch.Tensor): A tensor of predicted metrics from the LSTM.
                                              Shape: (1, num_model_features) or (num_model_features,).

        Returns:
            float: The calculated reward value.
        """
        metrics_dict = {k:v for k,v in zip(self.model_features, predicted_metrics.detach().cpu().numpy().flatten())}
        return calculate_sleep_score_proxy(metrics_dict) #the feature names are consistent