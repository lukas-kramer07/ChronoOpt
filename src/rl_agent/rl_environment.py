# src/rl_agent/RL_environment.py
# This module defines the reinforcement learning environment for our
# ChronoOpt agent. It will manage the state, actions, rewards, and
# transitions based on the LSTM model's predictions.

import numpy as np
import torch
from typing import Tuple, Dict, Any

# TODO: Import the BioMetricPredictor and DataProcessor classes once finalized.
from src.models.prediction_model import PredictionModel
from src.models.data_processor import DataProcessor
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
    
    Model-Predicted Features (12): avg_heart_rate, resting_heart_rate, avg_respiration_rate, avg_stress, body_battery_end_value, 
    total_sleep_seconds, deep_sleep_seconds, rem_sleep_seconds, awake_sleep_seconds, restless_moments_count, avg_sleep_stress, 
    sleep_resting_heart_rate

    The agents decisions are rewarded based on the calculated sleep score, the lstm predicts based on the agents decisions. 
    The agent however, is meant to adapt to the individual user and this environment is a pre-training for that purpose. 
    Hence, absolute accuracy is not necessary.
    """
    def __init__(self,
                 initial_state_data: np.ndarray,
                 model: PredictionModel,
                 processor: DataProcessor,
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

        # The state is a list or a fixed-length array of past daily feature vectors.
        self.initial_state_data = initial_state_data.copy()
        self.history = initial_state_data.tolist()
        self.current_step = 0
        self.max_steps = 30  # Episode length — 30 simulated days

        # Total features for one day = 11 (agent-controlled) + 12 (model-predicted) = 23
        self.num_total_features = 23
        self.num_agent_features = 11
        self.num_model_features = 12

        
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

        self.reward_fn = self._calculate_reward
        
        print("ChronoOpt environment initialized.")
        print(f"Initial state history length: {len(self.history)} days.")
        print(f"Action space size: {self.action_space}")
        
    def reset(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Resets the environment to its initial state.

        Returns:
            Tuple[np.ndarray, Dict[str, Any]]:
                observation (np.ndarray): Initial state history. Shape: (sequence_length, 23).
                info (Dict[str, Any]): Additional info.
        """
        self.history = self.initial_state_data.tolist()
        self.current_step = 0
        
        state_array = np.array(self.history, dtype=np.float32)
        if self.processor._is_scaler_fitted:
            observation = self.processor.transform_X(
                state_array.reshape(1, len(self.history), self.num_total_features)
            )[0]
        else:
            observation = state_array
        
        info = {"message": "Environment reset."}
        return observation, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Performs one step in the environment given a vector of agent actions.

        Args:
            action (np.ndarray): 1D array of 11 agent-controlled features in
                                UNSCALED human-readable values.
                                Order matches processor.agent_feature_keys:
                                [total_steps, act_Strength, act_Cardio, act_Yoga,
                                act_Stretching, act_OtherActivity, act_NoActivity,
                                bed_hour, bed_minute, wake_hour, wake_minute]

        Returns:
            Tuple:
                observation (np.ndarray): Updated state history. Shape: (sequence_length, 23).
                reward (float): Sleep score proxy (0-100).
                done (bool): True if episode has ended (max_steps reached).
                truncated (bool): Always False for now.
                info (Dict[str, Any]): Debug info including predicted metrics and reward.
        """
        # --- 1. Build the current state tensor for LSTM input ---
        # history is a list of unscaled 23-feature vectors
        state_array = np.array(self.history, dtype=np.float32)  # (seq_len, 23)

        # Scale the state for LSTM input using scaler_X
        state_scaled = self.processor.transform_X(
            state_array.reshape(1, len(self.history), self.num_total_features)
        )  # (1, seq_len, 23)

        # --- 2. get predicted model features (12) ---
        predicted_model_unscaled  = self._predict_next_state(state_scaled)

        # --- 3. Build the new full 23-feature day vector (unscaled) ---
        # agent features first (0-10), model features second (11-22)
        new_day_unscaled = np.concatenate([
            action.astype(np.float32),           
            predicted_model_unscaled.astype(np.float32)  
        ])  

        # --- 4. Compute reward from predicted sleep metrics ---
        # Reconstruct model features into a dict for sleep score calculation
        predicted_features_dict = self.processor.reconstruct_features_from_flat(
            predicted_model_unscaled  
        )
        reward = float(self._calculate_reward(
            torch.tensor(predicted_model_unscaled)
        ))

        # --- 5. Update state history ---
        # Drop oldest day, append new day (unscaled)
        self.history.pop(0)
        self.history.append(new_day_unscaled.tolist())

        # --- 6. Build next observation (scaled) ---
        next_state_array = np.array(self.history, dtype=np.float32)
        next_observation = self.processor.transform_X(
            next_state_array.reshape(1, len(self.history), self.num_total_features)
        )[0]  # (seq_len, 23)

        # --- 7. Check termination ---
        self.current_step += 1
        done = self.current_step >= self.max_steps
        truncated = False

        info = {
            "step": self.current_step,
            "reward": reward,
            "predicted_sleep_score": reward,
            "predicted_model_features": predicted_features_dict,
        }
        return next_observation, reward, done, truncated, info

    def _calculate_reward(self, predicted_metrics: torch.Tensor) -> float:
        """
        Calculate the reward (0-100) based on the sleep-proxy score. Can be expanded later.

        Args: 
            predicted_metrics (torch.Tensor): A tensor of predicted metrics from the LSTM.
                                              Shape: (1, num_model_features) or (num_model_features,).

        Returns:
            float: The calculated reward value.
        """
        metrics_dict = dict(zip(self.model_features, predicted_metrics.detach().cpu().numpy().flatten()))
        return calculate_sleep_score_proxy(metrics_dict) #the feature names are consistent

    def _predict_next_state(self, scaled_history: np.ndarray) -> np.ndarray:
        """
        Runs the LSTM on the scaled history and returns unscaled model-predicted
        features (12,).

        Args:
            scaled_history (np.ndarray): Scaled state history, shape (seq_len, 23).

        Returns:
            np.ndarray: Unscaled model-predicted features, shape (12,).
        """
        input_tensor = torch.tensor(scaled_history, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            predicted_scaled = self.model(input_tensor).squeeze(0).cpu().numpy()
        return self.processor.inverse_transform_y(predicted_scaled.reshape(1, -1))[0]
