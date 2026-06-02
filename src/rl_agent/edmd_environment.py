# src/rl_agent/edmd_environment.py
# Drop-in replacement for ChronoOptEnv that uses EDMDModel for dynamics.
#
# Inherits everything from ChronoOptEnv and overrides only _predict_next_state().
# The EDMD model operates on scaled last-day features and returns
# inverse-transformed + constrained unscaled (12,) biometric predictions.

import numpy as np
import torch

from src.rl_agent.rl_environment import ChronoOptEnv
from src.models.edmd_model import EDMDModel
from src.models.data_processor import DataProcessor
from src.models.prediction_model import PredictionModel


class EDMDEnv(ChronoOptEnv):
    """
    ChronoOptEnv variant that replaces the LSTM with a fitted EDMDModel.

    The EDMD model lifts the last day's 23-feature vector through polynomial
    observables and predicts next-day model features via a linear operator.
    """

    def __init__(self,
                 initial_state_data: np.ndarray,
                 model: PredictionModel,       # kept for interface compat, unused
                 processor: DataProcessor,
                 device: torch.device,
                 edmd_model: EDMDModel,
                 use_constraints: bool = True):
        """
        Args:
            initial_state_data: (seq_len, 23) unscaled initial state.
            model:              Unused LSTM — kept so train_agent.py needs no changes.
            processor:          Fitted DataProcessor with both scalers.
            device:             Torch device (unused in EDMD, kept for compat).
            edmd_model:         Fitted EDMDModel instance.
            use_constraints:    Toggle physiological constraints on/off.
        """
        super().__init__(
            initial_state_data=initial_state_data,
            model=model,
            processor=processor,
            device=device,
        )
        self.edmd_model      = edmd_model
        self.use_constraints = use_constraints
        print(f"EDMDEnv initialised. Constraints={'ON' if use_constraints else 'OFF'}")

    def _predict_next_state(self, scaled_history: np.ndarray) -> np.ndarray:
        """
        EDMD-based next-state prediction.

        Uses only the last day of the scaled history (EDMD is a one-step model,
        not sequence-based). Returns unscaled (12,) model-predicted features.

        Args:
            scaled_history: (1, seq_len, 23) scaled state history from transform_X.

        Returns:
            np.ndarray: Unscaled constrained model features, shape (12,).
        """
        # Last day of scaled history: (23,)
        last_day_scaled = scaled_history[0, -1, :]

        # EDMD prediction in scaled space
        pred_scaled = self.edmd_model.predict(last_day_scaled)  # (12,)

        # Inverse transform to human-readable values
        pred_unscaled = self.processor.inverse_transform_y(
            pred_scaled.reshape(1, -1)
        )[0]  # (12,)

        # Apply physiological constraints using last day's action
        action_unscaled = np.array(self.history[-1], dtype=np.float32)  # (23,)
        return self.edmd_model.apply_constraints(
            pred_unscaled,
            action_unscaled,
            use_constraints=self.use_constraints,
        )
