# src/models/edmd_model.py
# Extended Dynamic Mode Decomposition model for biometric prediction.
#
# Lifts the 23-feature state through polynomial observables, then fits
# a linear operator K: ψ(x_t) → y_{t+1}[model_features] via Ridge regression.
#
# Degree-2 observables on 23 inputs: 23 linear + 23 quadratic + 253 cross-terms = 299 total.
# Cross-terms (e.g. steps × bed_time) capture the action interaction effects the LSTM missed.
#
# Works in scaled space (same contract as PredictionModel):
#   - fit(X_scaled, y_scaled)
#   - predict(x_scaled) → y_scaled (12,)
#   - apply_constraints(pred_unscaled, action_unscaled) → constrained (12,)

import os
import pickle
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures


class EDMDModel:
    """
    EDMD-based single-step biometric predictor.

    Input:  last day's full 23-feature vector (scaled)
    Output: next day's 12 model-predicted features (scaled)

    Model feature order (indices 0-11 in unscaled output):
        avg_hr, rhr, resp, stress, body_battery,
        total_sleep, deep_sleep, rem_sleep, awake_sleep,
        restless, avg_sleep_stress, sleep_rhr
    """

    def __init__(self, degree: int = 2, alpha: float = 1.0):
        """
        Args:
            degree: Polynomial degree for observable lifting. 2 is standard EDMD.
            alpha:  Ridge regularization strength. Higher = smoother, less overfit.
        """
        self.degree = degree
        self.alpha = alpha
        self.poly = PolynomialFeatures(degree=degree, include_bias=False)
        self.regressor = Ridge(alpha=alpha)
        self._is_fitted = False
        self.n_observables = None  # set on fit

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit EDMD on scaled data.

        Args:
            X: (N, 23) scaled full-day feature vectors
            y: (N, 12) scaled next-day model feature targets
        """
        assert X.shape[1] == 23, f"Expected 23 input features, got {X.shape[1]}"
        assert y.shape[1] == 12, f"Expected 12 output features, got {y.shape[1]}"

        lifted = self.poly.fit_transform(X)  # (N, n_observables)
        self.n_observables = lifted.shape[1]
        print(f"EDMD: lifting {X.shape[1]} features → {self.n_observables} observables "
              f"(degree={self.degree})")

        self.regressor.fit(lifted, y)
        self._is_fitted = True

        train_pred = self.regressor.predict(lifted)
        train_mse = float(np.mean((train_pred - y) ** 2))
        print(f"EDMD fitted. Train MSE (scaled): {train_mse:.4f}")

    def predict(self, x_scaled: np.ndarray) -> np.ndarray:
        """
        Predict next day's scaled model features from last day's scaled features.

        Args:
            x_scaled: (23,) scaled full-day feature vector

        Returns:
            (12,) scaled model feature predictions
        """
        self._check_fitted()
        lifted = self.poly.transform(x_scaled.reshape(1, -1))
        return self.regressor.predict(lifted)[0]

    def apply_constraints(self,
                          predicted_unscaled: np.ndarray,
                          action_unscaled: np.ndarray,
                          use_constraints: bool = True) -> np.ndarray:
        """
        Enforce physiological consistency on raw EDMD output.

        Constraints:
            1. total_sleep ≤ time_in_bed - 10min (hard physics)
            2. deep + rem + awake ≤ total_sleep (normalized proportionally if violated)
            3. Hard physiological bounds per feature

        Args:
            predicted_unscaled: Raw inverse-transformed output. Shape (12,).
            action_unscaled:    Full 23-feature unscaled day vector.
                                Bed time at [7, 8], wake time at [9, 10].
            use_constraints:    Set False for unconstrained comparison runs.

        Returns:
            np.ndarray: Shape (12,), constrained predictions.
        """
        if not use_constraints:
            return predicted_unscaled.copy()

        p = predicted_unscaled.copy()

        # --- Derive max sleep from action bed/wake times ---
        bed_h,  bed_m  = float(action_unscaled[7]),  float(action_unscaled[8])
        wake_h, wake_m = float(action_unscaled[9]),  float(action_unscaled[10])

        bed_time  = bed_h  + bed_m  / 60.0
        wake_time = wake_h + wake_m / 60.0
        if wake_time <= bed_time:
            wake_time += 24.0  # overnight wrap

        time_in_bed_s = max(0.0, (wake_time - bed_time) * 3600.0)
        max_sleep_s   = max(0.0, time_in_bed_s - 600.0)  # 10min to fall asleep

        # 1. Total sleep bounded by time in bed
        p[5] = np.clip(p[5], 0.0, max_sleep_s)

        # 2. Sleep stages must sum ≤ total sleep
        stages_sum = p[6] + p[7] + p[8]
        if stages_sum > p[5] and stages_sum > 0:
            scale = p[5] / stages_sum
            p[6] *= scale  # deep_sleep_seconds
            p[7] *= scale  # rem_sleep_seconds
            p[8] *= scale  # awake_sleep_seconds

        # 3. Hard physiological bounds
        p[0]  = np.clip(p[0],  30.0,  220.0)   # avg_heart_rate
        p[1]  = np.clip(p[1],  25.0,  120.0)   # resting_heart_rate
        p[2]  = np.clip(p[2],   8.0,   30.0)   # avg_respiration_rate
        p[3]  = np.clip(p[3],   0.0,  100.0)   # avg_stress
        p[4]  = np.clip(p[4],   0.0,  100.0)   # body_battery_end_value
        # p[5] already handled above
        p[6]  = np.clip(p[6],   0.0,   None)   # deep_sleep_seconds
        p[7]  = np.clip(p[7],   0.0,   None)   # rem_sleep_seconds
        p[8]  = np.clip(p[8],   0.0,   None)   # awake_sleep_seconds
        p[9]  = np.clip(p[9],   0.0,  200.0)   # restless_moments_count
        p[10] = np.clip(p[10],  0.0,  100.0)   # avg_sleep_stress
        p[11] = np.clip(p[11], 25.0,  120.0)   # sleep_resting_heart_rate

        return p

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def print_sensitivity(self,
                          agent_feature_keys: list,
                          model_feature_keys: list):
        """
        Inspects the K matrix to show which action features drive each biometric prediction.

        PolynomialFeatures with include_bias=False, degree=2 orders features as:
            [x1, x2, ..., x23,  x1², x1*x2, ..., x23²]
        So the first 23 columns are the linear terms — most interpretable for sensitivity.
        """
        self._check_fitted()
        W = self.regressor.coef_  # (12, n_observables)

        # Linear action weights: first 23 cols, first 11 are agent features
        linear_w = np.abs(W[:, :11])  # (12, 11)

        print("\n" + "=" * 70)
        print("EDMD — Action Sensitivity Report (linear terms of K matrix)")
        print("=" * 70)
        for i, mname in enumerate(model_feature_keys):
            row       = linear_w[i]
            top_idx   = int(np.argmax(row))
            top_w     = row[top_idx]
            total_w   = row.sum()
            pct       = (top_w / total_w * 100) if total_w > 0 else 0.0
            print(f"  {mname:<35} ← {agent_feature_keys[top_idx]:<28} "
                  f"|w|={top_w:.5f}  ({pct:.1f}% of linear action influence)")

        print("\n  Top 3 action features by total influence across all predictions:")
        total_influence = linear_w.sum(axis=0)  # (11,)
        top3 = np.argsort(total_influence)[::-1][:3]
        for rank, idx in enumerate(top3, 1):
            print(f"    {rank}. {agent_feature_keys[idx]:<30} total |w|={total_influence[idx]:.5f}")
        print("=" * 70)

    def evaluate(self, X: np.ndarray, y: np.ndarray,
                 model_feature_keys: list) -> dict:
        """
        Evaluates EDMD on held-out scaled data.

        Args:
            X: (N, 23) scaled inputs
            y: (N, 12) scaled targets
            model_feature_keys: feature names for per-feature MAE reporting

        Returns:
            dict with overall_mse and per-feature MAE
        """
        self._check_fitted()
        pred = np.array([self.predict(x) for x in X])
        mse = float(np.mean((pred - y) ** 2))

        metrics = {'overall_mse': mse}
        print(f"EDMD Evaluation — Overall MSE (scaled): {mse:.4f}")
        print("MAE per feature (scaled):")
        for i, name in enumerate(model_feature_keys):
            mae = float(np.mean(np.abs(pred[:, i] - y[:, i])))
            metrics[f'MAE_{name}'] = mae
            print(f"  {name:<35} {mae:.4f}")

        return metrics

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str):
        """Save fitted model to disk (pickle — sklearn objects)."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            'poly':        self.poly,
            'regressor':   self.regressor,
            'degree':      self.degree,
            'alpha':       self.alpha,
            '_is_fitted':  self._is_fitted,
            'n_observables': self.n_observables,
        }
        with open(path, 'wb') as f:
            pickle.dump(payload, f)
        print(f"EDMDModel saved to {path}")

    @classmethod
    def load(cls, path: str) -> 'EDMDModel':
        """Load a fitted EDMDModel from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        model = cls(degree=data['degree'], alpha=data['alpha'])
        model.poly          = data['poly']
        model.regressor     = data['regressor']
        model._is_fitted    = data['_is_fitted']
        model.n_observables = data['n_observables']
        print(f"EDMDModel loaded from {path}")
        return model

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _check_fitted(self):
        if not self._is_fitted:
            raise RuntimeError("EDMDModel is not fitted yet. Call fit() first.")
