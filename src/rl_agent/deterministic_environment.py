# src/rl_agent/deterministic_environment.py
# A deterministic environment for validating the REINFORCE agent.
# Inherits from ChronoOptEnv and overrides _predict_next_state() with a
# fixed analytical function — no LSTM involved.
# The analytical mapping is physiologically motivated: actions that reflect
# good sleep hygiene (consistent schedule, adequate steps, some activity)
# produce better predicted biometrics and therefore higher rewards.

import numpy as np
import torch
from src.rl_agent.rl_environment import ChronoOptEnv


class DeterministicEnv(ChronoOptEnv):
    """
    A deterministic variant of ChronoOptEnv that replaces the LSTM prediction
    with a fixed analytical function mapping agent actions to model-predicted
    biometrics.

    Agent feature order (11):
        [total_steps, Strength, Cardio, Yoga, Stretching, OtherActivity,
         NoActivity, bed_hour, bed_minute, wake_hour, wake_minute]

    Model feature order (12):
        [avg_heart_rate, resting_heart_rate, avg_respiration_rate, avg_stress,
         body_battery_end_value, total_sleep_seconds, deep_sleep_seconds,
         rem_sleep_seconds, awake_sleep_seconds, restless_moments_count,
         avg_sleep_stress, sleep_resting_heart_rate]
    """

    def _predict_next_state(self, scaled_history: np.ndarray) -> np.ndarray:
        """
        Analytical replacement for the LSTM. Uses only the most recent day's
        agent-controlled features (last row of unscaled history) to compute
        physiologically motivated model-predicted features.

        Args:
            scaled_history (np.ndarray): Scaled state history, shape (seq_len, 23).
                                         Used only to recover the last day's action.

        Returns:
            np.ndarray: Unscaled model-predicted features, shape (12,).
        """
        # Recover unscaled last day from history (agent features are at indices 0-10)
        last_day_unscaled = self.history[-1]

        total_steps     = last_day_unscaled[0]
        is_strength     = last_day_unscaled[1]
        is_cardio       = last_day_unscaled[2]
        is_yoga         = last_day_unscaled[3]
        is_stretching   = last_day_unscaled[4]
        is_other        = last_day_unscaled[5]
        is_no_activity  = last_day_unscaled[6]
        bed_hour        = last_day_unscaled[7]
        bed_minute      = last_day_unscaled[8]
        wake_hour       = last_day_unscaled[9]
        wake_minute     = last_day_unscaled[10]

        # Bedtime: peak at 22.5h, but also accept wrapped late hours (0-3h = 24-27h)
        # Unwrap: if bed_hour < 6, treat as bed_hour + 24
        bed_hour_unwrapped = bed_hour + 24.0 if bed_hour < 6 else bed_hour
        bed_time = bed_hour_unwrapped + bed_minute / 60.0
        bed_score = float(np.exp(-0.5 * ((bed_time - 22.5) / 2.0) ** 2))  # std=2h, more forgiving

        # Wake time: peak at 7h, forgiving std=2h
        wake_time = wake_hour + wake_minute / 60.0
        wake_score = float(np.exp(-0.5 * ((wake_time - 7.0) / 2.0) ** 2))

        # Steps: more forgiving range, peak at 8500, std=6000
        steps_score = float(np.clip(
            1.0 - ((total_steps - 8500) / 6000.0) ** 2,
            0.0, 1.0
        ))

        # Activity: vigorous activity rewarded, no activity penalized
        vigorous = float(is_strength or is_cardio)
        any_activity = 1.0 - float(is_no_activity)
        activity_score = 0.3 * any_activity + 0.7 * vigorous  # max 1.0, no activity = 0.0

        # Sleep window
        if wake_time < bed_time:
            sleep_hours = (24.0 - bed_time) + wake_time
        else:
            sleep_hours = wake_time + (24.0 - bed_time)
        sleep_hours = float(np.clip(sleep_hours, 0.0, 12.0))

        # Sharp sleep duration scoring: quadratic penalty outside 7.5-8.5h
        sleep_score = float(np.clip(
            1.0 - ((sleep_hours - 8.0) / 3.0) ** 2,
            0.0, 1.0
        ))


        total_sleep_seconds = sleep_hours * 3600.0

        # --- Map scores to biometric features ---
        # All features now strongly sensitive to the scores above

        # Heart rate: drops significantly with good steps + activity
        avg_heart_rate = 80.0 - 15.0 * steps_score - 8.0 * vigorous
        resting_heart_rate = 65.0 - 12.0 * steps_score - 5.0 * bed_score

        avg_respiration_rate = 14.5 - 0.5 * any_activity

        # Stress: sharp reduction with good bed + activity combination
        avg_stress = 55.0 - 20.0 * bed_score - 15.0 * activity_score - 10.0 * steps_score

        # Body battery: requires all three to be good simultaneously
        body_battery_end_value = 20.0 + 35.0 * bed_score + 25.0 * sleep_score + 20.0 * steps_score

        # Sleep architecture — tightly coupled to bed/wake scores
        asleep_seconds = total_sleep_seconds * 0.92

        deep_ratio = 0.10 + 0.15 * vigorous * bed_score  # needs BOTH vigorous AND good bedtime
        deep_sleep_seconds = asleep_seconds * float(np.clip(deep_ratio, 0.0, 0.30))

        rem_ratio = 0.12 + 0.14 * wake_score * bed_score  # needs consistent schedule
        rem_sleep_seconds = asleep_seconds * float(np.clip(rem_ratio, 0.0, 0.28))

        # Awake time: penalized sharply by late bedtime
        awake_sleep_seconds = total_sleep_seconds * (0.15 - 0.10 * bed_score)
        awake_sleep_seconds = float(np.clip(awake_sleep_seconds, 0.0, None))

        # Restlessness: needs activity AND good schedule
        restless_moments_count = 75.0 - 30.0 * activity_score - 25.0 * bed_score - 15.0 * steps_score
        restless_moments_count = float(np.clip(restless_moments_count, 3.0, 80.0))

        avg_sleep_stress = 45.0 - 20.0 * bed_score - 12.0 * activity_score - 8.0 * steps_score
        avg_sleep_stress = float(np.clip(avg_sleep_stress, 3.0, 55.0))

        sleep_resting_heart_rate = resting_heart_rate - 3.0

        return np.array([
            avg_heart_rate,
            resting_heart_rate,
            avg_respiration_rate,
            avg_stress,
            body_battery_end_value,
            total_sleep_seconds,
            deep_sleep_seconds,
            rem_sleep_seconds,
            awake_sleep_seconds,
            restless_moments_count,
            avg_sleep_stress,
            sleep_resting_heart_rate,
        ], dtype=np.float32)