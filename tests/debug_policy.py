# tests/debug_policy.py
# Run from project root: python -m tests.debug_policy
#
# Diagnoses why the PPO policy may be recommending suboptimal actions.
# Three checks:
#   A. Observation sanity  — does the real scaled state look like training data?
#   B. Scorer sanity       — do known-good actions outscore known-bad ones?
#   C. Policy behaviour    — what does the policy actually output and why?

import numpy as np
import torch
from datetime import date

from src import config
from src.rl_agent.train_agent import build_fitted_processor, build_initial_state, load_trained_lstm
from src.rl_agent.policy_network import PolicyNetwork
from src.data_ingestion.garmin_parser import get_historical_metrics
from src.features.feature_engineer import extract_daily_features
from src.api.inference import (
    ModelBundle, _score_policy_rollout, _score_fixed_action, _build_real_state, _scale_state,
)

SEP  = "=" * 60
SEP2 = "-" * 60

KNOWN_GOOD = np.array([9000, 1, 0, 0, 0, 0, 0, 22, 30,  7,  0], dtype=np.float32)  # Strength, bed 22:30, wake 07:00
KNOWN_MID  = np.array([5000, 0, 0, 0, 1, 0, 0, 24, 0,   7, 30], dtype=np.float32)  # Stretching, bed 23:00
KNOWN_BAD  = np.array([1000, 0, 0, 0, 0, 0, 1,  2, 0,  5,  0], dtype=np.float32)  # NoActivity, bed 02:00, wake 09:00

ACTIVITY_NAMES = ['Strength', 'Cardio', 'Yoga', 'Stretching', 'OtherActivity', 'NoActivity']


def load_models() -> ModelBundle:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    lstm = load_trained_lstm(device)
    processor, processed_features = build_fitted_processor()

    policy = PolicyNetwork.load(config.POLICY_SAVE_PATH, device)

    try:
        from src.models.edmd_model import EDMDModel
        edmd = EDMDModel.load(config.EDMD_MODEL_SAVE_PATH)
    except (FileNotFoundError, ImportError):
        edmd = None

    return ModelBundle(
        lstm=lstm, processor=processor, policy=policy,
        policy_source="trained_policy", device=device, edmd=edmd,
    ), processed_features


def check_a_observation(state_array, scaled_obs):
    """Does the real scaled observation look like training data?"""
    print(SEP)
    print("A. OBSERVATION SANITY")
    print(SEP)

    print(f"\nUnscaled state shape: {state_array.shape}")
    print(f"Scaled obs shape:     {scaled_obs.shape}\n")

    print("Unscaled — per-feature stats across 10 days:")
    feature_names = [
        'total_steps', 'act_Strength', 'act_Cardio', 'act_Yoga',
        'act_Stretching', 'act_Other', 'act_NoActivity',
        'bed_hour', 'bed_min', 'wake_hour', 'wake_min',
        'avg_hr', 'resting_hr', 'respiration', 'avg_stress',
        'body_battery', 'total_sleep_s', 'deep_sleep_s',
        'rem_sleep_s', 'awake_sleep_s', 'restless', 'sleep_stress', 'sleep_rhr'
    ]
    print(f"  {'Feature':<18} {'Min':>8} {'Max':>8} {'Mean':>8}")
    print(f"  {'-'*46}")
    for i, name in enumerate(feature_names):
        col = state_array[:, i]
        print(f"  {name:<18} {col.min():>8.2f} {col.max():>8.2f} {col.mean():>8.2f}")

    print(f"\nScaled obs — overall stats:")
    print(f"  min={scaled_obs.min():.3f}  max={scaled_obs.max():.3f}  "
          f"mean={scaled_obs.mean():.3f}  std={scaled_obs.std():.3f}")
    print("  [Expected range roughly -3 to +3 for a fitted StandardScaler]")

    # Flag if saturated
    saturated = np.abs(scaled_obs) > 4
    if saturated.any():
        idxs = np.argwhere(saturated)
        print(f"\n  WARNING: {len(idxs)} values outside [-4, 4] — possible scaling issue")
        for day, feat in idxs[:5]:
            print(f"    day={day}, feature={feature_names[feat]}, value={scaled_obs[day, feat]:.2f}")
    else:
        print("  No saturation detected.")


def check_b_scorer(state_array, scaled_obs, models):
    """Do known-good actions outscore known-bad ones?"""
    print(f"\n{SEP}")
    print("B. SCORER SANITY")
    print(SEP)

    actions = [
        ("known_good  (Strength, bed 22:30, wake 07:00)", KNOWN_GOOD),
        ("known_mid   (Stretching, bed 23:00, wake 07:30)", KNOWN_MID),
        ("known_bad   (NoActivity, bed 02:00, wake 12:00)", KNOWN_BAD),
    ]

    print(f"\n  {'Action':<46} {'Score':>6}")
    print(f"  {'-'*54}")
    for label, action in actions:
        score = _score_fixed_action(state_array, action, models, n_steps=5)
        print(f"  {label:<46} {score:>6.1f}")

    print("\n  [Expected: good > mid > bad. If not, scorer or LSTM/EDMD is the problem]")


def check_c_policy(state_array, scaled_obs, models):
    """What does the policy actually output?"""
    print(f"\n{SEP}")
    print("C. POLICY BEHAVIOUR")
    print(SEP)

    device = models.device
    policy = models.policy
    policy.eval()

    x = torch.tensor(scaled_obs, dtype=torch.float32).flatten().unsqueeze(0).to(device)

    with torch.no_grad():
        continuous_out, activity_probs, value = policy.forward(x)

    cont = continuous_out.cpu().numpy().flatten()
    probs = activity_probs.cpu().numpy().flatten()
    val = value.item()

    print(f"\nValue estimate (V(s)): {val:.3f}")
    print(f"\nRaw continuous head output (sigmoid, [0,1]):")
    cont_names = ['steps_norm', 'bed_hour_norm', 'bed_min_norm', 'wake_hour_norm', 'wake_min_norm']
    for name, v in zip(cont_names, cont):
        print(f"  {name:<18} {v:.4f}")

    print(f"\nContinuous decoded:")
    steps     = 2000 + cont[0] * (25000 - 2000)
    bed_hour  = 20 + cont[1] * 6
    bed_min   = cont[2] * 59
    wake_hour = 5 + cont[3] * 5
    wake_min  = cont[4] * 59
    
    # FIX: Konsistentes Runden wie in get_action, BEVOR wir modulo 24 rechnen
    b_h_dec = int(round(bed_hour)) % 24
    b_m_dec = int(round(bed_min))
    w_h_dec = int(round(wake_hour)) % 24
    w_m_dec = int(round(wake_min))
    
    print(f"  steps={steps:.0f}  bed={b_h_dec:02d}:{b_m_dec:02d}  wake={w_h_dec:02d}:{w_m_dec:02d}")

    print(f"\nActivity probabilities:")
    for name, p in zip(ACTIVITY_NAMES, probs):
        bar = "█" * int(p * 30)
        print(f"  {name:<14} {p:.4f}  {bar}")

    print(f"\nDeterministic action:")
    action, log_prob, value2, _, _ = policy.get_action(scaled_obs, device, deterministic=True)
    activity = ACTIVITY_NAMES[int(np.argmax(action[1:7]))]
    
    # Hier ebenfalls sauberes Runden und Modulo
    b_h_act = int(round(action[7])) % 24
    b_m_act = int(round(action[8]))
    w_h_act = int(round(action[9])) % 24
    w_m_act = int(round(action[10]))
    
    print(f"  steps={int(action[0])}  activity={activity}  "
          f"bed={b_h_act:02d}:{b_m_act:02d}  wake={w_h_act:02d}:{w_m_act:02d}")
    print(f"  log_prob={log_prob.item():.4f}")

    print(f"\n7-step deterministic rollout (what the policy does over a week):")
    print(f"  {'Step':<5} {'Steps':>7} {'Activity':<14} {'Bed':>5} {'Wake':>5}")
    print(f"  {SEP2[:45]}")

    from src.rl_agent.rl_environment import ChronoOptEnv
    env = ChronoOptEnv(initial_state_data=state_array, model=models.lstm,
                       processor=models.processor, device=models.device)
    env.history = state_array.tolist()
    obs = scaled_obs.copy()
    policy.eval()
    for i in range(15):
        a, _, _, _, _ = policy.get_action(obs, device, deterministic=True)
        obs, reward, _, _, _ = env.step(a)
        act = ACTIVITY_NAMES[int(np.argmax(a[1:7]))]
        
        # Rollout Formatierung gefixt
        b_h_roll = int(round(a[7])) % 24
        b_m_roll = int(round(a[8]))
        w_h_roll = int(round(a[9])) % 24
        w_m_roll = int(round(a[10]))
        
        print(f"  {i+1:<5} {int(a[0]):>7}  {act:<14} "
              f"{b_h_roll:02d}:{b_m_roll:02d}  {w_h_roll:02d}:{w_m_roll:02d}  "
              f"reward={reward:.1f}")

    print(f"\nScores:")
    rec_score  = _score_policy_rollout(state_array, scaled_obs, models, n_steps=15)
    base_score = _score_fixed_action(state_array, state_array[-1, :11], models, n_steps=15)
    print(f"  policy rollout (5-step): {rec_score:.2f}")
    print(f"  baseline (yesterday):    {base_score:.2f}")
    print(f"  delta:                   {rec_score - base_score:.2f}")


def main():
    print(SEP)
    print("ChronoOpt — Policy Debug")
    print(SEP)

    models, _ = load_models()

    state_array,_, days_used = _build_real_state(models)
    scaled_obs = _scale_state(state_array, models)
    print(f"\nBuilt real state from {days_used} Garmin days.\n")

    check_a_observation(state_array, scaled_obs)
    check_b_scorer(state_array, scaled_obs, models)
    check_c_policy(state_array, scaled_obs, models)

    print(f"\n{SEP}")
    print("Debug complete.")
    print(SEP)


if __name__ == "__main__":
    main()