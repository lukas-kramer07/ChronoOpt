# ChronoOpt

Personal circadian rhythm optimisation system. Ingests Garmin biometric data, trains an LSTM to predict next-day physiological responses, and uses a PPO reinforcement learning agent to recommend daily behaviours that maximise sleep quality.

Built as a full ML pipeline from raw wearable data to a served daily recommendation, with an online personalisation loop that adapts to the individual user over time.

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange?style=flat-square)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green?style=flat-square)

---

## Architecture

```mermaid
flowchart TD
    subgraph pipeline ["Offline Training"]
        A[Garmin Cache] --> B[feature_engineer.py]
        B -->|23 features/day| C[DataProcessor\ndual StandardScalers]
        C -->|10x23 sequences| D[LSTM PredictionModel\ninput 23, output 12]
        D -->|lstm_chronoopt.pt| E
        C --> E[ChronoOptEnv\nRL Environment]
        E -->|sleep score reward| F[PPO Agent\nRolloutBuffer + GAE]
        F -->|policy gradient| G[PolicyNetwork\n2-headed MLP + value head]
        G -->|policy.pt| H
    end

    subgraph serving ["Daily Inference"]
        H[FastAPI] -->|loads policy + EDMD + LSTM| I[GET /recommend]
        I -->|real 10-day Garmin state| J[Recommendation\nsteps, activity, bed, wake]
        J -->|5-day policy rollout| K[EDMD scorer\nLSTM fallback if no EDMD]
        J --> L[POST /log-outcome]
        L --> M[(SQLite\nrecommendations, outcomes)]
        M -->|actual sleep score| N[Nightly loop\nPPO update + EDMD refit ×7d]
        N --> G
        N -->|refit| K
    end
```

**Inference flow:** the trained policy maps the last 10 days of real Garmin state to today's recommended action. Both the recommendation and the baseline (repeat yesterday) are scored via identical 5-day rollouts — using EDMD as the dynamics model when available, falling back to the LSTM otherwise.

---

## Quick start

```bash
git clone https://github.com/lukas-kramer07/ChronoOpt
cd ChronoOpt

pip install -r requirements.txt

cp .env.example .env        # add GARMIN_EMAIL and GARMIN_PASSWORD

# Train the LSTM prediction model
python -m src.models.train_pred_model

# Train the PPO agent
python -m src.rl_agent.train_agent

# Serve the dashboard
uvicorn src.api.main:app --reload
# http://localhost:8000
# http://localhost:8000/docs   (interactive API)
```

---

## How it works

### 1. Dynamics models

An LSTM trained on personal Garmin data predicts the next day's 12 physiological features from a 10-day history window. Used as the world model during offline RL training.

EDMD (Extended Dynamic Mode Decomposition) is fit on real Garmin data by the online loop and serves as the primary scoring model at inference time. It takes a single scaled 23-feature day vector as input rather than a sequence, making it faster and more transparent than the LSTM. The LSTM is used as a fallback when no EDMD model is loaded.

### 2. RL agent (PPO)

A two-headed policy network maps the scaled observation (10 x 23 = 230 inputs) to a continuous action (steps, bed time, wake time) and a categorical action (activity type). Trained with Proximal Policy Optimisation on `ChronoOptEnv`, where the reward is a sleep score proxy (0-100) computed from LSTM-predicted sleep metrics.

`DeterministicEnv` provides an analytical world model for validating the training loop without LSTM dependency.

### 3. Scoring

At inference time, both the recommendation and the baseline are scored using identical 5-day rollouts. The recommendation rollout lets the policy re-observe and re-act at each step. EDMD is the preferred dynamics model for scoring; the LSTM is used as a fallback. The delta is the headline metric in the dashboard.

### 4. Online personalisation (in progress)

Each logged outcome provides a real (state, action, reward) tuple. A nightly APScheduler job at 3 AM runs a single online PPO update using the actual sleep score as reward, and refits the EDMD model on all available real data every 7 days. The offline training is the prior; daily logging is the personalisation signal.

---

## Project structure

```
ChronoOpt/
├── src/
│   ├── config.py
│   ├── data_ingestion/
│   │   └── garmin_parser.py         Garmin API + local JSON cache
│   ├── features/
│   │   ├── feature_engineer.py      raw metrics to 23-feature daily dict
│   │   └── utils.py                 calculate_sleep_score_proxy()
│   ├── models/
│   │   ├── data_processor.py        dual StandardScalers, flatten/reconstruct
│   │   ├── prediction_model.py      LSTM, input=23, output=12
│   │   └── train_pred_model.py      training pipeline
│   └── rl_agent/
│       ├── rl_environment.py        ChronoOptEnv (LSTM world model)
│       ├── deterministic_environment.py  analytical env for training validation
│       ├── policy_network.py        two-headed MLP + value head
│       ├── ppo_agent.py             RolloutBuffer + PPOAgent
│       └── train_agent.py           full PPO training pipeline
├── src/api/
│   ├── main.py                      FastAPI app, lifespan, endpoints
│   ├── inference.py                 ModelBundle, recommendation logic
│   ├── database.py                  SQLite: recommendations, outcomes, model_log
│   ├── models.py                    Pydantic request/response types
│   └── static/index.html            dashboard
├── data/
│   └── raw_data/                    Garmin JSON cache (git-ignored)
├── notebooks/
│   └── training_walkthrough.ipynb   LSTM curves, PPO reward progression
└── check_data_quality.py
```

---

## Feature vector

23 features per day, split into two groups:

| Group            | Indices | Features                                                                                                                 |
| ---------------- | ------- | ------------------------------------------------------------------------------------------------------------------------ |
| Agent-controlled | 0-10    | total_steps, activity flags x6, bed_hour, bed_minute, wake_hour, wake_minute                                             |
| Model-predicted  | 11-22   | avg_hr, resting_hr, respiration, stress, body_battery, total/deep/REM/awake sleep, restlessness, sleep_stress, sleep_rhr |

---

## API

Interactive docs at `http://localhost:8000/docs`.

| Endpoint                      | Description                               |
| ----------------------------- | ----------------------------------------- |
| `GET /health`                 | model load status, system check           |
| `GET /recommend`              | today's recommendation + predicted scores |
| `GET /recommend?refresh=true` | force re-run inference                    |
| `POST /log-outcome`           | log actual daily behaviour                |
| `GET /history?days=30`        | recommendation vs outcome history         |

---

## Stack

| Layer          | Technology                               |
| -------------- | ---------------------------------------- |
| Data ingestion | `garminconnect` (unofficial API)         |
| ML             | PyTorch 2.x, LSTM + PPO                  |
| Dynamics model | EDMD (polynomial features, scikit-learn) |
| API            | FastAPI, Pydantic, SQLite                |
| Frontend       | Vanilla JS, Chart.js                     |
| Training       | CUDA                                     |

---

## Notes

Uses the unofficial `garminconnect` Python library for personal use only. May conflict with Garmin's Terms of Service. Use at your own discretion.
