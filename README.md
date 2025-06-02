# Circadian Rhythm Optimization System – Project Outline

## 🧠 Objective

To prototype an intelligent system that helps optimize personal circadian rhythms by leveraging bio-sensor data (e.g., Garmin) and reinforcement learning. The system simulates and learns optimal behaviors to improve sleep quality, recovery, and well-being.

---

## 🎯 Final Goal

Create a generalizable, adaptive agent that:

- Learns what you personally need for optimal sleep and recovery

- Makes evidence-based, actionable suggestions

- Is backed by real physiology data or guided self-report

## 🧩 Conceptual Architecture

### 1. **Reward Signal: Sleep Score Proxy**

- Garmin’s sleep score combines:
  - Total sleep duration
  - Deep and REM sleep ratio
  - Restlessness
  - Recovery indicators (heart rate, respiratory rate)
- We emulate this by using:
  - `sleepTimeSeconds`
  - `deepSleepSeconds`, `remSleepSeconds`, `awakeSleepSeconds`
  - `restlessMomentsCount`
  - `restingHeartRate`
  - `averageRespirationValue`

If sensor data is not available (e.g., for external users), we fallback to **self-reported sleep scores (1–100)**.

---

### 2. **State Space Definition (Input)**

Each day is modeled based on features from prior *x* days:

- `avgStress` (from body metrics or modeled if unavailable)
- `totalSteps`
- `avgHeartRate`
- `avgRespirationRate`
- `previousSleepScore`
- `activityType` flags (e.g., strength training, yoga, rest day)
- `wakeTime`, `bedTime`
- Additional behavioral flags (e.g., caffeine, screen exposure – optional for future)

→ State vector: a time series (length `x`) of these daily features.

---

### 3. **Prediction Model**

Develop a **supervised prediction model** to estimate next-day **sleep quality** (sleep score proxy) based on the previous `x` days.

- Architecture: LSTM / Temporal Convolutional Network (TCN) / Transformer
- Loss function: Mean Squared Error (MSE) between predicted and actual sleep score
- Purpose: Enables simulation before RL agent is trained on live data

---

### 4. **Action Space for Agent**

Agent's policy space should be simple but impactful:

#### Initial action space:
- `GoToSleep(hour)`
- `WakeUp(hour)`
- `DoActivity(type)`  
  (`None`, `Strength`, `Cardio`, `Yoga`, `Stretching`)


Later:
- Adjust diet window
- Caffein intake 
- Meditation or breathwork

→ Actions are constrained by real-life blocks (e.g., user has school/work from 8–16h).

---

### 5. **Reinforcement Learning Agent**

- Use a general policy initially, trained on aggregated population data (simulated or anonymized real data)
- Personalization phase:  
  The agent gradually adapts to the **individual user** via **online learning** or fine-tuning on the user’s historical and incoming data.

---

## 🧪 Simulation Loop

```text
Past state (x days) → Model → Predicted sleep quality
     ↓
 Agent selects action → Environment applies change → Next state
     ↓
 Reward (simulated or self-reported) computed
     ↓
 Update agent
```


## 🛤️ Roadmap

Circadian Rhythm Optimization System – Roadmap

1. Data Collection & Preparation  
   - Gather and organize Garmin bio-sensor data  [DONE]  
   - Clean, preprocess, and format for modeling  [DONE]

2. Exploratory Data Analysis (EDA)  
   - Extract key correlations between metrics and sleep quality  [DONE]  
   - Identify missing data patterns and fallback strategies  [DONE]

3. State Space & Feature Engineering  
   - Define input features and time window `x`  
   - Encode activity and behavioral flags

4. Prediction Model Development  
   - Prototype LSTM / TCN / Transformer architectures  
   - Train and validate sleep quality prediction model

5. Reward Signal & Proxy Calibration  
   - Calibrate sleep score proxy against Garmin scores  
   - Implement fallback for self-reported data

6. Action Space Definition & Constraints  
   - Finalize agent actions, including real-life constraints  
   - Simulate action effects on state transitions

7. Reinforcement Learning Agent  
   - Train general policy on aggregated/simulated data  
   - Develop personalization strategy (online learning/fine-tuning)

8. Simulation Environment & Loop  
   - Build full simulation pipeline integrating prediction and RL agent  
   - Validate reward feedback and policy updates

9. Evaluation & Iteration  
   - Test agent recommendations on holdout user data  
   - Refine models and action policies based on results

10. Deployment & User Interface  
    - Design user feedback loop for real-time adaptation  
    - Implement visualization and actionable insights
