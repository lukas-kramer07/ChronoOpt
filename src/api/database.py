# src/api/database.py
# SQLite persistence layer for ChronoOpt.
#
# Three tables:
#   recommendations  — what the system suggested each day
#   outcomes         — what you actually did (partially auto-filled from Garmin)
#   model_log        — predicted vs actual score delta for calibration tracking


import sqlite3
import os
from typing import Optional
from datetime import date

DB_PATH = os.getenv("CHRONOOPT_DB_PATH", "data/chronoopt.db")


def get_connection() -> sqlite3.Connection:
    """Returns a connection with row_factory set so rows behave like dicts."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """
    Creates all tables if they don't already exist.
    Safe to call on every startup — uses CREATE TABLE IF NOT EXISTS.
    """
    with get_connection() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS recommendations (
                date                TEXT PRIMARY KEY,
                target_steps        INTEGER,
                activity_type       TEXT,
                bed_hour            INTEGER,
                bed_minute          INTEGER,
                wake_hour           INTEGER,
                wake_minute         INTEGER,
                predicted_score     REAL,
                baseline_score      REAL,
                score_delta         REAL,
                score_days          INTEGER,
                score_model         TEXT,
                policy_source       TEXT,
                created_at          TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS outcomes (
                date                TEXT PRIMARY KEY,
                actual_steps        INTEGER,
                actual_activity     TEXT,
                actual_bed_hour     INTEGER,
                actual_bed_minute   INTEGER,
                actual_wake_hour    INTEGER,
                actual_wake_minute  INTEGER,
                actual_score        REAL,
                followed_recommendation INTEGER,  -- 0/1 boolean
                notes               TEXT,
                logged_at           TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS model_log (
                date                TEXT PRIMARY KEY,
                predicted_score     REAL,
                actual_score        REAL,
                calibration_error   REAL,  -- actual - predicted
                created_at          TEXT DEFAULT (datetime('now'))
            );
        """)


def upsert_recommendation(rec: dict) -> None:
    """
    Inserts or replaces a daily recommendation record.
    Calling this twice on the same date overwrites the previous record —
    useful if the policy is re-run during the day after new Garmin data arrives.
    """
    with get_connection() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO recommendations
                (date, target_steps, activity_type, bed_hour, bed_minute,
                 wake_hour, wake_minute, predicted_score, baseline_score,
                 score_delta, policy_source)
            VALUES
                (:date, :target_steps, :activity_type, :bed_hour, :bed_minute,
                 :wake_hour, :wake_minute, :predicted_score, :baseline_score,
                 :score_delta, :policy_source)
        """, {
            "date":            rec["date"],
            "target_steps":    rec["recommendation"]["target_steps"],
            "activity_type":   rec["recommendation"]["activity_type"],
            "bed_hour":        rec["recommendation"]["bed_hour"],
            "bed_minute":      rec["recommendation"]["bed_minute"],
            "wake_hour":       rec["recommendation"]["wake_hour"],
            "wake_minute":     rec["recommendation"]["wake_minute"],
            "predicted_score": rec["predicted_scores"]["recommended"],
            "baseline_score":  rec["predicted_scores"]["baseline"],
            "score_delta":     rec["predicted_scores"]["delta"],
            "score_days":      rec["predicted_scores"]["days"],
            "score_model":     rec["predicted_scores"]["model"],
            "policy_source":   rec["policy_source"],
        })


def upsert_outcome(outcome: dict) -> None:
    """
    Inserts or updates what you actually did on a given day.
    Also computes and writes the calibration error to model_log if a
    predicted score exists for the same date.
    """
    with get_connection() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO outcomes
                (date, actual_steps, actual_activity, actual_bed_hour,
                 actual_bed_minute, actual_wake_hour, actual_wake_minute,
                 actual_score, followed_recommendation, notes)
            VALUES
                (:date, :actual_steps, :actual_activity, :actual_bed_hour,
                 :actual_bed_minute, :actual_wake_hour, :actual_wake_minute,
                 :actual_score, :followed_recommendation, :notes)
        """, {
            "date":                  outcome["date"],
            "actual_steps":          outcome.get("actual_steps"),
            "actual_activity":       outcome.get("actual_activity"),
            "actual_bed_hour":       outcome.get("actual_bed_hour"),
            "actual_bed_minute":     outcome.get("actual_bed_minute"),
            "actual_wake_hour":      outcome.get("actual_wake_hour"),
            "actual_wake_minute":    outcome.get("actual_wake_minute"),
            "actual_score":          outcome.get("actual_score"),
            "followed_recommendation": int(outcome["followed_recommendation"])
                                        if outcome.get("followed_recommendation") is not None else None,
            "notes":                 outcome.get("notes"),
        })

        # If we have both a predicted and actual score for this date, log calibration
        rec_row = conn.execute(
            "SELECT predicted_score FROM recommendations WHERE date = ?",
            (outcome["date"],)
        ).fetchone()

        if rec_row and outcome.get("actual_score") is not None:
            error = outcome["actual_score"] - rec_row["predicted_score"]
            conn.execute("""
                INSERT OR REPLACE INTO model_log
                    (date, predicted_score, actual_score, calibration_error)
                VALUES (?, ?, ?, ?)
            """, (outcome["date"], rec_row["predicted_score"],
                  outcome["actual_score"], error))


def get_history(limit: int = 30) -> list[dict]:
    """
    Returns the last `limit` days of joined recommendation + outcome data,
    ordered newest-first. Missing outcome fields come through as None.
    """
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT
                r.date,
                r.target_steps        AS recommended_steps,
                r.activity_type       AS recommended_activity,
                printf('%02d:%02d', r.bed_hour % 24, r.bed_minute)  AS recommended_bed,
                printf('%02d:%02d', r.wake_hour,      r.wake_minute) AS recommended_wake,
                r.predicted_score,
                o.actual_steps,
                o.actual_activity,
                o.actual_score,
                o.followed_recommendation AS followed,
                CASE
                    WHEN o.actual_score IS NOT NULL
                    THEN ROUND(o.actual_score - r.predicted_score, 2)
                    ELSE NULL
                END AS score_delta
            FROM recommendations r
            LEFT JOIN outcomes o ON r.date = o.date
            ORDER BY r.date DESC
            LIMIT ?
        """, (limit,)).fetchall()

        return [dict(row) for row in rows]


def get_recommendation_for_date(target_date: str) -> Optional[dict]:
    """
    Returns today's cached recommendation in RecommendationResponse-compatible
    shape, or None if no recommendation has been generated for this date yet.
    """
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM recommendations WHERE date = ?", (target_date,)
        ).fetchone()

    if not row:
        return None

    row = dict(row)
    return {
        "date": row["date"],
        "recommendation": {
            "target_steps":  row["target_steps"],
            "activity_type": row["activity_type"],
            "bed_hour":      row["bed_hour"],
            "bed_minute":    row["bed_minute"],
            "wake_hour":     row["wake_hour"],
            "wake_minute":   row["wake_minute"],
        },
        "predicted_scores": {
            "recommended": row["predicted_score"],
            "baseline":    row["baseline_score"],
            "delta":       row["score_delta"],
            "days":        row["score_days"],
            "model":       row["score_model"]
        },
        "state_days_used": 10,  # stored recommendations always used full state
        "policy_source":   row["policy_source"],
    }


def get_outcome_for_date(target_date: str) -> Optional[dict]:
    """Returns the outcome record for a specific date, or None if not logged."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM outcomes WHERE date = ?", (target_date,)
        ).fetchone()
        return dict(row) if row else None

def get_last_recommendation_date() -> Optional[str]:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT date FROM recommendations ORDER BY date DESC LIMIT 1"
        ).fetchone()
    return row["date"] if row else None
