# src/api/models.py
# Pydantic models for ChronoOpt API request and response types.

from pydantic import BaseModel, Field
from typing import Optional
from datetime import date


class DailyRecommendation(BaseModel):
    """
    What the policy recommends you do today.
    All time values are in 24h format, hours and minutes split for easy display.
    """
    target_steps: int = Field(..., description="Recommended daily step count")
    activity_type: str = Field(..., description="Recommended activity: Strength, Cardio, Yoga, Stretching, OtherActivity, or NoActivity")
    bed_hour: int = Field(..., description="Recommended bed time hour (24h, may be > 23 for post-midnight, display mod 24)")
    bed_minute: int = Field(..., description="Recommended bed time minute")
    wake_hour: int = Field(..., description="Recommended wake time hour (24h)")
    wake_minute: int = Field(..., description="Recommended wake time minute")

    @property
    def bed_time_display(self) -> str:
        return f"{self.bed_hour % 24:02d}:{self.bed_minute:02d}"

    @property
    def wake_time_display(self) -> str:
        return f"{self.wake_hour:02d}:{self.wake_minute:02d}"


class SleepScoreComparison(BaseModel):
    """
    Predicted sleep scores for the recommended behaviour vs doing what you did yesterday.
    The gap between these two numbers is the core value proposition of ChronoOpt.
    """
    recommended: float = Field(..., description="Predicted sleep score (0-100) if you follow the recommendation")
    baseline: float = Field(..., description="Predicted sleep score (0-100) if you repeat yesterday's behaviour")
    delta: float = Field(..., description="Improvement from following the recommendation")


class RecommendationResponse(BaseModel):
    """Full response for GET /recommend."""
    date: str = Field(..., description="Date this recommendation is for (YYYY-MM-DD)")
    recommendation: DailyRecommendation
    predicted_scores: SleepScoreComparison
    state_days_used: int = Field(..., description="How many days of real Garmin data were used to build the state")
    policy_source: str = Field(..., description="'trained_policy' or 'deterministic_fallback'")


class OutcomeLog(BaseModel):
    """
    POST /log-outcome — what you actually did today.
    All fields optional because you might not know/track everything.
    """
    date: str = Field(..., description="Date of the logged activity (YYYY-MM-DD)")
    actual_steps: Optional[int] = None
    actual_activity: Optional[str] = None
    actual_bed_hour: Optional[int] = None
    actual_bed_minute: Optional[int] = None
    actual_wake_hour: Optional[int] = None
    actual_wake_minute: Optional[int] = None
    followed_recommendation: Optional[bool] = Field(None, description="Did you roughly follow the recommendation?")
    notes: Optional[str] = None


class HistoryEntry(BaseModel):
    """One row in the GET /history response."""
    date: str
    recommended_steps: Optional[int] = None
    recommended_activity: Optional[str] = None
    recommended_bed: Optional[str] = None
    recommended_wake: Optional[str] = None
    predicted_score: Optional[float] = None
    actual_steps: Optional[int] = None
    actual_activity: Optional[str] = None
    actual_score: Optional[float] = None  # computed from next-morning Garmin data
    followed: Optional[bool] = None
    score_delta: Optional[float] = None   # actual - predicted, to track model calibration


class HealthResponse(BaseModel):
    """GET /health — operational status of the system."""
    status: str
    lstm_loaded: bool
    policy_loaded: bool
    processor_fitted: bool
    last_garmin_fetch_date: Optional[str] = None
    garmin_days_available: int
    policy_path: str
    message: str
