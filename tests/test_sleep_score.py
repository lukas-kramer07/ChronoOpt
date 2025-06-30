# tests/test_sleep_score.py
# A simple test file to calculate and print the sleep score proxy for yesterday.

from datetime import datetime, timedelta
import json # For pretty printing raw data if needed

# Import functions from our data ingestion and feature engineering modules
from src.data_ingestion.garmin_parser import get_daily_metrics
from src.features.utils import calculate_sleep_score_proxy

def test_yesterday_sleep_score():
    """
    Fetches todays's raw Garmin metrics, extracts sleep data,
    calculates the sleep score proxy, and prints the result.
    """
    print("--- Running Sleep Score Test for Yesterday ---")

    # Calculate yesterday's date
    yesterday = datetime.now().date() - timedelta(3)
    yesterday_str = yesterday.strftime("%Y-%m-%d")

    print(f"Fetching data for: {yesterday_str}")

    # 1. Fetch raw daily metrics for yesterday
    raw_metrics = get_daily_metrics(yesterday_str)

    if raw_metrics:
        print("\n--- Raw Sleep Data for Yesterday (for verification) ---")
        sleep_data = raw_metrics.get('sleepData', {})
        if sleep_data:
            print(f"Type of sleepData: {type(sleep_data)}")
            print(f"Sleep Data Keys: {list(sleep_data.keys())}")
            daily_sleep_dto = sleep_data.get('dailySleepDTO', {})
            if isinstance(daily_sleep_dto, dict):
                print(f"  dailySleepDTO Keys: {list(daily_sleep_dto.keys())}")
            print(f"  Total Sleep Seconds (from dailySleepDTO): {daily_sleep_dto.get('sleepTimeSeconds', 'N/A')}")
            print(f"  Deep Sleep Seconds (from dailySleepDTO): {daily_sleep_dto.get('deepSleepSeconds', 'N/A')}")
            print(f"  REM Sleep Seconds (from dailySleepDTO): {daily_sleep_dto.get('remSleepSeconds', 'N/A')}")
            print(f"  Awake Sleep Seconds (from dailySleepDTO): {daily_sleep_dto.get('awakeSleepSeconds', 'N/A')}")
            print(f"  Restless Moments Count: {sleep_data.get('restlessMomentsCount', 'N/A')}")
            print(f"  Resting Heart Rate (from sleepData): {sleep_data.get('restingHeartRate', 'N/A')}")
        else:
            print("No sleep data found for yesterday.")

        # 2. Calculate the sleep score proxy
        if sleep_data:
            sleep_score = calculate_sleep_score_proxy(sleep_data)
            print(f"\n--- Calculated Sleep Score Proxy for {yesterday_str}: {sleep_score:.2f} ---")
        else:
            print("\nCannot calculate sleep score proxy: No sleep data available.")
    else:
        print(f"Failed to fetch any raw metrics for {yesterday_str}. Check garmin_parser.py logs.")

# Run the test function when the script is executed
if __name__ == "__main__":
    test_yesterday_sleep_score()
