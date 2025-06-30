# src/data_ingestion/garmin_parser.py
# This module handles authentication with Garmin Connect and fetches
# various daily bio-sensor metrics.
#
# IMPORTANT: This uses an unofficial garminconnect API for personal use only.
# It may violate Garmin's Terms of Service. Use at your own discretion.

from dotenv import load_dotenv
import os
from datetime import datetime, timedelta
import json # For debugging/pretty printing raw data if needed

# Import GarminConnect library
from garminconnect import (
        Garmin,
        GarminConnectConnectionError,
        GarminConnectTooManyRequestsError,
        GarminConnectAuthenticationError
)

# --- Configuration ---
# Load environment variables from .env file
load_dotenv()
GARMIN_EMAIL = os.getenv("GARMIN_EMAIL")
GARMIN_PASSWORD = os.getenv("GARMIN_PASSWORD")

# --- Garmin Client Management ---
_garmin_client = None # Private variable to hold the authenticated client instance

def get_garmin_client():
    """
    Authenticates with Garmin Connect and returns the client object.
    Uses a singleton pattern to avoid re-authenticating multiple times.
    """
    global _garmin_client

    if _garmin_client is None:
        if not GARMIN_EMAIL or not GARMIN_PASSWORD:
            print("Error: Garmin credentials (GARMIN_EMAIL, GARMIN_PASSWORD) not set in .env file.")
            print("Please ensure your .env file is correctly configured.")
            return None

        try:
            print("Attempting to log into Garmin Connect...")
            _garmin_client = Garmin(GARMIN_EMAIL, GARMIN_PASSWORD)
            _garmin_client.login()
            print("Successfully logged into Garmin Connect.")
        except GarminConnectAuthenticationError:
            print("Garmin Authentication Error: Wrong credentials or 2FA issue.")
            _garmin_client = None
        except GarminConnectTooManyRequestsError:
            print("Garmin API Error: Too many requests. Please wait and try again later.")
            _garmin_client = None
        except GarminConnectConnectionError as e:
            print(f"Garmin Connection Error: {e}. Check your internet connection.")
            _garmin_client = None
        except Exception as e:
            print(f"An unexpected error occurred during Garmin login: {e}")
            _garmin_client = None
    return _garmin_client

def get_daily_metrics(date_str: str) -> dict:
    """
    Fetches a comprehensive set of daily Garmin metrics for a specified date.
    Args:
        date_str (str): The date in 'YYYY-MM-DD' format.
    Returns:
        dict: A dictionary containing various metrics for the day.
              Returns an empty dictionary if data cannot be fetched or client is not available.
    """
    client = get_garmin_client()
    if not client:
        print(f"Garmin client not available. Cannot fetch data for {date_str}.")
        return {}

    daily_data = {
        'date': date_str,
        'sleepData': {},
        'heartRateData': {},
        'respirationData': {},
        'stressData': {},
        'stepsData': {},
        'activityData': [], # List of activities for the day
        'bodyBatteryData': {}
    }

    print(f"  Fetching data for {date_str}...")
    try:
        # Daily Summary Data (contains steps, resting heart rate, etc.)
        daily_summary_data = client.get_user_summary(date_str)
        daily_data['dailySummaryData'] = daily_summary_data
        if daily_summary_data:
            daily_data['stepsData'] = {'totalSteps': daily_summary_data.get('steps', 0)}
            # We can also get restingHeartRate from here if available
            # For now, keeping heartRateData separate as it might contain more detail
            # daily_data['heartRateData']['restingHeartRate'] = daily_summary_data.get('restingHeartRate', 0)


        # Sleep Data
        sleep_data = client.get_sleep_data(date_str)
        daily_data['sleepData'] = sleep_data

        # Heart Rate Data (summary for the day)
        heart_rate_data = client.get_heart_rates(date_str)
        daily_data['heartRateData'] = heart_rate_data

        # Respiration Data
        respiration_data = client.get_respiration_data(date_str)
        daily_data['respirationData'] = respiration_data

        # Stress Data
        stress_data = client.get_stress_data(date_str)
        daily_data['stressData'] = stress_data

        # Body Battery Data
        # Changed to use get_body_battery_data as per common library usage
        body_battery_data = client.get_body_battery(date_str) # This returns a list of dictionaries
        daily_data['bodyBatteryData'] = body_battery_data

        # Activity Data (list of activities for the day)
        # This will return a list of dictionaries, each representing an activity
        activities = client.get_activities_by_date(date_str, date_str)
        daily_data['activityData'] = activities

    except GarminConnectTooManyRequestsError:
        print(f"  Garmin API Error: Too many requests for {date_str}. Skipping this date.")
        # Return empty data for this date to indicate failure
        return {}
    except Exception as e:
        print(f"  Error fetching data for {date_str}: {e}. Skipping this date.")
        # Return empty data for this date to indicate failure
        return {}

    return daily_data

def get_historical_metrics(num_days: int) -> list:
    """
    Collects historical Garmin metrics for the specified number of past days.
    Args:
        num_days (int): The number of past days to collect data for.
    Returns:
        list: A list of dictionaries, each containing daily metrics.
              Sorted from oldest to newest.
    """
    historical_data = []
    today = datetime.now().date()

    print(f"\nCollecting historical data for the last {num_days} days...")
    for i in range(num_days):
        current_date = today - timedelta(days=i)
        date_str = current_date.strftime("%Y-%m-%d")
        daily_metrics = get_daily_metrics(date_str)
        if daily_metrics:
            historical_data.append(daily_metrics)
        else:
            print(f"  No data retrieved for {date_str}. Skipping.")

    # Sort data oldest to newest for time series consistency
    historical_data.sort(key=lambda x: x['date'])
    return historical_data

# --- Main execution for testing purposes ---
if __name__ == "__main__":
    # Example usage: Fetch data for the last 3 days
    print("--- Running garmin_parser.py in standalone test mode ---")
    num_days_to_fetch = 3
    historical_metrics = get_historical_metrics(num_days_to_fetch)

    if historical_metrics:
        print(f"\n--- Successfully fetched data for {len(historical_metrics)} days ---")
        for day_data in historical_metrics:
            print(f"\nDate: {day_data['date']}")
            print(f"  --- Raw Data Structures for Debugging (Keys Only) ---")

            # Sleep Data
            print(f"  Type of sleepData: {type(day_data['sleepData'])}")
            if isinstance(day_data['sleepData'], dict):
                print(f"  Sleep Data Keys: {list(day_data['sleepData'].keys())}")
            else:
                print(f"  Sleep Data: (Not a dictionary, cannot show keys directly)")

            # Heart Rate Data
            print(f"  Type of heartRateData: {type(day_data['heartRateData'])}")
            if isinstance(day_data['heartRateData'], dict):
                print(f"  Heart Rate Data Keys: {list(day_data['heartRateData'].keys())}")
            else:
                print(f"  Heart Rate Data: (Not a dictionary, cannot show keys directly)")

            # Respiration Data
            print(f"  Type of respirationData: {type(day_data['respirationData'])}")
            if isinstance(day_data['respirationData'], dict):
                print(f"  Respiration Data Keys: {list(day_data['respirationData'].keys())}")
            else:
                print(f"  Respiration Data: (Not a dictionary, cannot show keys directly)")

            # Stress Data
            print(f"  Type of stressData: {type(day_data['stressData'])}")
            if isinstance(day_data['stressData'], dict):
                print(f"  Stress Data Keys: {list(day_data['stressData'].keys())}")
            else:
                print(f"  Stress Data: (Not a dictionary, cannot show keys directly)")

            # Steps Data
            print(f"  Type of stepsData: {type(day_data['stepsData'])}")
            if isinstance(day_data['stepsData'], dict):
                print(f"  Steps Data Keys: {list(day_data['stepsData'].keys())}")
            else:
                print(f"  Steps Data: (Not a dictionary, cannot show keys directly)")

            # Body Battery Data (often a list of dicts)
            print(f"  Type of bodyBatteryData: {type(day_data['bodyBatteryData'])}")
            if isinstance(day_data['bodyBatteryData'], list) and len(day_data['bodyBatteryData']) > 0 and isinstance(day_data['bodyBatteryData'][0], dict):
                print(f"  Body Battery Data Keys (first item): {list(day_data['bodyBatteryData'][0].keys())}")
            elif isinstance(day_data['bodyBatteryData'], dict): # In case it's a dict directly
                 print(f"  Body Battery Data Keys: {list(day_data['bodyBatteryData'].keys())}")
            else:
                print(f"  Body Battery Data: (Not a dictionary or list of dictionaries, cannot show keys directly)")

            # Activity Data (always a list of dicts)
            print(f"  Type of activityData: {type(day_data['activityData'])}")
            if isinstance(day_data['activityData'], list) and len(day_data['activityData']) > 0 and isinstance(day_data['activityData'][0], dict):
                print(f"  Activity Data Keys (first item): {list(day_data['activityData'][0].keys())}")
            else:
                print(f"  Activity Data: (Empty list or not a list of dictionaries, cannot show keys directly)")

            # Daily Summary Data
            print(f"  Type of dailySummaryData: {type(day_data['dailySummaryData'])}")
            if isinstance(day_data['dailySummaryData'], dict):
                print(f"  Daily Summary Data Keys: {list(day_data['dailySummaryData'].keys())}")
            else:
                print(f"  Daily Summary Data: (Not a dictionary, cannot show keys directly)")

            print(f"  ----------------------------------------")

            # Simplified print for common metrics (will be N/A if raw data is empty)
            print(f"  Sleep Time (seconds): {day_data['sleepData'].get('sleepTimeSeconds', 'N/A')}")
            print(f"  Total Steps: {day_data['stepsData'].get('totalSteps', 'N/A')}")
            print(f"  Avg Heart Rate: {day_data['heartRateData'].get('heartRate', {}).get('avg', 'N/A')}")
            print(f"  Avg Respiration Rate: {day_data['respirationData'].get('respiration', {}).get('avg', 'N/A')}")
            print(f"  Avg Stress: {day_data['stressData'].get('stress', {}).get('avg', 'N/A')}")
            # Corrected access for bodyBatteryData, assuming it's a list of dicts
            if isinstance(day_data['bodyBatteryData'], list) and len(day_data['bodyBatteryData']) > 0:
                print(f"  Body Battery (Total): {day_data['bodyBatteryData'][0].get('total', 'N/A')}")
            else:
                print(f"  Body Battery (Total): N/A (No valid data or empty list)")

            print(f"  Activities Count: {len(day_data['activityData'])}")
    else:
        print("\nFailed to fetch any historical data. Check credentials, .env, and internet connection.")