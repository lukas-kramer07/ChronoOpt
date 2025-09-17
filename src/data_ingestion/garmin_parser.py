# src/data_ingestion/garmin_parser.py
# This module handles authentication with Garmin Connect and fetches
# various daily bio-sensor metrics.
#
# IMPORTANT: This uses an unofficial garminconnect API for personal use only.
# It may violate Garmin's Terms of Service. Use at your own discretion.

from dotenv import load_dotenv
import os
from datetime import datetime, timedelta
import json
import copy
from typing import Dict, Any

# Import GarminConnect library
from garminconnect import (
        Garmin,
        GarminConnectConnectionError,
        GarminConnectTooManyRequestsError,
        GarminConnectAuthenticationError
)

# --- Configuration ---
CACHE_DIR = "data/raw_data/" #run from root only
os.makedirs(CACHE_DIR, exist_ok=True) # Ensure cache directory exists

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
    Fetches a comprehensive set of daily Garmin metrics for a specified date. Caches the data for quicker access times.
    Args:
        date_str (str): The date in 'YYYY-MM-DD' format.
    Returns:
        dict: A dictionary containing various metrics for the day.
              Returns an empty dictionary if data cannot be fetched or client is not available.
    """
    #check whether data is already cached
    cache_file_path = os.path.join(CACHE_DIR, f"{date_str}.json") 
    if os.path.exists(cache_file_path):
        try:
            with open(cache_file_path, 'r') as f:
                data = json.load(f)
            print(f"  Fetching data for {date_str}... (from cache)")
            return data
        except json.JSONDecodeError:
            print(f"  Error reading cache file {cache_file_path} (corrupted JSON). Will re-fetch.")
            os.remove(cache_file_path) # Remove corrupted cache file
        except Exception as e:
            print(f"  Unexpected error reading cache file {cache_file_path}: {e}. Will re-fetch.")
            os.remove(cache_file_path) # Remove corrupted cache file

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
    
    # now cache daily data (only if the data isn't from today)
    if datetime.today().date() > datetime.strptime(date_str, "%Y-%m-%d").date():
        try:
            with open(cache_file_path, 'w') as f:
                json.dump(daily_data, f, indent=4) # Use indent for readability
        except IOError as e:
            print(f"  Warning: Could not save data to cache file {cache_file_path}: {e}")

    return daily_data

def _is_data_valid(data: Dict[str, Any]) -> bool:
    """
    Checks if the fetched data is valid and not just a set of empty/zero values.
    Returns True if at least one core metric is present and non-zero.
    """
    if not isinstance(data, dict):
        return False
    
    # Check for a few core metrics. If all are zero, the data is likely missing.
    daily_summary = data.get('dailySummaryData', {})
    if daily_summary.get('steps', 0) > 0:
        return True
    
    # Sleep data check
    sleep_data = data.get('sleepData', {})
    if sleep_data and sleep_data.get('dailySleepDTO', {}).get('sleepTimeSeconds', 0) > 0:
        return True
        
    return False

def get_historical_metrics(num_days: int) -> list:
    """
    Collects historical Garmin metrics for the specified number of past days.
    If data for a day is missing or invalid, it uses the data from the previous day.

    Args:
        num_days (int): The number of past days to collect data for.
    Returns:
        list: A list of dictionaries, each containing daily metrics.
              Sorted from oldest to newest.
    """
    historical_data = []
    yesterday = datetime.now().date() - timedelta(1) # Values for the current day typically not updated yet

    print(f"\nCollecting historical data for the last {num_days} days...")
    for i in range(num_days):
        current_date = yesterday - timedelta(days=i)
        date_str = current_date.strftime("%Y-%m-%d")
        daily_metrics = get_daily_metrics(date_str)

        if _is_data_valid(daily_metrics):
            historical_data.append(daily_metrics)
        else:
            print(f"  Data for {date_str} appears invalid/missing (key metrics are 0). Attempting to fill from previous day.")
            if historical_data:
                # Use a deep copy to avoid modifying the previous day's original data
                last_day_data = copy.deepcopy(historical_data[-1])
                last_day_data['date'] = date_str # Update the date to the current day
                historical_data.append(last_day_data)
                print(f"  Successfully filled data for {date_str} with data from {historical_data[-2]['date']}.")
            else:
                print(f"  No valid previous day's data available to fill the gap. Skipping {date_str}.")

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
                # Dive deeper into specific nested sleep data keys
                daily_sleep_dto = day_data['sleepData'].get('dailySleepDTO', {})
                if isinstance(daily_sleep_dto, dict):
                    print(f"    dailySleepDTO Keys: {list(daily_sleep_dto.keys())}")
                rem_sleep_data = day_data['sleepData'].get('remSleepData', {})
                if isinstance(rem_sleep_data, dict):
                    print(f"    remSleepData Keys: {list(rem_sleep_data.keys())}")
                sleep_levels = day_data['sleepData'].get('sleepLevels', {})
                if isinstance(sleep_levels, dict):
                    print(f"    sleepLevels Keys: {list(sleep_levels.keys())}")
            else:
                print(f"  Sleep Data: (Not a dictionary, cannot show keys directly)")
            
            # Daily Summary Data
            print(f"  Type of dailySummaryData: {type(day_data['dailySummaryData'])}")
            if isinstance(day_data['dailySummaryData'], dict):
                print(f"  Daily Summary Data Keys: {list(day_data['dailySummaryData'].keys())}")
            else:
                print(f"  Daily Summary Data: (Not a dictionary, cannot show keys directly)")

            print(f"  ----------------------------------------")
    else:
        print("No historical data fetched.")