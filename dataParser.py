# Uses unofficial garminconnect API for personal use only.
# May violate Garmin's Terms of Service.

from dotenv import load_dotenv
import os
from garminconnect import (
    Garmin,
    GarminConnectConnectionError,
    GarminConnectTooManyRequestsError,
    GarminConnectAuthenticationError
)


load_dotenv()

email = os.getenv("GARMIN_EMAIL")
password = os.getenv("GARMIN_PASSWORD")

try:
    client = Garmin(email, password)
    client.login()
except GarminConnectAuthenticationError:
    print("Wrong credentials or 2FA issue.")

print("\n\”")

client.get_heart_rates("2025-05-30")
client.get_body_battery("2025-05-30")
client.get_respiration_data("2025-05-30")
client.get_stress_data("2025-05-30")
print(client.get_sleep_data("2025-05-30"))

client.get



