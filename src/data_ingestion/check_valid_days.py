# src/data_ingestion/check_valid_days.py
# This module is a utility module to get an understanding of the data quality. It analyses for which days there is partially or fully missing data 
# and outputs contiguos valid ranges of data
import os
import json
from datetime import datetime

CACHE_DIR = "data/raw_data/"

def check_day(data: dict) -> dict:
    """Returns a dict with validity flags for each key metric."""
    result = {
        'sleep_seconds': None,
        'steps': None,
        'resting_hr': None,
        'avg_stress': None,
        'valid': False,
    }

    # Sleep
    sleep_dto = data.get('sleepData', {}).get('dailySleepDTO', {})
    sleep_seconds = sleep_dto.get('sleepTimeSeconds') if isinstance(sleep_dto, dict) else None
    result['sleep_seconds'] = sleep_seconds if sleep_seconds and sleep_seconds > 0 else None

    # Steps + other summary metrics
    summary = data.get('dailySummaryData', {})
    if isinstance(summary, dict):
        steps = summary.get('totalSteps')
        result['steps'] = steps if steps and steps > 0 else None

        rhr = summary.get('restingHeartRate')
        result['resting_hr'] = rhr if rhr and rhr > 0 else None

        stress = summary.get('averageStressLevel')
        result['avg_stress'] = stress if stress and stress > 0 else None

    # A day is valid if at least sleep OR steps is present
    result['valid'] = bool(result['sleep_seconds'] or result['steps'])
    return result


def main():
    if not os.path.exists(CACHE_DIR):
        print(f"Cache directory not found: {CACHE_DIR}")
        return

    files = sorted([f for f in os.listdir(CACHE_DIR) if f.endswith('.json')])

    if not files:
        print("No cached files found.")
        return

    print(f"Scanning {len(files)} cached days...\n")

    valid_days = []
    invalid_days = []
    partial_days = []

    for filename in files:
        date_str = filename.replace('.json', '')
        filepath = os.path.join(CACHE_DIR, filename)

        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            invalid_days.append((date_str, f"FILE ERROR: {e}"))
            continue

        metrics = check_day(data)

        has_sleep = bool(metrics['sleep_seconds'])
        has_steps = bool(metrics['steps'])
        has_hr = bool(metrics['resting_hr'])
        has_stress = bool(metrics['avg_stress'])

        all_present = has_sleep and has_steps and has_hr and has_stress
        none_present = not (has_sleep or has_steps or has_hr or has_stress)

        if all_present:
            valid_days.append(date_str)
        elif none_present:
            invalid_days.append((date_str, "ALL METRICS NULL"))
        else:
            flags = []
            if not has_sleep: flags.append("NO_SLEEP")
            if not has_steps: flags.append("NO_STEPS")
            if not has_hr:    flags.append("NO_RHR")
            if not has_stress: flags.append("NO_STRESS")
            partial_days.append((date_str, ", ".join(flags)))

    # --- Report ---
    total = len(files)
    print(f"{'='*55}")
    print(f"  TOTAL DAYS SCANNED : {total}")
    print(f"  FULLY VALID        : {len(valid_days)}")
    print(f"  PARTIAL            : {len(partial_days)}")
    print(f"  FULLY INVALID      : {len(invalid_days)}")
    print(f"{'='*55}\n")

    if valid_days:
        print(f"VALID RANGE: {valid_days[0]} → {valid_days[-1]}")
        print(f"  ({len(valid_days)} valid days)\n")

    if partial_days:
        print(f"PARTIAL DAYS ({len(partial_days)}):")
        for date, reason in partial_days:
            print(f"  {date}  [{reason}]")
        print()

    if invalid_days:
        print(f"INVALID DAYS ({len(invalid_days)}):")
        for date, reason in invalid_days:
            print(f"  {date}  [{reason}]")
        print()

    # --- Identify contiguous valid ranges ---
    print("CONTIGUOUS VALID RANGES:")
    if not valid_days:
        print("  None found.")
        return

    ranges = []
    start = valid_days[0]
    prev = valid_days[0]

    for date_str in valid_days[1:]:
        prev_date = datetime.strptime(prev, "%Y-%m-%d")
        curr_date = datetime.strptime(date_str, "%Y-%m-%d")
        if (curr_date - prev_date).days == 1:
            prev = date_str
        else:
            ranges.append((start, prev))
            start = date_str
            prev = date_str
    ranges.append((start, prev))

    for r_start, r_end in ranges:
        d_start = datetime.strptime(r_start, "%Y-%m-%d")
        d_end = datetime.strptime(r_end, "%Y-%m-%d")
        days = (d_end - d_start).days + 1
        print(f"  {r_start} → {r_end}  ({days} days)")


if __name__ == "__main__":
    main()