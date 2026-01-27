import argparse
import csv
import math
from collections import defaultdict
from datetime import datetime, timedelta

from parcae import Parcae


def parse_csv(path):
    timestamps = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if fieldnames is None or "timestamp" not in fieldnames:
            raise ValueError("! CSV must have a 'timestamp' column")

        for row in reader:
            timestamps.append(row["timestamp"])

    return timestamps


def minutes_since_midnight(dt):
    return dt.hour * 60 + dt.minute


def format_hm(minutes):
    h = (minutes // 60) % 24
    m = minutes % 60
    return f"{h:02d}:{m:02d}"


def circular_mean_minutes(values):
    angles = [2 * math.pi * v / 1440.0 for v in values]

    x = sum(math.cos(a) for a in angles)
    y = sum(math.sin(a) for a in angles)

    if x == 0 and y == 0:
        return int(values[0])

    mean_angle = math.atan2(y, x)
    if mean_angle < 0:
        mean_angle += 2 * math.pi

    mean_minutes = int(round(mean_angle * 1440.0 / (2 * math.pi)))
    return mean_minutes % 1440


def main():
    parser = argparse.ArgumentParser(prog="parcae")
    parser.add_argument("csv", help="CSV file with a 'timestamp' column")
    parser.add_argument("-v", "--version", action="version", version="%(prog)s 0.1.0")
    args = parser.parse_args()

    print("+ Parcae analysis\n")

    timestamps = parse_csv(args.csv)

    p = Parcae()
    result = p.analyze(timestamps)

    tz = result["timezone_offset_hours"]
    sleep_blocks = result["sleep_blocks"]

    print(f"~ inferred timezone: UTC{tz:+d}\n")

    offset = timedelta(hours=tz)

    local_blocks = []
    for b in sleep_blocks:
        start = datetime.fromisoformat(b["start"]) + offset
        end = datetime.fromisoformat(b["end"]) + offset
        local_blocks.append((start, end))

    by_day = defaultdict(list)

    for start, end in local_blocks:
        day = start.date()
        dur = (end - start).total_seconds()
        by_day[day].append((dur, start, end))

    main_sleeps = []
    for day, blocks in by_day.items():
        blocks.sort(reverse=True)
        _, start, end = blocks[0]
        main_sleeps.append((start, end))

    if not main_sleeps:
        print("! no sleep blocks detected")
        return

    sleep_starts = [minutes_since_midnight(s) for s, e in main_sleeps]
    sleep_ends = [minutes_since_midnight(e) for s, e in main_sleeps]
    durations = [int((e - s).total_seconds() / 60) for s, e in main_sleeps]

    mean_start = circular_mean_minutes(sleep_starts)
    mean_end = circular_mean_minutes(sleep_ends)
    durations.sort()
    med_dur = durations[len(durations) // 2]

    print("+ typical schedule:")
    print(
        f"\t- sleep: {format_hm(mean_start)} -> {format_hm(mean_end)}  (≈ {med_dur // 60}h {med_dur % 60:02d}m)"
    )
    print(f"\t- awake: {format_hm(mean_end)} -> {format_hm(mean_start)}\n")

    print(f"~ based on {len(main_sleeps)} days of data")
    print(f"~ bin size: {p.bin_minutes} minutes")


if __name__ == "__main__":
    main()
