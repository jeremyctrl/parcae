import argparse
import csv
import math

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


def angle_to_minutes(sin_v, cos_v):
    ang = math.atan2(sin_v, cos_v)
    if ang < 0:
        ang += 2 * math.pi
    return int(round(ang * 1440 / (2 * math.pi)))


def main():
    parser = argparse.ArgumentParser(prog="parcae")
    parser.add_argument("csv", help="CSV file with a 'timestamp' column")
    parser.add_argument("-v", "--version", action="version", version="%(prog)s 0.1.1")
    args = parser.parse_args()

    print("+ Parcae analysis\n")

    timestamps = parse_csv(args.csv)

    p = Parcae()
    result = p.analyze(timestamps)

    tz = result["timezone_offset_hours"]
    days = result["days"]

    print(f"~ inferred timezone: UTC{tz:+d}\n")

    sleep_phase = result["sleep_phase"]
    sleep_stats = result["sleep_stats"]

    mean_start = angle_to_minutes(sleep_phase[0], sleep_phase[1])
    mean_end = angle_to_minutes(sleep_phase[2], sleep_phase[3])
    
    std_dur = int(round(sleep_stats[1] * 1440))
    med_dur = int(round(sleep_stats[2] * 1440))

    print("+ typical schedule:")
    print(
        f"\t- sleep: {format_hm(mean_start)} -> {format_hm(mean_end)}  (≈ {med_dur // 60}h {med_dur % 60:02d}m)"
    )
    print(f"\t- awake: {format_hm(mean_end)} -> {format_hm(mean_start)}")
    print(f"\t- variability: ±{std_dur}m\n")

    print(f"~ based on {days} days of data")
    print(f"~ bin size: {p.bin_minutes} minutes")


if __name__ == "__main__":
    main()
