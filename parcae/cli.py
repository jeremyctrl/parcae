import argparse
import base64
import csv
import math

import numpy as np

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


def decode_fp(s):
    s = s.split(":", 2)[2]
    raw = base64.urlsafe_b64decode(s)
    q = np.frombuffer(raw, dtype=np.int16)
    return q.astype(np.float32) / 4096.0


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def sparkline(x):
    ticks = "▁▂▃▄▅▆▇█"
    x = np.asarray(x, dtype=float)

    lo = x.min()
    hi = x.max()

    if hi == lo:
        return ticks[0] * len(x)

    scaled = (x - lo) / (hi - lo) * (len(ticks) - 1)
    idx = np.round(scaled).astype(int)

    return "".join(ticks[i] for i in idx)


def hour_axis(n=24, marks=(0, 6, 12, 18, 24)):
    row = [" "] * n
    for m in marks:
        if m < n:
            row[m] = "|"
    return "".join(row)


def hour_labels(n=24, marks=(0, 6, 12, 18, 24)):
    row = [" "] * n
    for m in marks:
        s = f"{m:02d}"
        if m < n:
            for i, c in enumerate(s):
                if m + i < n:
                    row[m + i] = c
    return "".join(row)


def main():
    parser = argparse.ArgumentParser(prog="parcae")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_analyze = sub.add_parser("analyze")
    p_analyze.add_argument("csv", help="CSV file with a 'timestamp' column")

    p_cmp = sub.add_parser("compare")
    p_cmp.add_argument("fp1")
    p_cmp.add_argument("fp2")

    parser.add_argument("-v", "--version", action="version", version="%(prog)s 0.2.0")

    args = parser.parse_args()

    print("+ Parcae analysis\n")

    if args.cmd == "compare":
        v1 = decode_fp(args.fp1)
        v2 = decode_fp(args.fp2)
        sim = cosine(v1, v2)

        print("+ fingerprint comparison:")
        print(f"\tcosine similarity: {sim:.4f}")

        if sim > 0.95:
            print("\tmatch: very likely same user")
        elif sim > 0.90:
            print("\tmatch: probable")
        else:
            print("\tmatch: unlikely")

        return

    timestamps = parse_csv(args.csv)

    p = Parcae()
    result = p.analyze(timestamps)

    tz = result["timezone_offset_hours"]
    days = result["days"]

    print(f"~ inferred timezone: UTC{tz:+d}\n")

    sleep_phase = result["sleep_phase"]
    sleep_stats = result["sleep_stats"]

    profile_24h = result["profile_24h"]

    mean_start = angle_to_minutes(sleep_phase[0], sleep_phase[1])
    mean_end = angle_to_minutes(sleep_phase[2], sleep_phase[3])

    std_dur = int(round(sleep_stats[1] * 1440))
    med_dur = int(round(sleep_stats[2] * 1440))

    vec = np.concatenate(
        [profile_24h, result["sleep_phase"], result["sleep_stats"]]
    ).astype(np.float32)

    q = np.round(vec * 4096).astype(np.int16)
    fp = base64.urlsafe_b64encode(q.tobytes()).decode()

    print("+ typical schedule:")
    print(
        f"\t- sleep: {format_hm(mean_start)} -> {format_hm(mean_end)}  (≈ {med_dur // 60}h {med_dur % 60:02d}m)"
    )
    print(f"\t- awake: {format_hm(mean_end)} -> {format_hm(mean_start)}")
    print(f"\t- variability: ±{std_dur}m\n")

    print("+ activity profile (24h):")
    print(f"\t{sparkline(profile_24h)}")
    print(f"\t{hour_axis(len(profile_24h))}")
    print(f"\t{hour_labels(len(profile_24h))}\n")

    print("+ fingerprint:")
    print(f"\tparcae:v1:{fp}\n")

    print(f"~ based on {days} days of data")
    print(f"~ bin size: {p.bin_minutes} minutes")


if __name__ == "__main__":
    main()
