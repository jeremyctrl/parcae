from datetime import timedelta

import numpy as np
import pandas as pd
from config import BIN_MINUTES, BINNED_DATASET, DATA_CSV, MIN_DAYS


def bin_user(timestamps):
    start = timestamps.min().floor("D")
    end = timestamps.max().ceil("D")

    bin_delta = timedelta(minutes=BIN_MINUTES)
    n_bins = int((end - start) / bin_delta) + 1

    if n_bins <= 0:
        return None, None

    bins = np.zeros(n_bins, dtype=np.uint8)
    indices = ((timestamps - start) / bin_delta).astype(int)

    bins[np.unique(indices)] = 1

    return start, bins


def main():
    print("[*] loading csv")

    df = pd.read_csv(DATA_CSV)

    print("[*] parsing timestamps")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])

    print("[*] grouping")

    groups = df.groupby("user_id")

    X = []
    user_ids = []
    start_times = []

    kept = 0
    skipped = 0

    print("[*] binning")

    for i, (uid, group) in enumerate(groups):
        ts = group["timestamp"].sort_values()

        start, bins = bin_user(ts)

        if bins is None:
            skipped += 1
            continue

        bins_per_day = 24 * 60 // BIN_MINUTES
        min_bins = bins_per_day * MIN_DAYS

        if len(bins) < min_bins:
            skipped += 1
            continue

        X.append(bins)
        user_ids.append(uid)
        start_times.append(start)

        kept += 1
        if kept % 50 == 0:
            print(f"\tkept {kept} users...")

    print(f"[=] kept {kept} users")
    print(f"[=] skipped {skipped} users")

    start_times = [t.tz_convert(None) for t in start_times]

    np.savez_compressed(
        BINNED_DATASET,
        X=np.array(X, dtype=object),
        user_ids=np.array(user_ids, dtype=object),
        start_times=np.array(start_times, dtype="datetime64[ns]"),
        bin_minutes=BIN_MINUTES,
    )

    print(f"[~] saved to {BINNED_DATASET}")

    lengths = np.array([len(x) for x in X])
    bins_per_day = (24 * 60) // BIN_MINUTES

    print(f"\tmin: {lengths.min()}")
    print(f"\tmax: {lengths.max()}")
    print(f"\tmean: {lengths.mean():.1f}")
    print(f"\tbins per day: {bins_per_day}")
    print(f"\tapprox days (mean): {lengths.mean() / bins_per_day:.1f}")


if __name__ == "__main__":
    main()
