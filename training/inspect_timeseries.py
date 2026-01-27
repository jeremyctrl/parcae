import random

import matplotlib.pyplot as plt
import numpy as np
from config import BINNED_DATASET, RANDOM_SEED

rng = random.Random(RANDOM_SEED)


def main():
    print("[*] loading dataset")

    data = np.load(BINNED_DATASET, allow_pickle=True)

    X = data["X"]
    user_ids = data["user_ids"]
    start_times = data["start_times"]
    bin_minutes = int(data["bin_minutes"])

    bins_per_day = (24 * 60) // bin_minutes

    print(f"[*] loaded {len(X)} users")
    print(f"[=] bin size: {bin_minutes} minutes ({bins_per_day} bins/day)")

    indices = rng.sample(range(len(X)), 5)

    for idx in indices:
        x = X[idx]
        uid = user_ids[idx]
        start = start_times[idx]

        print(f"[=] user {uid}")
        print(f"\tstart: {start}")
        print(f"\tlength: {len(x)} bins (~{len(x) / bins_per_day:.1f}) days")

        plt.figure(figsize=(12, 3))
        max_bins = min(len(x), bins_per_day * 7)

        plt.plot(x[:max_bins], drawstyle="steps-post")
        plt.ylim(-0.1, 1.1)
        plt.title(f"user {uid} | first ~7 days ({bin_minutes}-min bins)")
        plt.xlabel("time bins")
        plt.ylabel("activity (0/1)")
        plt.tight_layout()
        plt.show()

        n_days = len(x) // bins_per_day
        x_days = x[: n_days * bins_per_day].reshape(n_days, bins_per_day)

        plt.figure(figsize=(10, 6))
        plt.imshow(
            x_days,
            aspect="auto",
            interpolation="nearest",
            cmap="gray_r",
        )
        plt.colorbar(label="Activity")
        plt.title(f"user {uid} | days * time-of-day")
        plt.xlabel("time of day (bins)")
        plt.ylabel("day index")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
