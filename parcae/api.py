import sysconfig
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np


def _logsumexp(a):
    m = np.max(a)
    return m + np.log(np.sum(np.exp(a - m)))


def _forward_log(obs, log_trans, log_emit, log_init):
    T = len(obs)
    alpha = np.zeros((T, 2))

    alpha[0] = log_init + log_emit[:, obs[0]]

    for t in range(1, T):
        for j in range(2):
            alpha[t, j] = log_emit[j, obs[t]] + _logsumexp(
                alpha[t - 1] + log_trans[:, j]
            )

    return _logsumexp(alpha[T - 1])


def _viterbi(obs, log_trans, log_emit, log_init):
    T = len(obs)
    dp = np.zeros((T, 2))
    back = np.zeros((T, 2), dtype=np.uint8)

    dp[0] = log_init + log_emit[:, obs[0]]

    for t in range(1, T):
        for j in range(2):
            scores = dp[t - 1] + log_trans[:, j]
            k = np.argmax(scores)
            dp[t, j] = scores[k] + log_emit[j, obs[t]]
            back[t, j] = k

    last = np.argmax(dp[T - 1])
    best = dp[T - 1, last]

    path = np.zeros(T, dtype=np.int8)
    path[T - 1] = last

    for t in range(T - 2, -1, -1):
        path[t] = back[t + 1, path[t + 1]]

    return path, best


class Parcae:
    def __init__(self, model_path=None, bin_minutes=15):
        if model_path is None:
            data_path = Path(sysconfig.get_paths()["data"]) / "models"
            model_path = data_path / "hmm.npz"

        data = np.load(model_path)

        self.startprob = data["startprob"]
        self.transmat = data["transmat"]
        self.emissionprob = data["emissionprob"]

        self.log_startprob = np.log(self.startprob)
        self.log_transmat = np.log(self.transmat)
        self.log_emissionprob = np.log(self.emissionprob)

        self.bin_minutes = int(data.get("bin_minutes", bin_minutes))

        self.sleep_state = int(np.argmin(self.emissionprob[:, 1]))
        self.awake_state = 1 - self.sleep_state

    def _parse_timestamps(self, timestamps):
        out = []
        for t in timestamps:
            if isinstance(t, datetime):
                out.append(t)
            else:
                out.append(datetime.fromisoformat(str(t)))
        return sorted(out)

    def _bin(self, timestamps):
        start = timestamps[0].replace(hour=0, minute=0, second=0, microsecond=0)
        end = timestamps[-1].replace(
            hour=0, minute=0, second=0, microsecond=0
        ) + timedelta(days=1)

        bin_delta = timedelta(minutes=self.bin_minutes)
        n_bins = int((end - start) / bin_delta)

        bins = np.zeros(n_bins, dtype=np.uint8)

        for t in timestamps:
            idx = int((t - start) / bin_delta)
            if 0 <= idx < n_bins:
                bins[idx] = 1

        return start, bins

    def analyze(self, timestamps, tz_range=range(-12, 13)):
        ts = self._parse_timestamps(timestamps)

        span = ts[-1] - ts[0]
        if span < timedelta(days=2):  # arbitrary number that seems fine
            raise ValueError("not enough time span to analyze (need at least ~2 days)")

        start_time, bins = self._bin(ts)

        bins_per_day = (24 * 60) // self.bin_minutes

        if len(bins) < 2 * bins_per_day:  # arbitrary number that seems fine
            raise ValueError("not enough data after binning (need at least ~2 days)")

        best_phi = 0
        best_score = -np.inf

        for phi in tz_range:
            shift_bins = int(phi * bins_per_day / 24)
            bins_phi = np.roll(bins, shift_bins)

            score = _forward_log(
                bins_phi,
                self.log_transmat,
                self.log_emissionprob,
                self.log_startprob,
            )

            if score > best_score:
                best_score = score
                best_phi = phi

        shift_bins = int(best_phi * bins_per_day / 24)
        best_bins = np.roll(bins, shift_bins)

        states, _ = _viterbi(
            best_bins, self.log_transmat, self.log_emissionprob, self.log_startprob
        )

        sleep_blocks = []
        awake_blocks = []

        current_state = states[0]
        block_start = 0

        for i in range(1, len(states)):
            if states[i] != current_state:
                (
                    sleep_blocks if current_state == self.sleep_state else awake_blocks
                ).append((block_start, i))

                block_start = i
                current_state = states[i]

        if current_state == self.sleep_state:
            sleep_blocks.append((block_start, len(states)))
        else:
            awake_blocks.append((block_start, len(states)))

        def blocks_to_time(blocks):
            out = []
            for a, b in blocks:
                t0 = start_time + timedelta(minutes=self.bin_minutes * a)
                t1 = start_time + timedelta(minutes=self.bin_minutes * b)
                out.append({"start": t0.isoformat(), "end": t1.isoformat()})
            return out

        return {
            "timezone_offset_hours": int(best_phi),
            "sleep_blocks": blocks_to_time(sleep_blocks),
            "awake_blocks": blocks_to_time(awake_blocks),
        }
