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

        return {"timezone_offset_hours": int(best_phi)}
