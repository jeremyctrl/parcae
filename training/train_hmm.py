import numpy as np
from config import BINNED_DATASET, HMM_MODEL_OUT
from hmmlearn.hmm import CategoricalHMM


def main():
    print("[*] loading dataset")

    data = np.load(BINNED_DATASET, allow_pickle=True)

    X = data["X"]
    bin_minutes = int(data["bin_minutes"])

    bins_per_day = (24 * 60) // bin_minutes

    print(f"[*] loaded {len(X)} users")
    print(f"[*] bin size: {bin_minutes} minutes ({bins_per_day} bins/day)")

    print("[*] concatenating sequences")

    X_cat = []
    lengths = []

    for x in X:
        x = np.asarray(x, dtype=np.int64)
        x = x.reshape(-1, 1)
        X_cat.append(x)
        lengths.append(len(x))

    X_cat = np.vstack(X_cat)

    print(f"[*] total bins: {len(X_cat)}")

    print("[*] building model")

    model = CategoricalHMM(
        n_components=2,
        n_iter=50,
        tol=1e-3,
        verbose=True,
        init_params="ste",
        params="ste",
    )

    print("[*] training")

    model.fit(X_cat, lengths)

    print("[*] training done")

    print(f"[=] startprob: {model.startprob_}")
    print(f"[=] transmat: {model.transmat_}")
    print(f"[=] emissionprob: {model.emissionprob_}")

    np.savez(
        HMM_MODEL_OUT,
        startprob=model.startprob_,
        transmat=model.transmat_,
        emissionprob=model.emissionprob_,
        bin_minutes=bin_minutes,
    )

    print(f"[~] model saved to {HMM_MODEL_OUT}")


if __name__ == "__main__":
    main()
