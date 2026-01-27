from pathlib import Path

DATA_DIR = Path("data")
MODEL_DIR = Path("models")

for d in (MODEL_DIR, DATA_DIR):
    d.mkdir(exist_ok=True, parents=True)

BIN_MINUTES = 15
MIN_DAYS = 7

DATA_CSV = DATA_DIR / "data.csv"
BINNED_DATASET = DATA_DIR / f"users_{BIN_MINUTES}.npz"

RANDOM_SEED = 42