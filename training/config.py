from pathlib import Path

DATA_DIR = Path("data")
MODEL_DIR = Path("models")

for d in (MODEL_DIR, DATA_DIR):
    d.mkdir(exist_ok=True, parents=True)
