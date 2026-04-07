from __future__ import annotations
from pathlib import Path
import json
import pandas as pd

def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def append_metrics_row(metrics: dict, csv_path: str | Path) -> None:
    csv_path = Path(csv_path)
    ensure_dir(csv_path.parent)
    df = pd.DataFrame([metrics])
    if csv_path.exists():
        df.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df.to_csv(csv_path, index=False)

def save_json(obj: dict, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
