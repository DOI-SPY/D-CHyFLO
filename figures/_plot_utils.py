"""Small shared helpers for public result plots."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def table(name: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / "outputs" / name)


def finish(name: str) -> Path:
    target = ROOT / "outputs" / name
    plt.tight_layout()
    plt.savefig(target, dpi=180)
    plt.close()
    return target
