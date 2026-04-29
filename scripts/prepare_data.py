"""
Data preparation pipeline (paper Section 4.1).

Steps:
  1. Load balanced_combined_lotl_dataset.csv  (7049 rows, Command/Label)
  2. SHA-256 exact deduplication
  3. Levenshtein near-dedup at >90% string similarity
  4. Stratified 70/15/15 split → train / val / test
  5. Save splits to data/splits/

Run:
  python scripts/prepare_data.py
"""

import hashlib
import json
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from Levenshtein import ratio as lev_ratio

from halo_bert.config import (
    TRAIN_CSV, RANDOM_SEED,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO,
    LEVENSHTEIN_THRESHOLD,
)
from halo_bert.dataset import load_csv

OUT_DIR = "data/splits"


def sha256_dedup(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact-duplicate commands using SHA-256 hashing."""
    hashes = df["Command"].apply(lambda c: hashlib.sha256(str(c).encode()).hexdigest())
    before = len(df)
    df = df[~hashes.duplicated(keep="first")].reset_index(drop=True)
    print(f"  SHA-256 dedup: removed {before - len(df)} exact duplicates")
    return df


def levenshtein_dedup(df: pd.DataFrame, threshold: float = LEVENSHTEIN_THRESHOLD) -> pd.DataFrame:
    """
    Remove near-duplicate commands where Levenshtein similarity > threshold.
    O(n²) — feasible at n~7000.
    Paper reports 89 near-duplicates removed at >90% threshold.
    """
    commands = df["Command"].tolist()
    keep = np.ones(len(commands), dtype=bool)
    for i in range(len(commands)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(commands)):
            if not keep[j]:
                continue
            if lev_ratio(commands[i], commands[j]) > threshold:
                keep[j] = False

    before = len(df)
    df = df[keep].reset_index(drop=True)
    print(f"  Levenshtein dedup (>{threshold*100:.0f}%): removed {before - len(df)} near-duplicates")
    return df


def split_dataset(df: pd.DataFrame):
    """Stratified 70/15/15 split (paper Section 4.1)."""
    X = df["Command"].values
    y = df["Label"].values

    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=(VAL_RATIO + TEST_RATIO),
        stratify=y, random_state=RANDOM_SEED,
    )
    val_frac = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=(1.0 - val_frac),
        stratify=y_tmp, random_state=RANDOM_SEED,
    )
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading training corpus...")
    df = load_csv(TRAIN_CSV)
    print(f"  Loaded {len(df)} rows  (benign={int((df.Label==0).sum())}, malicious={int((df.Label==1).sum())})")

    print("Deduplicating...")
    df = sha256_dedup(df)
    df = levenshtein_dedup(df)
    print(f"  After dedup: {len(df)} rows")

    print("Splitting 70/15/15...")
    (X_tr, y_tr), (X_val, y_val), (X_te, y_te) = split_dataset(df)
    print(f"  train={len(X_tr)}  val={len(X_val)}  test={len(X_te)}")

    for name, X, y in [("train", X_tr, y_tr), ("val", X_val, y_val), ("test", X_te, y_te)]:
        out = pd.DataFrame({"Command": X, "Label": y})
        path = os.path.join(OUT_DIR, f"{name}.csv")
        out.to_csv(path, index=False)
        print(f"  Saved {path}  (benign={int((y==0).sum())}, malicious={int((y==1).sum())})")

    # Save metadata
    meta = {
        "train_n": int(len(X_tr)),
        "val_n":   int(len(X_val)),
        "test_n":  int(len(X_te)),
        "random_seed": RANDOM_SEED,
    }
    with open(os.path.join(OUT_DIR, "split_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("Done. Split metadata saved.")


if __name__ == "__main__":
    main()
