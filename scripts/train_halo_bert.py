"""
Full HALO-BERT training script (paper Section 4.2).

Usage:
  python scripts/train_halo_bert.py

Expects data/splits/{train,val}.csv produced by prepare_data.py.
Saves checkpoints to checkpoints/halo_bert_best.pt.

Training proceeds in two phases:
  Phase 1 — AnomalyAutoencoder pretrained on benign training samples (5 epochs).
  Phase 2 — Full HALO-BERT trained end-to-end with frozen AE (10 epochs, patience=3).
"""

import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from halo_bert.config import (
    BERT_MODEL_NAME, BERT_MAX_LENGTH, BATCH_SIZE, RANDOM_SEED, TEMPORAL_NUM_HEADS,
)
from halo_bert.dataset import LOTLDataset
from halo_bert.model import HALOBert, InterCommandAttention
from halo_bert.train import pretrain_autoencoder, compute_anomaly_threshold, train_halo_bert
from halo_bert.evaluate import evaluate

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

SPLITS_DIR = "data/splits"
CKPT_DIR   = "checkpoints"
RESULTS_DIR = "results"


def load_split(name: str):
    df = pd.read_csv(os.path.join(SPLITS_DIR, f"{name}.csv"))
    return df["Command"].values, df["Label"].values.astype(int)


def main():
    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading splits...")
    X_train, y_train = load_split("train")
    X_val,   y_val   = load_split("val")
    X_test,  y_test  = load_split("test")

    # Load adversarial test sets
    volt_df   = pd.read_csv("complete_volttyphoon_dataset.csv")
    obfusc_df = pd.read_csv("enhanced_lotl_obfuscated_dataset.csv")
    X_volt,   y_volt   = volt_df["Command"].values, volt_df["Label"].values.astype(int)
    X_obfusc, y_obfusc = obfusc_df["Command"].values, obfusc_df["Label"].values.astype(int)

    print(f"  train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    print(f"  volt_typhoon={len(X_volt)}, obfuscated={len(X_obfusc)}")

    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)

    def make_loader(X, y, shuffle=False):
        ds = LOTLDataset(X, y, tokenizer, BERT_MAX_LENGTH)
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=0)

    dl_train  = make_loader(X_train, y_train, shuffle=True)
    dl_val    = make_loader(X_val,   y_val)
    dl_test   = make_loader(X_test,  y_test)
    dl_volt   = make_loader(X_volt,  y_volt)
    dl_obfusc = make_loader(X_obfusc, y_obfusc)

    model          = HALOBert(BERT_MODEL_NAME).to(device)
    temporal_attn  = InterCommandAttention(768, TEMPORAL_NUM_HEADS).to(device)

    # ── Phase 1: Pretrain autoencoder on benign training data ──────────────────
    print("\n=== Phase 1: Autoencoder pretraining ===")
    pretrain_autoencoder(model, dl_train, device, epochs=5)

    # Set anomaly threshold τ at 95th percentile of benign validation errors
    tau = compute_anomaly_threshold(model, dl_val, device, percentile=95.0)

    # ── Phase 2: Train HALO-BERT (AE frozen) ──────────────────────────────────
    print("\n=== Phase 2: HALO-BERT training ===")
    save_path = os.path.join(CKPT_DIR, "halo_bert_best.pt")
    history = train_halo_bert(
        model, temporal_attn,
        train_loader=dl_train, seq_loader=None,   # no sequence dataset by default
        val_loader=dl_val, device=device,
        save_path=save_path,
    )

    # ── Load best checkpoint and evaluate on all splits ───────────────────────
    print("\n=== Loading best checkpoint ===")
    ckpt = torch.load(save_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    temporal_attn.load_state_dict(ckpt["temporal_attn"])

    all_results = {}
    for split_name, dl in [
        ("validation",   dl_val),
        ("test",         dl_test),
        ("volt_typhoon", dl_volt),
        ("obfuscated",   dl_obfusc),
    ]:
        print(f"\n{'─'*60}")
        res = evaluate(model, temporal_attn, dl, device, split_name=split_name, tau=tau)
        all_results[split_name] = {
            k: v.tolist() if hasattr(v, "tolist") else v
            for k, v in res.items()
        }

    out_path = os.path.join(RESULTS_DIR, "halo_bert_results.json")
    with open(out_path, "w") as f:
        json.dump({"history": history, "results": all_results, "tau": tau}, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
