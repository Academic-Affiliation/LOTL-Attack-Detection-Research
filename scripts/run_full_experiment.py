"""
Run all baselines + HALO-BERT and print/save the comparison tables from the paper.

Usage:
  python scripts/run_full_experiment.py [--skip-deep] [--skip-halo]

  --skip-deep   skip BERT/CharCNN/LOLWTC/BERTAug (saves time if GPU unavailable)
  --skip-halo   skip HALO-BERT (run baselines only)

Requires: data/splits/ produced by prepare_data.py
          checkpoints/halo_bert_best.pt produced by train_halo_bert.py (unless --skip-halo)

Outputs:
  results/table3_overall.csv      → paper Table 3 (accuracy across all splits)
  results/table4_perclass.csv     → paper Table 4 (per-class on Volt Typhoon)
  results/table5_lowfpr_val.csv   → paper Table 5 (TPR at low FPR, validation)
  results/table6_lowfpr_adv.csv   → paper Table 6 (TPR at low FPR, adversarial)
"""

import argparse
import json
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
)

from halo_bert.config import (
    BERT_MODEL_NAME, BERT_MAX_LENGTH, BATCH_SIZE, RANDOM_SEED, TEMPORAL_NUM_HEADS,
    ANOMALY_THRESHOLD,
)
from halo_bert.dataset import LOTLDataset
from halo_bert.model import HALOBert, InterCommandAttention
from halo_bert.evaluate import (
    predict as halo_predict, overall_metrics, per_class_metrics, tpr_at_fpr,
)
from baselines.classical import CLASSICAL_BASELINES
from baselines.deep_learning import (
    BERTBaseline, CharCNNBaseline, LOLWTCBaseline, BERTAugBaseline, PowerPeelerBaseline,
)

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

SPLITS_DIR  = "data/splits"
CKPT_DIR    = "checkpoints"
RESULTS_DIR = "results"

FPR_TARGETS_VAL = (0.01, 0.005, 0.001)
FPR_TARGETS_ADV = (0.01, 0.005, 0.001, 0.0001, 0.00001)


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_csv(path):
    df = pd.read_csv(path)
    if "command" in df.columns:
        df = df.rename(columns={"command": "Command", "label": "Label"})
    return df["Command"].values, df["Label"].values.astype(int)


def load_all_splits():
    X_tr, y_tr = _load_csv(os.path.join(SPLITS_DIR, "train.csv"))
    X_va, y_va = _load_csv(os.path.join(SPLITS_DIR, "val.csv"))
    X_te, y_te = _load_csv(os.path.join(SPLITS_DIR, "test.csv"))
    X_vo, y_vo = _load_csv("complete_volttyphoon_dataset.csv")
    X_ob, y_ob = _load_csv("enhanced_lotl_obfuscated_dataset.csv")
    return {
        "train": (X_tr, y_tr),
        "val":   (X_va, y_va),
        "test":  (X_te, y_te),
        "volt":  (X_vo, y_vo),
        "obf":   (X_ob, y_ob),
    }


# ── Classical baseline evaluation ─────────────────────────────────────────────

def run_classical(splits):
    X_tr, y_tr = splits["train"]
    records = []
    for name, builder in CLASSICAL_BASELINES.items():
        print(f"\nTraining {name}...")
        clf = builder()
        clf.fit(X_tr, y_tr)

        row = {"model": name}
        for split, (X, y) in [
            ("val", splits["val"]), ("volt", splits["volt"]), ("obf", splits["obf"]),
        ]:
            preds = clf.predict(X)
            probs = clf.predict_proba(X)
            if hasattr(probs, "__len__") and probs.ndim == 2:
                probs = probs[:, 1]
            acc = accuracy_score(y, preds)
            f1  = f1_score(y, preds, average="macro", zero_division=0)
            row[f"{split}_acc"] = acc
            row[f"{split}_f1"]  = f1
            # Per-class on volt
            if split == "volt":
                prec = precision_score(y, preds, average=None, zero_division=0)
                rec  = recall_score(y, preds, average=None, zero_division=0)
                row["volt_ben_prec"] = prec[0]; row["volt_ben_rec"] = rec[0]
                row["volt_mal_prec"] = prec[1]; row["volt_mal_rec"] = rec[1]
            # TPR at low FPR on val
            if split == "val":
                targets = FPR_TARGETS_VAL
            else:
                targets = FPR_TARGETS_ADV
            for fpr_t in targets:
                row[f"{split}_tpr@{fpr_t}"] = tpr_at_fpr(y, probs, fpr_t)
        records.append(row)
    return records


# ── Deep-learning baseline evaluation ────────────────────────────────────────

def run_deep(splits):
    X_tr, y_tr = splits["train"]
    X_va, y_va = splits["val"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = []

    dl_baselines = [
        ("BERT",         BERTBaseline()),
        ("CharCNN",      CharCNNBaseline()),
        ("LOLWTC",       LOLWTCBaseline()),
        ("BERT+Aug",     BERTAugBaseline()),
        ("PowerPeeler",  PowerPeelerBaseline()),
    ]

    for name, clf in dl_baselines:
        print(f"\nTraining {name}...")
        clf.fit(X_tr, y_tr, val_cmds=X_va, val_labels=y_va)

        row = {"model": name}
        for split, (X, y) in [
            ("val", splits["val"]), ("volt", splits["volt"]), ("obf", splits["obf"]),
        ]:
            preds = clf.predict(X)
            probs = clf.predict_proba(X)
            row[f"{split}_acc"] = accuracy_score(y, preds)
            row[f"{split}_f1"]  = f1_score(y, preds, average="macro", zero_division=0)
            if split == "volt":
                prec = precision_score(y, preds, average=None, zero_division=0)
                rec  = recall_score(y, preds, average=None, zero_division=0)
                row["volt_ben_prec"] = prec[0]; row["volt_ben_rec"] = rec[0]
                row["volt_mal_prec"] = prec[1]; row["volt_mal_rec"] = rec[1]
            targets = FPR_TARGETS_VAL if split == "val" else FPR_TARGETS_ADV
            for fpr_t in targets:
                row[f"{split}_tpr@{fpr_t}"] = tpr_at_fpr(y, probs, fpr_t)
        records.append(row)
    return records


# ── HALO-BERT evaluation ──────────────────────────────────────────────────────

def run_halo_bert(splits):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)

    def make_loader(X, y):
        ds = LOTLDataset(X, y, tokenizer, BERT_MAX_LENGTH)
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    ckpt = torch.load(os.path.join(CKPT_DIR, "halo_bert_best.pt"), map_location=device)

    # Load tau from training run if available
    tau = ANOMALY_THRESHOLD
    tau_path = os.path.join(RESULTS_DIR, "halo_bert_results.json")
    if os.path.exists(tau_path):
        with open(tau_path) as f:
            tau = json.load(f).get("tau", ANOMALY_THRESHOLD)

    model = HALOBert(BERT_MODEL_NAME).to(device)
    temporal_attn = InterCommandAttention(768, TEMPORAL_NUM_HEADS).to(device)
    model.load_state_dict(ckpt["model"])
    temporal_attn.load_state_dict(ckpt["temporal_attn"])

    row = {"model": "HALO-BERT"}
    for split, (X, y) in [
        ("val",  splits["val"]),
        ("volt", splits["volt"]),
        ("obf",  splits["obf"]),
    ]:
        dl = make_loader(X, y)
        y_true, y_pred, y_prob, _ = halo_predict(model, temporal_attn, dl, device, tau)
        row[f"{split}_acc"] = accuracy_score(y_true, y_pred)
        row[f"{split}_f1"]  = f1_score(y_true, y_pred, average="macro", zero_division=0)
        if split == "volt":
            prec = precision_score(y_true, y_pred, average=None, zero_division=0)
            rec  = recall_score(y_true, y_pred, average=None, zero_division=0)
            row["volt_ben_prec"] = prec[0]; row["volt_ben_rec"] = rec[0]
            row["volt_mal_prec"] = prec[1]; row["volt_mal_rec"] = rec[1]
        targets = FPR_TARGETS_VAL if split == "val" else FPR_TARGETS_ADV
        for fpr_t in targets:
            row[f"{split}_tpr@{fpr_t}"] = tpr_at_fpr(y_true, y_prob, fpr_t)
    return [row]


# ── Table printing ────────────────────────────────────────────────────────────

def _fmt(v):
    if isinstance(v, float):
        return f"{v*100:.2f}%"
    return str(v)


def print_table3(records):
    print("\n" + "=" * 70)
    print("TABLE 3 — Overall Accuracy across splits")
    print("=" * 70)
    print(f"{'Model':<22} {'Val':>8} {'Volt':>8} {'Obf':>8}")
    print("-" * 50)
    for r in records:
        print(f"{r['model']:<22} {_fmt(r['val_acc']):>8} {_fmt(r['volt_acc']):>8} {_fmt(r['obf_acc']):>8}")


def print_table4(records):
    print("\n" + "=" * 70)
    print("TABLE 4 — Per-class metrics on Volt Typhoon")
    print("=" * 70)
    hdr = f"{'Model':<22} {'Ben-P':>7} {'Ben-R':>7} {'Mal-P':>7} {'Mal-R':>7}"
    print(hdr)
    print("-" * 52)
    for r in records:
        if "volt_ben_prec" not in r:
            continue
        print(
            f"{r['model']:<22} {_fmt(r['volt_ben_prec']):>7} {_fmt(r['volt_ben_rec']):>7}"
            f" {_fmt(r['volt_mal_prec']):>7} {_fmt(r['volt_mal_rec']):>7}"
        )


def print_table5(records):
    print("\n" + "=" * 70)
    print("TABLE 5 — TPR at low-FPR (validation)")
    print("=" * 70)
    fprs = FPR_TARGETS_VAL
    hdr_cols = "  ".join(f"TPR@{f*100:.1f}%" for f in fprs)
    print(f"{'Model':<22}  {hdr_cols}")
    print("-" * 60)
    for r in records:
        vals = "  ".join(f"{_fmt(r.get(f'val_tpr@{fpr_t}', 0.0)):>10}" for fpr_t in fprs)
        print(f"{r['model']:<22}  {vals}")


def print_table6(records):
    print("\n" + "=" * 70)
    print("TABLE 6 — TPR at low-FPR (Volt Typhoon + Obfuscated)")
    print("=" * 70)
    fprs = FPR_TARGETS_ADV
    for split in ["volt", "obf"]:
        label = "Volt Typhoon" if split == "volt" else "Obfuscated"
        print(f"\n  {label}")
        hdr_cols = "  ".join(f"TPR@{f*100:.2f}%" for f in fprs)
        print(f"  {'Model':<20}  {hdr_cols}")
        print("  " + "-" * 68)
        for r in records:
            vals = "  ".join(f"{_fmt(r.get(f'{split}_tpr@{fpr_t}', 0.0)):>12}" for fpr_t in fprs)
            print(f"  {r['model']:<20}  {vals}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-deep", action="store_true")
    parser.add_argument("--skip-halo", action="store_true")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    splits = load_all_splits()

    records = []

    print("─" * 60)
    print("Running classical baselines...")
    records.extend(run_classical(splits))

    if not args.skip_deep:
        print("─" * 60)
        print("Running deep-learning baselines...")
        records.extend(run_deep(splits))

    if not args.skip_halo:
        print("─" * 60)
        print("Loading HALO-BERT checkpoint...")
        records.extend(run_halo_bert(splits))

    # Save combined CSV
    df_out = pd.DataFrame(records)
    df_out.to_csv(os.path.join(RESULTS_DIR, "all_results.csv"), index=False)

    # Print tables
    print_table3(records)
    print_table4(records)
    print_table5(records)
    print_table6(records)

    print(f"\nAll results saved to {RESULTS_DIR}/all_results.csv")


if __name__ == "__main__":
    main()
