"""
Evaluation utilities producing all metrics reported in the paper:
  - Overall accuracy, precision, recall, macro-F1
  - Per-class precision & recall (Table 4)
  - TPR at low-FPR operating points (Table 5 / Table 6)
  - Wilson 95% confidence intervals (Appendix D)
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve,
)
from scipy.stats import norm

from halo_bert.model import HALOBert, InterCommandAttention
from halo_bert.config import ANOMALY_THRESHOLD


# ── Core prediction function ──────────────────────────────────────────────────

def predict(
    model:         HALOBert,
    temporal_attn: InterCommandAttention,
    loader:        DataLoader,
    device:        torch.device,
    tau:           float = ANOMALY_THRESHOLD,
) -> tuple:
    """
    Returns (y_true, y_pred, y_prob_malicious, anomaly_flags).

    anomaly_flags: bool array; True when reconstruction error > τ
    (anomaly flags trigger triage but do NOT override the classification output).
    """
    model.eval()
    temporal_attn.eval()

    all_labels, all_preds, all_probs, all_anom = [], [], [], []

    with torch.no_grad():
        for batch in loader:
            input_ids  = batch["input_ids"].to(device)
            attn_mask  = batch["attention_mask"].to(device)
            char_ids   = batch["char_ids"].to(device)
            labels     = batch["label"].cpu().numpy()

            logits, recon_loss = model(input_ids, attn_mask, char_ids)
            probs  = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds  = (probs > 0.5).astype(int)
            anomaly = (recon_loss.cpu().numpy() > tau)

            all_labels.extend(labels.tolist())
            all_preds.extend(preds.tolist())
            all_probs.extend(probs.tolist())
            all_anom.extend(anomaly.tolist())

    return (
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs),
        np.array(all_anom),
    )


# ── Metric helpers ────────────────────────────────────────────────────────────

def overall_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall":    recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1":        f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Returns precision and recall for each class (paper Table 4)."""
    prec = precision_score(y_true, y_pred, average=None, zero_division=0)
    rec  = recall_score(y_true, y_pred, average=None, zero_division=0)
    return {
        "benign_precision":    prec[0],
        "benign_recall":       rec[0],
        "malicious_precision": prec[1],
        "malicious_recall":    rec[1],
    }


def tpr_at_fpr(y_true: np.ndarray, y_prob: np.ndarray, fpr_target: float) -> float:
    """
    TPR at a given FPR operating point (paper Tables 5/6).
    Uses sklearn's roc_curve to find the threshold closest to fpr_target.
    """
    fprs, tprs, _ = roc_curve(y_true, y_prob)
    idx = np.searchsorted(fprs, fpr_target, side="right") - 1
    idx = max(0, min(idx, len(tprs) - 1))
    return float(tprs[idx])


def wilson_ci(n_correct: int, n: int, confidence: float = 0.95) -> tuple:
    """95% Wilson score confidence interval (paper Appendix D)."""
    if n == 0:
        return (0.0, 0.0)
    z   = norm.ppf(1 - (1 - confidence) / 2)
    p   = n_correct / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


# ── Full evaluation on one dataset split ─────────────────────────────────────

def evaluate(
    model:         HALOBert,
    temporal_attn: InterCommandAttention,
    loader:        DataLoader,
    device:        torch.device,
    split_name:    str = "test",
    tau:           float = ANOMALY_THRESHOLD,
    fpr_targets:   tuple = (0.01, 0.005, 0.001, 0.0001, 0.00001),
) -> dict:
    """
    Full evaluation producing all numbers reported in the paper.
    """
    y_true, y_pred, y_prob, anom_flags = predict(model, temporal_attn, loader, device, tau)

    n = len(y_true)
    n_correct = int((y_true == y_pred).sum())

    ci_lo, ci_hi = wilson_ci(n_correct, n)
    overall = overall_metrics(y_true, y_pred)
    per_cls = per_class_metrics(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    tpr_results = {}
    for fpr_t in fpr_targets:
        label = f"tpr@fpr={fpr_t}"
        tpr_results[label] = tpr_at_fpr(y_true, y_prob, fpr_t)

    pattern_cov  = float(anom_flags.mean())
    avg_conf     = float(y_prob.mean())

    results = {
        "split":             split_name,
        "n":                 n,
        **overall,
        **per_cls,
        "wilson_ci":         (ci_lo, ci_hi),
        "confusion_matrix":  cm.tolist(),
        "avg_confidence":    avg_conf,
        "anomaly_flag_rate": pattern_cov,
        **tpr_results,
    }

    _print_results(results)
    return results


def _print_results(r: dict):
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"Evaluation: {r['split']}  (n={r['n']})")
    print(sep)
    print(f"  Accuracy  : {r['accuracy']:.4f}  {tuple(f'{v:.3f}' for v in r['wilson_ci'])}")
    print(f"  Precision : {r['precision']:.4f}")
    print(f"  Recall    : {r['recall']:.4f}")
    print(f"  F1 (macro): {r['f1']:.4f}")
    print(f"  Benign    — prec {r['benign_precision']:.4f}  rec {r['benign_recall']:.4f}")
    print(f"  Malicious — prec {r['malicious_precision']:.4f}  rec {r['malicious_recall']:.4f}")
    print(f"  Anomaly flag rate: {r['anomaly_flag_rate']:.4f}")
    for k, v in r.items():
        if k.startswith("tpr@"):
            print(f"  {k}: {v:.4f}")
