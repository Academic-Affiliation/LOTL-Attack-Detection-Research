"""
Training pipeline for HALO-BERT (paper Section 4.2).

Steps:
  1. Pretrain AnomalyAutoencoder on benign training data only (then freeze).
  2. Set reconstruction threshold τ at 95th percentile on benign validation data.
  3. Train full HALO-BERT with combined loss L = L_CE + λ*L_recon.
     Mixed batches: 60% single-command, 40% multi-stage sequence.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

from halo_bert.config import (
    LEARNING_RATE, NUM_EPOCHS, EARLY_STOPPING_PATIENCE,
    CLASS_WEIGHT_BENIGN, CLASS_WEIGHT_MALICIOUS,
    ANOMALY_LAMBDA, ALPHA,
)
from halo_bert.model import HALOBert, InterCommandAttention


# ── Helpers ───────────────────────────────────────────────────────────────────

def _class_weights(device: torch.device) -> torch.Tensor:
    return torch.tensor([CLASS_WEIGHT_BENIGN, CLASS_WEIGHT_MALICIOUS], device=device)


# ── Autoencoder pretraining ───────────────────────────────────────────────────

def pretrain_autoencoder(
    model:       HALOBert,
    train_loader: DataLoader,
    device:      torch.device,
    epochs:      int = 5,
    lr:          float = 1e-3,
) -> None:
    """
    Train model.anomaly_ae on benign samples only; keep BERT frozen to avoid
    leakage into the classification pathway.
    """
    model.bert.eval()
    model.anomaly_ae.train()
    for p in model.bert.parameters():
        p.requires_grad = False

    optimizer = torch.optim.Adam(model.anomaly_ae.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0.0
        n = 0
        for batch in tqdm(train_loader, desc=f"AE pretrain epoch {epoch+1}/{epochs}"):
            labels = batch["label"].to(device)
            benign_mask = labels == 0
            if benign_mask.sum() == 0:
                continue

            input_ids = batch["input_ids"][benign_mask].to(device)
            attn_mask = batch["attention_mask"][benign_mask].to(device)

            with torch.no_grad():
                bert_out = model.bert(input_ids=input_ids, attention_mask=attn_mask)
                h_bert   = bert_out.pooler_output

            _, _, recon_loss = model.anomaly_ae(h_bert)
            loss = recon_loss.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * benign_mask.sum().item()
            n += benign_mask.sum().item()

        if n > 0:
            print(f"  AE epoch {epoch+1}: avg recon loss = {total_loss/n:.5f}")

    model.anomaly_ae.freeze()
    # Unfreeze BERT for main training
    for p in model.bert.parameters():
        p.requires_grad = True


def compute_anomaly_threshold(
    model:      HALOBert,
    val_loader: DataLoader,
    device:     torch.device,
    percentile: float = 95.0,
) -> float:
    """
    Set τ as the 95th percentile of reconstruction errors on benign validation
    samples (paper Section 3.5).
    """
    model.eval()
    errors = []
    with torch.no_grad():
        for batch in val_loader:
            labels = batch["label"].to(device)
            benign_mask = labels == 0
            if benign_mask.sum() == 0:
                continue
            input_ids = batch["input_ids"][benign_mask].to(device)
            attn_mask = batch["attention_mask"][benign_mask].to(device)

            bert_out = model.bert(input_ids=input_ids, attention_mask=attn_mask)
            _, _, recon_loss = model.anomaly_ae(bert_out.pooler_output)
            errors.extend(recon_loss.cpu().numpy().tolist())

    tau = float(np.percentile(errors, percentile))
    print(f"Anomaly threshold τ set to {tau:.4f} (p{percentile:.0f} of benign val errors)")
    return tau


# ── Main HALO-BERT training loop ─────────────────────────────────────────────

def train_halo_bert(
    model:          HALOBert,
    temporal_attn:  InterCommandAttention,
    train_loader:   DataLoader,    # single-command batches
    seq_loader:     DataLoader,    # multi-stage sequence batches (may be None)
    val_loader:     DataLoader,
    device:         torch.device,
    save_path:      str = "halo_bert_best.pt",
) -> list:
    """
    Train HALO-BERT end-to-end.

    Mixed batching (paper Section 4.2):
      60% of steps use single-command batches  (L_single, weight α=0.6)
      40% of steps use multi-stage sequences   (L_sequence, weight 1-α=0.4)
    Combined objective:
      L_total = L_CE + λ * L_recon   (λ = ANOMALY_LAMBDA = 0.1)
    """
    model.to(device)
    temporal_attn.to(device)

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad] + list(temporal_attn.parameters()),
        lr=LEARNING_RATE,
    )
    ce_loss_fn = nn.CrossEntropyLoss(weight=_class_weights(device))

    best_val_f1   = 0.0
    patience_cnt  = 0
    history       = []

    seq_iter = iter(seq_loader) if seq_loader is not None else None

    for epoch in range(NUM_EPOCHS):
        model.train()
        temporal_attn.train()
        epoch_loss = 0.0
        steps = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
            # ── Single-command step ────────────────────────────────────────
            input_ids  = batch["input_ids"].to(device)
            attn_mask  = batch["attention_mask"].to(device)
            char_ids   = batch["char_ids"].to(device)
            labels     = batch["label"].to(device)

            logits, recon_loss = model(input_ids, attn_mask, char_ids)
            l_single = ce_loss_fn(logits, labels) + ANOMALY_LAMBDA * recon_loss.mean()
            loss = ALPHA * l_single

            # ── Sequence step (40% of iterations) ─────────────────────────
            if seq_iter is not None:
                try:
                    seq_batch = next(seq_iter)
                except StopIteration:
                    seq_iter = iter(seq_loader)
                    seq_batch = next(seq_iter)

                seq_ids      = seq_batch["input_ids"].to(device)    # (B, T, L)
                seq_mask     = seq_batch["attention_mask"].to(device)
                seq_len      = seq_batch["seq_len"].to(device)
                seq_labels   = seq_batch["label"].to(device)
                B, T, L      = seq_ids.shape

                # Encode each command in sequence via BERT + mean-pool
                with torch.set_grad_enabled(True):
                    flat_ids  = seq_ids.view(B * T, L)
                    flat_mask = seq_mask.view(B * T, L)
                    flat_out  = model.bert(input_ids=flat_ids, attention_mask=flat_mask)
                    flat_emb  = flat_out.last_hidden_state   # (B*T, L, 768)
                    flat_mask_f = flat_mask.unsqueeze(2).float()
                    flat_mean   = (flat_emb * flat_mask_f).sum(1) / flat_mask_f.sum(1).clamp(min=1)
                    seq_emb   = flat_mean.view(B, T, -1)     # (B, T, 768)

                h_temp_seq  = temporal_attn(seq_emb, seq_len)  # (B, 768)
                # Use first command in sequence for token-level inputs
                first_ids  = seq_ids[:, 0, :]
                first_mask = seq_mask[:, 0, :]
                # CharCNN on first command chars (sequences don't have char_ids)
                # Use zero chars as placeholder — sequence pathway uses h_temporal
                char_placeholder = torch.zeros(B, 512, dtype=torch.long, device=device)
                logits_seq, recon_seq = model(
                    first_ids, first_mask, char_placeholder, h_temporal=h_temp_seq
                )
                l_seq = ce_loss_fn(logits_seq, seq_labels) + ANOMALY_LAMBDA * recon_seq.mean()
                loss  = loss + (1.0 - ALPHA) * l_seq

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(temporal_attn.parameters()), 1.0
            )
            optimizer.step()

            epoch_loss += loss.item()
            steps += 1

        avg_loss = epoch_loss / max(steps, 1)
        val_metrics = evaluate_epoch(model, temporal_attn, val_loader, device)
        history.append({"epoch": epoch + 1, "loss": avg_loss, **val_metrics})

        print(
            f"Epoch {epoch+1}: loss={avg_loss:.4f}  "
            f"val_acc={val_metrics['accuracy']:.4f}  "
            f"val_f1={val_metrics['f1']:.4f}"
        )

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            torch.save(
                {"model": model.state_dict(), "temporal_attn": temporal_attn.state_dict()},
                save_path,
            )
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping at epoch {epoch+1} (patience={EARLY_STOPPING_PATIENCE})")
                break

    return history


# ── Per-epoch validation ──────────────────────────────────────────────────────

def evaluate_epoch(
    model:         HALOBert,
    temporal_attn: InterCommandAttention,
    loader:        DataLoader,
    device:        torch.device,
) -> dict:
    from sklearn.metrics import accuracy_score, f1_score
    model.eval()
    temporal_attn.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids  = batch["input_ids"].to(device)
            attn_mask  = batch["attention_mask"].to(device)
            char_ids   = batch["char_ids"].to(device)
            labels     = batch["label"]

            logits, _ = model(input_ids, attn_mask, char_ids)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())

    return {
        "accuracy": accuracy_score(all_labels, all_preds),
        "f1":       f1_score(all_labels, all_preds, average="macro", zero_division=0),
    }
