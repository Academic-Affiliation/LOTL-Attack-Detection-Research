"""
HALO-BERT: Hierarchical Analysis for Living-Off-the-Land with BERT.

Architecture (paper Section 3):
  ┌─────────────────────────────────────────────────────┐
  │  Input: WordPiece tokens  +  raw character sequence  │
  └────────────┬────────────────────────┬────────────────┘
               │                        │
        ┌──────▼──────┐          ┌──────▼──────┐
        │  BERT (12L) │          │  CharCNN    │
        │  768-dim    │          │  192-dim    │
        └──┬──┬──┬────┘          └──────┬──────┘
           │  │  │                       │
    ┌──────▼┐ │ ┌▼──────────┐            │
    │Temporal│ │ │Anomaly AE │            │
    │ Attn   │ │ │ 768→128   │            │
    │ 768-dim│ │ └─────┬─────┘            │
    └────────┘ │       │                  │
               │   z_anomaly(128)         │
               │                         │
         h_BERT(768)                      │
    ─────────────────────────────────────┘
    concat → [768 + 192 + 768 + 128] = 1856
    → LayerNorm → Linear(1856,768) → ReLU → Dropout(0.3)
    → Linear(768,384) → ReLU → Linear(384,2) → softmax
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

from halo_bert.config import (
    BERT_MODEL_NAME,
    CHAR_VOCAB_SIZE, CHAR_EMBED_DIM, CHAR_FILTER_SIZES, CHAR_NUM_FILTERS,
    TEMPORAL_NUM_HEADS,
    AE_HIDDEN_DIM, AE_LATENT_DIM,
    CONCAT_DIM, FUSION_DIM, CLASSIFIER_DIM,
    DROPOUT,
)


# ── Character-level CNN (paper Section 3.4) ───────────────────────────────────

class CharCNN(nn.Module):
    """
    Three parallel Conv1D layers (filter sizes 3,4,5; 64 filters each).
    Output dimension: len(filter_sizes) * num_filters = 3 * 64 = 192.
    """

    def __init__(
        self,
        vocab_size: int  = CHAR_VOCAB_SIZE,
        embed_dim:  int  = CHAR_EMBED_DIM,
        filter_sizes     = CHAR_FILTER_SIZES,
        num_filters: int = CHAR_NUM_FILTERS,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        # Conv2d over (batch, 1, seq_len, embed_dim) — same idiom as LOLWTC notebook
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (k, embed_dim)) for k in filter_sizes
        ])
        self.out_dim = len(filter_sizes) * num_filters  # 192

    def forward(self, char_ids: torch.Tensor) -> torch.Tensor:
        # char_ids: (B, 512)
        x = self.embedding(char_ids)           # (B, 512, 32)
        x = x.unsqueeze(1)                    # (B, 1, 512, 32)
        pooled = []
        for conv in self.convs:
            h = F.relu(conv(x))               # (B, num_filters, 512-k+1, 1)
            h = h.squeeze(3)                  # (B, num_filters, 512-k+1)
            h = F.max_pool1d(h, h.size(2))    # (B, num_filters, 1)
            pooled.append(h.squeeze(2))       # (B, num_filters)
        return torch.cat(pooled, dim=1)       # (B, 192)


# ── Inter-command attention (paper Section 3.3) ───────────────────────────────

class InterCommandAttention(nn.Module):
    """
    Multi-head self-attention over a sequence of per-command BERT embeddings.
    Input:  (B, T, 768)  — T mean-pooled BERT command representations
    Output: (B, 768)     — mean-pooled after attention
    Paper: H=4 heads, d_k=192 (768/4).
    """

    def __init__(self, hidden_dim: int = 768, num_heads: int = TEMPORAL_NUM_HEADS):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, seq_emb: torch.Tensor, seq_len: torch.Tensor) -> torch.Tensor:
        # seq_emb : (B, T, 768)   seq_len : (B,)
        T = seq_emb.size(1)
        key_padding_mask = torch.arange(T, device=seq_emb.device).unsqueeze(0) >= seq_len.unsqueeze(1)
        out, _ = self.attn(seq_emb, seq_emb, seq_emb, key_padding_mask=key_padding_mask)
        out = self.norm(out + seq_emb)
        # mean-pool over valid positions only
        mask = (~key_padding_mask).float().unsqueeze(2)  # (B, T, 1)
        return (out * mask).sum(1) / mask.sum(1).clamp(min=1)   # (B, 768)


# ── Anomaly autoencoder (paper Section 3.5) ──────────────────────────────────

class AnomalyAutoencoder(nn.Module):
    """
    Pretrained on benign training data only; frozen during HALO-BERT training.
    Encoder: 768 → 256 → 128 (z_anomaly)
    Decoder: 128 → 256 → 768 (reconstruction)
    """

    def __init__(
        self,
        input_dim:  int = 768,
        hidden_dim: int = AE_HIDDEN_DIM,
        latent_dim: int = AE_LATENT_DIM,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, h: torch.Tensor):
        z       = self.encoder(h)
        h_recon = self.decoder(z)
        recon_loss = F.mse_loss(h_recon, h, reduction="none").mean(dim=-1)  # (B,)
        return z, h_recon, recon_loss

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False


# ── Full HALO-BERT model ──────────────────────────────────────────────────────

class HALOBert(nn.Module):
    """
    End-to-end HALO-BERT for single-command classification.

    During training, the multi-stage sequence pathway is handled externally by
    InterCommandAttention.forward() called from the training loop (see train.py).
    This module processes one command at a time; h_temporal is mean-pooled BERT
    for single commands (paper Section 3.3: "No additional attention is added for
    single commands").
    """

    def __init__(self, bert_model_name: str = BERT_MODEL_NAME):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.char_cnn = CharCNN()
        self.anomaly_ae = AnomalyAutoencoder()

        # Fusion + classifier (paper Section 3.6)
        self.fusion = nn.Sequential(
            nn.LayerNorm(CONCAT_DIM),
            nn.Linear(CONCAT_DIM, FUSION_DIM),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
        )
        self.classifier = nn.Sequential(
            nn.Linear(FUSION_DIM, CLASSIFIER_DIM),
            nn.ReLU(),
            nn.Linear(CLASSIFIER_DIM, 2),
        )

    def forward(
        self,
        input_ids:      torch.Tensor,   # (B, max_length)
        attention_mask: torch.Tensor,   # (B, max_length)
        char_ids:       torch.Tensor,   # (B, 512)
        h_temporal:     torch.Tensor = None,  # (B, 768) — pre-computed or None
    ):
        # ── BERT encoding ──────────────────────────────────────────────────
        bert_out  = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        h_bert    = bert_out.pooler_output          # (B, 768)  — semantic (CLS)
        token_emb = bert_out.last_hidden_state      # (B, seq_len, 768)

        # ── Temporal representation ────────────────────────────────────────
        # For single commands: mean-pool over valid tokens (paper Section 3.3)
        if h_temporal is None:
            mask_f = attention_mask.unsqueeze(2).float()   # (B, seq_len, 1)
            h_temporal = (token_emb * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)

        # ── CharCNN ────────────────────────────────────────────────────────
        h_char = self.char_cnn(char_ids)    # (B, 192)

        # ── Anomaly autoencoder ────────────────────────────────────────────
        z_anomaly, _h_recon, recon_loss = self.anomaly_ae(h_bert)  # z: (B,128)

        # ── Feature fusion ─────────────────────────────────────────────────
        h_concat = torch.cat([h_bert, h_char, h_temporal, z_anomaly], dim=1)  # (B,1856)
        h_fused  = self.fusion(h_concat)   # (B, 768)
        logits   = self.classifier(h_fused)  # (B, 2)

        return logits, recon_loss
