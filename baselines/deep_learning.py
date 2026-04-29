"""
Deep-learning baselines (paper Section 4.2):
  BERTBaseline      — bert-base-uncased fine-tuned, 2 manual features, identical
                      architecture to BERT LOLWTC notebook Section 7.
  CharCNNBaseline   — character-level CNN with 128 filters (paper: "128 filters each")
  LOLWTCBaseline    — Word2Vec (400-dim) + TextCNN from BERT LOLWTC notebook Section 8.
  BERTAugBaseline   — BERT fine-tuned on obfuscation-augmented training data.
  PowerPeelerBaseline — reimplementation of PowerPeeler (Li et al., 2024) following the
                        published pipeline: Shannon entropy detection, PowerShell
                        deobfuscation (base64 decode + alias resolution), then
                        TF-IDF + RF classification. Paper Section 4.2 / Related Work.

Each class exposes:
  .fit(train_commands, train_labels, val_commands, val_labels)
  .predict(commands) -> np.ndarray
  .predict_proba(commands) -> np.ndarray[:, 1]
"""

import numpy as np
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim import AdamW, Adam
from transformers import BertTokenizer, BertModel
from gensim.models import Word2Vec
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from halo_bert.config import (
    BERT_MODEL_NAME, BERT_MAX_LENGTH,
    BERT_BASELINE_HIDDEN, BERT_BASELINE_EPOCHS, BERT_BASELINE_LR,
    CHAR_MAX_LENGTH, CHAR_VOCAB_SIZE, CHAR_FILTER_SIZES, CHAR_NUM_FILTERS_BASELINE,
    W2V_VECTOR_SIZE, W2V_WINDOW, CNN_FILTER_SIZES_LOLWTC, CNN_NUM_FILTERS_LOLWTC,
    CNN_MAX_LENGTH, LOLWTC_EPOCHS, LOLWTC_LR,
    BATCH_SIZE, RANDOM_SEED, DROPOUT, AUG_FRACTIONS,
)
from halo_bert.dataset import preprocess_command
from halo_bert.obfuscation import TECHNIQUE_MAP


torch.manual_seed(RANDOM_SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Manual 2-feature extractor (BERT LOLWTC notebook Section 4) ───────────────

def _extract_2_features(commands):
    feats = [[len(c), len(c.split())] for c in commands]
    return np.array(feats, dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# BERT Baseline  (BERT LOLWTC notebook Section 7)
# ═══════════════════════════════════════════════════════════════════════════════

class _BERTDataset(Dataset):
    def __init__(self, commands, features, labels, tokenizer, max_length=BERT_MAX_LENGTH):
        self.commands  = commands
        self.features  = features
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self): return len(self.commands)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            preprocess_command(self.commands[idx]),
            truncation=True, padding="max_length",
            max_length=self.max_length, return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "features":       torch.tensor(self.features[idx], dtype=torch.float),
            "label":          torch.tensor(self.labels[idx], dtype=torch.long),
        }


class _BERTLotLDetector(nn.Module):
    """
    BERT + 2 manual features → FC(128) → FC(64) → 2.
    Matches BERTLotLDetector in the BERT LOLWTC notebook.
    """
    def __init__(self, num_features: int = 2, hidden_dim: int = BERT_BASELINE_HIDDEN):
        super().__init__()
        self.bert = BertModel.from_pretrained(BERT_MODEL_NAME)
        bert_dim  = self.bert.config.hidden_size  # 768
        self.fc1  = nn.Linear(bert_dim + num_features, hidden_dim)
        self.fc2  = nn.Linear(hidden_dim, 64)
        self.fc3  = nn.Linear(64, 2)
        self.drop = nn.Dropout(DROPOUT)

    def forward(self, input_ids, attention_mask, features):
        cls = self.bert(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        x = torch.cat([cls, features], dim=1)
        x = self.drop(F.relu(self.fc1(x)))
        x = self.drop(F.relu(self.fc2(x)))
        return self.fc3(x)


class BERTBaseline:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
        self.model     = _BERTLotLDetector().to(DEVICE)
        self.scaler    = StandardScaler()

    def fit(self, train_cmds, train_labels, val_cmds=None, val_labels=None):
        feats_tr = self.scaler.fit_transform(_extract_2_features(train_cmds))
        ds_tr = _BERTDataset(train_cmds, feats_tr, train_labels, self.tokenizer)
        dl_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True)

        optimizer = AdamW(self.model.parameters(), lr=BERT_BASELINE_LR)
        ce = nn.CrossEntropyLoss()
        best_f1, best_state = 0.0, None

        for epoch in range(BERT_BASELINE_EPOCHS):
            self.model.train()
            for batch in tqdm(dl_tr, desc=f"BERT baseline epoch {epoch+1}"):
                ids  = batch["input_ids"].to(DEVICE)
                mask = batch["attention_mask"].to(DEVICE)
                feat = batch["features"].to(DEVICE)
                lbl  = batch["label"].to(DEVICE)
                loss = ce(self.model(ids, mask, feat), lbl)
                optimizer.zero_grad(); loss.backward(); optimizer.step()

            if val_cmds is not None:
                from sklearn.metrics import f1_score
                preds = self.predict(val_cmds)
                f1 = f1_score(val_labels, preds, average="macro", zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                print(f"  BERT baseline val macro-F1 = {f1:.4f}")

        if best_state:
            self.model.load_state_dict(best_state)

    def _forward(self, commands):
        feats = self.scaler.transform(_extract_2_features(commands))
        ds = _BERTDataset(commands, feats, np.zeros(len(commands), int), self.tokenizer)
        dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
        self.model.eval()
        all_logits = []
        with torch.no_grad():
            for batch in dl:
                ids  = batch["input_ids"].to(DEVICE)
                mask = batch["attention_mask"].to(DEVICE)
                feat = batch["features"].to(DEVICE)
                all_logits.append(self.model(ids, mask, feat).cpu())
        return torch.cat(all_logits, dim=0)

    def predict(self, commands):
        return self._forward(commands).argmax(1).numpy()

    def predict_proba(self, commands):
        return torch.softmax(self._forward(commands), dim=1)[:, 1].numpy()


# ═══════════════════════════════════════════════════════════════════════════════
# CharCNN Standalone Baseline  (paper Section 4.2: "128 filters each")
# ═══════════════════════════════════════════════════════════════════════════════

class _CharCNNModel(nn.Module):
    """
    Character-level CNN: filter sizes (3,4,5), 128 filters each → 384 → 2.
    Paper Section 4.2 / Zhang et al. (2015).
    """
    def __init__(self, vocab=CHAR_VOCAB_SIZE, embed=32,
                 filters=CHAR_FILTER_SIZES, nf=CHAR_NUM_FILTERS_BASELINE):
        super().__init__()
        self.embed = nn.Embedding(vocab, embed, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv2d(1, nf, (k, embed)) for k in filters])
        self.fc1   = nn.Linear(len(filters) * nf, 256)
        self.fc2   = nn.Linear(256, 2)
        self.drop  = nn.Dropout(DROPOUT)
        self.out_dim = len(filters) * nf   # 384

    def forward(self, x):
        e = self.embed(x).unsqueeze(1)    # (B, 1, L, 32)
        pooled = []
        for conv in self.convs:
            h = F.relu(conv(e)).squeeze(3)
            h = F.max_pool1d(h, h.size(2)).squeeze(2)
            pooled.append(h)
        h = torch.cat(pooled, 1)
        return self.fc2(self.drop(F.relu(self.fc1(h))))


def _encode_chars_batch(commands, max_len=CHAR_MAX_LENGTH):
    arr = np.zeros((len(commands), max_len), dtype=np.int64)
    for i, cmd in enumerate(commands):
        for j, ch in enumerate(cmd[:max_len]):
            arr[i, j] = min(ord(ch), CHAR_VOCAB_SIZE - 1)
    return arr


class CharCNNBaseline:
    def __init__(self):
        self.model = _CharCNNModel().to(DEVICE)

    def fit(self, train_cmds, train_labels, val_cmds=None, val_labels=None, epochs=10, lr=1e-3):
        X = torch.tensor(_encode_chars_batch(train_cmds), dtype=torch.long)
        y = torch.tensor(train_labels, dtype=torch.long)
        dl = DataLoader(TensorDataset(X, y), batch_size=BATCH_SIZE, shuffle=True)
        opt = Adam(self.model.parameters(), lr=lr)
        ce  = nn.CrossEntropyLoss()
        best_f1, best_state = 0.0, None

        for epoch in range(epochs):
            self.model.train()
            for xb, yb in tqdm(dl, desc=f"CharCNN epoch {epoch+1}"):
                loss = ce(self.model(xb.to(DEVICE)), yb.to(DEVICE))
                opt.zero_grad(); loss.backward(); opt.step()

            if val_cmds is not None:
                from sklearn.metrics import f1_score
                preds = self.predict(val_cmds)
                f1 = f1_score(val_labels, preds, average="macro", zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                print(f"  CharCNN val macro-F1 = {f1:.4f}")
        if best_state:
            self.model.load_state_dict(best_state)

    def _forward(self, commands):
        X = torch.tensor(_encode_chars_batch(commands), dtype=torch.long)
        dl = DataLoader(TensorDataset(X), batch_size=BATCH_SIZE, shuffle=False)
        self.model.eval()
        all_logits = []
        with torch.no_grad():
            for (xb,) in dl:
                all_logits.append(self.model(xb.to(DEVICE)).cpu())
        return torch.cat(all_logits, 0)

    def predict(self, commands):
        return self._forward(commands).argmax(1).numpy()

    def predict_proba(self, commands):
        return torch.softmax(self._forward(commands), dim=1)[:, 1].numpy()


# ═══════════════════════════════════════════════════════════════════════════════
# LOLWTC Baseline  (BERT LOLWTC notebook Section 8)
# ═══════════════════════════════════════════════════════════════════════════════

class _LOLWTC_TextCNN(nn.Module):
    """
    Matches LOLWTC_TextCNN in the BERT LOLWTC notebook exactly:
      filter_sizes = [2,3,4,5,6], num_filters = 100
      4 hidden FC layers: 256, 128, 64, 32
      manual features: 2 (cmd_length, word_count, scaled)
    """
    def __init__(self, embed_dim=W2V_VECTOR_SIZE,
                 filter_sizes=CNN_FILTER_SIZES_LOLWTC, nf=CNN_NUM_FILTERS_LOLWTC,
                 num_features=2):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(1, nf, (k, embed_dim)) for k in filter_sizes
        ])
        fc_in = len(filter_sizes) * nf + num_features
        self.fc1 = nn.Linear(fc_in, 256); self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128);   self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64);    self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, 32);     self.bn4 = nn.BatchNorm1d(32)
        self.fc5 = nn.Linear(32, 2)
        self.drop = nn.Dropout(0.5)

    def forward(self, x, feats):
        # x: (B, max_len, embed_dim)
        x = x.unsqueeze(1)    # (B, 1, max_len, embed_dim)
        pooled = []
        for conv in self.convs:
            h = F.relu(conv(x))
            h = F.max_pool2d(h, (h.size(2), 1)).squeeze(3).squeeze(2)
            pooled.append(h)
        h = torch.cat([*pooled, feats], dim=1)
        h = self.drop(F.relu(self.bn1(self.fc1(h))))
        h = self.drop(F.relu(self.bn2(self.fc2(h))))
        h = self.drop(F.relu(self.bn3(self.fc3(h))))
        h = self.drop(F.relu(self.bn4(self.fc4(h))))
        return self.fc5(h)


class LOLWTCBaseline:
    def __init__(self):
        self.w2v    = None
        self.model  = None
        self.scaler = StandardScaler()

    def _to_matrix(self, commands):
        mats = []
        for cmd in commands:
            words = preprocess_command(cmd).split()
            m = np.zeros((CNN_MAX_LENGTH, W2V_VECTOR_SIZE), dtype=np.float32)
            for i, w in enumerate(words[:CNN_MAX_LENGTH]):
                if w in self.w2v.wv:
                    m[i] = self.w2v.wv[w]
            mats.append(m)
        return np.array(mats, dtype=np.float32)

    def fit(self, train_cmds, train_labels, val_cmds=None, val_labels=None):
        # Train Word2Vec on training commands
        tokenised = [preprocess_command(c).split() for c in train_cmds]
        self.w2v  = Word2Vec(tokenised, vector_size=W2V_VECTOR_SIZE,
                             window=W2V_WINDOW, min_count=1, workers=4, seed=RANDOM_SEED)

        X_mat  = self._to_matrix(train_cmds)
        feats_raw = _extract_2_features(train_cmds)
        feats  = self.scaler.fit_transform(feats_raw)

        self.model = _LOLWTC_TextCNN().to(DEVICE)
        opt = Adam(self.model.parameters(), lr=LOLWTC_LR)
        ce  = nn.CrossEntropyLoss()

        X_t = torch.tensor(X_mat, dtype=torch.float)
        f_t = torch.tensor(feats, dtype=torch.float)
        y_t = torch.tensor(train_labels, dtype=torch.long)
        dl  = DataLoader(TensorDataset(X_t, f_t, y_t), batch_size=32, shuffle=True)

        best_f1, best_state = 0.0, None
        for epoch in range(LOLWTC_EPOCHS):
            self.model.train()
            for xb, fb, yb in tqdm(dl, desc=f"LOLWTC epoch {epoch+1}"):
                loss = ce(self.model(xb.to(DEVICE), fb.to(DEVICE)), yb.to(DEVICE))
                opt.zero_grad(); loss.backward(); opt.step()

            if val_cmds is not None and (epoch + 1) % 5 == 0:
                from sklearn.metrics import f1_score
                preds = self.predict(val_cmds)
                f1 = f1_score(val_labels, preds, average="macro", zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                print(f"  LOLWTC epoch {epoch+1} val macro-F1 = {f1:.4f}")
        if best_state:
            self.model.load_state_dict(best_state)

    def _forward(self, commands):
        X    = self._to_matrix(commands)
        feats = self.scaler.transform(_extract_2_features(commands))
        X_t  = torch.tensor(X, dtype=torch.float)
        f_t  = torch.tensor(feats, dtype=torch.float)
        dl   = DataLoader(TensorDataset(X_t, f_t), batch_size=32, shuffle=False)
        self.model.eval()
        all_logits = []
        with torch.no_grad():
            for xb, fb in dl:
                all_logits.append(self.model(xb.to(DEVICE), fb.to(DEVICE)).cpu())
        return torch.cat(all_logits, 0)

    def predict(self, commands):
        return self._forward(commands).argmax(1).numpy()

    def predict_proba(self, commands):
        return torch.softmax(self._forward(commands), dim=1)[:, 1].numpy()


# ═══════════════════════════════════════════════════════════════════════════════
# BERT + Augmentation Baseline  (paper Section 4.2)
# ═══════════════════════════════════════════════════════════════════════════════

def _augment_training(commands, labels, fractions=AUG_FRACTIONS, seed=RANDOM_SEED):
    """
    Apply each obfuscation technique to the specified fraction of malicious commands.
    Returns augmented (commands, labels) appended to original.
    """
    import random as rnd
    rnd.seed(seed)
    mal_mask  = np.array(labels) == 1
    mal_cmds  = np.array(commands)[mal_mask]
    aug_cmds, aug_lbls = [], []
    for tech, frac in fractions.items():
        n = max(1, int(len(mal_cmds) * frac))
        subset = rnd.choices(mal_cmds.tolist(), k=n)
        for cmd in subset:
            try:
                aug_cmds.append(TECHNIQUE_MAP[tech](cmd))
            except Exception:
                aug_cmds.append(cmd)
            aug_lbls.append(1)
    all_cmds   = list(commands) + aug_cmds
    all_labels = list(labels) + aug_lbls
    return all_cmds, all_labels


class BERTAugBaseline(BERTBaseline):
    """BERT fine-tuned on obfuscation-augmented training data (paper Section 4.2)."""

    def fit(self, train_cmds, train_labels, val_cmds=None, val_labels=None):
        aug_cmds, aug_labels = _augment_training(train_cmds, train_labels)
        print(f"  BERTAug: {len(train_cmds)} → {len(aug_cmds)} commands after augmentation")
        super().fit(aug_cmds, aug_labels, val_cmds, val_labels)


# ═══════════════════════════════════════════════════════════════════════════════
# PowerPeeler Baseline  (paper Section 4.2 / Related Work / Li et al., 2024)
# ═══════════════════════════════════════════════════════════════════════════════
#
# PowerPeeler (Li et al., 2024) handles PowerShell obfuscation via:
#   1. Shannon entropy to flag encoded/obfuscated content
#   2. Iterative deobfuscation: base64 decode, alias resolution, backtick removal,
#      string concatenation flattening (AST-level parsing)
#   3. Classification on the deobfuscated text
#
# Paper note: "PowerPeeler addresses PowerShell obfuscation through deobfuscation
# techniques but lacks general command semantics" — it is specialised to PowerShell;
# non-PowerShell commands fall through to direct TF-IDF + RF classification.
# ─────────────────────────────────────────────────────────────────────────────

import base64
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from halo_bert.config import (
    TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE, RF_N_ESTIMATORS, RF_MAX_DEPTH,
)


# PowerShell alias table (same set used in obfuscation.py)
_PS_ALIAS_REVERSE = {v: k for k, v in {
    "Invoke-WebRequest": "iwr",
    "Get-Content":       "gc",
    "Set-Content":       "sc",
    "Get-ChildItem":     "gci",
    "Remove-Item":       "ri",
    "Write-Output":      "echo",
    "ForEach-Object":    "%",
    "Where-Object":      "?",
    "Select-Object":     "select",
    "Invoke-Expression": "iex",
}.items()}

_PS_REGEX = re.compile(r"powershell|pwsh", re.IGNORECASE)
_B64_REGEX = re.compile(r"[A-Za-z0-9+/]{20,}={0,2}")
_ENC_FLAG  = re.compile(r"-e(?:nc(?:odedcommand)?)?(?:\s+|$)", re.IGNORECASE)


def _shannon_entropy(text: str) -> float:
    """Shannon entropy of character distribution — high values flag encoded content."""
    if not text:
        return 0.0
    freq = {}
    for ch in text:
        freq[ch] = freq.get(ch, 0) + 1
    n = len(text)
    return -sum((c / n) * math.log2(c / n) for c in freq.values())


def _try_b64_decode(token: str) -> str:
    """Attempt UTF-16LE then UTF-8 base64 decode; return original on failure."""
    for enc in ("utf-16-le", "utf-8"):
        try:
            return base64.b64decode(token + "==").decode(enc)
        except Exception:
            pass
    return token


def _resolve_aliases(text: str) -> str:
    """Expand short PowerShell aliases to full cmdlet names."""
    for alias, full in _PS_ALIAS_REVERSE.items():
        text = re.sub(rf"\b{re.escape(alias)}\b", full, text, flags=re.IGNORECASE)
    return text


def _remove_backticks(text: str) -> str:
    return text.replace("`", "")


def _flatten_concat(text: str) -> str:
    """Remove simple string-concatenation operators: ("foo"+"bar") → foobar."""
    return re.sub(r'["\']?\s*\+\s*["\']?', "", text)


def deobfuscate_powershell(command: str) -> str:
    """
    Iterative PowerShell deobfuscation pipeline (paper Related Work Section 2.2).
    Steps mirror the published PowerPeeler pipeline:
      1. Strip -EncodedCommand flag and decode base64 payload
      2. Remove backtick escapes
      3. Flatten string concatenation
      4. Resolve short aliases to full cmdlet names
    Non-PowerShell commands are returned unchanged.
    """
    if not _PS_REGEX.search(command):
        return command

    text = command

    # Step 1 — base64 decode any -EncodedCommand payload
    if _ENC_FLAG.search(text):
        for token in _B64_REGEX.findall(text):
            if _shannon_entropy(token) > 3.5:   # high entropy ≈ encoded content
                decoded = _try_b64_decode(token)
                if decoded != token:
                    text = text.replace(token, decoded)

    # Step 2 — also decode standalone high-entropy base64 tokens (without -enc flag)
    for token in _B64_REGEX.findall(text):
        if _shannon_entropy(token) > 4.5:
            decoded = _try_b64_decode(token)
            if decoded != token:
                text = text.replace(token, decoded)

    # Step 3-5 — surface normalisation
    text = _remove_backticks(text)
    text = _flatten_concat(text)
    text = _resolve_aliases(text)
    return text


def _powerpeel_features(commands):
    """
    Per-command feature vector used by PowerPeeler classifier:
      - Shannon entropy of raw command
      - Shannon entropy of deobfuscated command
      - Is PowerShell (binary)
      - Had -EncodedCommand flag (binary)
      - Had high-entropy base64 token (binary)
    """
    rows = []
    for cmd in commands:
        deob = deobfuscate_powershell(cmd)
        rows.append([
            _shannon_entropy(cmd),
            _shannon_entropy(deob),
            float(bool(_PS_REGEX.search(cmd))),
            float(bool(_ENC_FLAG.search(cmd))),
            float(any(_shannon_entropy(t) > 3.5 for t in _B64_REGEX.findall(cmd))),
        ])
    return np.array(rows, dtype=np.float32)


class PowerPeelerBaseline:
    """
    Reimplementation of PowerPeeler (Li et al., 2024) following the published
    pipeline (paper Section 4.2): entropy-based obfuscation detection,
    iterative PowerShell deobfuscation, then TF-IDF (1-3 gram) + RF classification
    on the deobfuscated text combined with entropy features.

    Reference: Li et al. (2024), "PowerPeeler: A Precise and General Dynamic
    Deobfuscation Approach for PowerShell Scripts."
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES,
            ngram_range=TFIDF_NGRAM_RANGE,
            token_pattern=r"\b\w+\b",
        )
        self.clf = RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS, max_depth=RF_MAX_DEPTH,
            random_state=RANDOM_SEED, n_jobs=-1,
        )

    def _transform(self, commands, fit: bool = False):
        deobfuscated = [deobfuscate_powershell(c) for c in commands]
        if fit:
            tfidf = self.vectorizer.fit_transform(deobfuscated).toarray()
        else:
            tfidf = self.vectorizer.transform(deobfuscated).toarray()
        entropy_feats = _powerpeel_features(commands)   # (n, 5)
        return np.hstack([tfidf, entropy_feats])

    def fit(self, train_cmds, train_labels, val_cmds=None, val_labels=None):
        print("  PowerPeeler: deobfuscating + fitting TF-IDF + RF...")
        X = self._transform(train_cmds, fit=True)
        self.clf.fit(X, train_labels)
        if val_cmds is not None:
            from sklearn.metrics import f1_score
            preds = self.predict(val_cmds)
            f1 = f1_score(val_labels, preds, average="macro", zero_division=0)
            print(f"  PowerPeeler val macro-F1 = {f1:.4f}")

    def predict(self, commands):
        return self.clf.predict(self._transform(commands))

    def predict_proba(self, commands):
        return self.clf.predict_proba(self._transform(commands))[:, 1]
