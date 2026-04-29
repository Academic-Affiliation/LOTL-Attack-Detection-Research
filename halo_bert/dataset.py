"""
Dataset classes for HALO-BERT training and evaluation.

Column conventions (all three CSVs use the same schema):
  Command : str   raw Windows command string
  Label   : int   0 = benign, 1 = malicious
"""

import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

from halo_bert.config import (
    BERT_MAX_LENGTH,
    CHAR_MAX_LENGTH, CHAR_VOCAB_SIZE,
    MAX_SEQ_LEN,
)


# ── Text preprocessing (mirrors BERT LOLWTC notebook) ─────────────────────────

def preprocess_command(command: str) -> str:
    """Lowercase, normalise backslashes, collapse whitespace."""
    command = str(command).lower()
    command = re.sub(r"\\", "/", command)
    command = " ".join(command.split())
    return command


def encode_chars(command: str, max_length: int = CHAR_MAX_LENGTH) -> np.ndarray:
    """Convert raw command to fixed-length ASCII array [0,255] (paper Section 3.2)."""
    arr = np.zeros(max_length, dtype=np.int64)
    for i, ch in enumerate(command[:max_length]):
        arr[i] = min(ord(ch), CHAR_VOCAB_SIZE - 1)
    return arr


# ── Single-command dataset ────────────────────────────────────────────────────

class LOTLDataset(Dataset):
    """
    Tokenises commands for BERT and encodes character sequences.
    Used for HALO-BERT single-command batches and all baselines.
    """

    def __init__(
        self,
        commands: np.ndarray,
        labels: np.ndarray,
        tokenizer: BertTokenizer,
        max_length: int = BERT_MAX_LENGTH,
    ):
        self.commands  = commands
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.commands)

    def __getitem__(self, idx):
        raw = str(self.commands[idx])
        processed = preprocess_command(raw)
        label = int(self.labels[idx])

        enc = self.tokenizer(
            processed,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        char_ids = encode_chars(raw)

        return {
            "input_ids":      enc["input_ids"].squeeze(0),       # (max_length,)
            "attention_mask": enc["attention_mask"].squeeze(0),  # (max_length,)
            "char_ids":       torch.tensor(char_ids, dtype=torch.long),  # (512,)
            "label":          torch.tensor(label, dtype=torch.long),
        }


# ── Multi-stage sequence dataset ──────────────────────────────────────────────

class SequenceDataset(Dataset):
    """
    Groups commands into sequences of length T ≤ MAX_SEQ_LEN for training the
    inter-command attention pathway (paper Section 3.3).

    sequences : list of lists of command strings, each inner list is one sequence.
    labels    : sequence-level label (1 if the sequence is a malicious progression).
    """

    def __init__(
        self,
        sequences: list,
        labels: np.ndarray,
        tokenizer: BertTokenizer,
        max_length: int = BERT_MAX_LENGTH,
        max_seq_len: int = MAX_SEQ_LEN,
    ):
        self.sequences   = sequences
        self.labels      = labels
        self.tokenizer   = tokenizer
        self.max_length  = max_length
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq   = self.sequences[idx][:self.max_seq_len]
        label = int(self.labels[idx])
        T     = len(seq)

        all_input_ids = []
        all_attn_mask = []
        for cmd in seq:
            enc = self.tokenizer(
                preprocess_command(cmd),
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            )
            all_input_ids.append(enc["input_ids"].squeeze(0))
            all_attn_mask.append(enc["attention_mask"].squeeze(0))

        # Pad sequence dimension to max_seq_len
        pad_ids  = torch.zeros(self.max_length, dtype=torch.long)
        pad_mask = torch.zeros(self.max_length, dtype=torch.long)
        while len(all_input_ids) < self.max_seq_len:
            all_input_ids.append(pad_ids)
            all_attn_mask.append(pad_mask)

        return {
            "input_ids":      torch.stack(all_input_ids),      # (T_pad, max_length)
            "attention_mask": torch.stack(all_attn_mask),      # (T_pad, max_length)
            "seq_len":        torch.tensor(T, dtype=torch.long),
            "label":          torch.tensor(label, dtype=torch.long),
        }


# ── CSV loading helpers ───────────────────────────────────────────────────────

def load_csv(path: str) -> pd.DataFrame:
    """Load dataset CSV, ensuring Command and Label columns exist."""
    df = pd.read_csv(path)
    df = df.rename(columns={"command": "Command", "label": "Label"})
    df = df[["Command", "Label"]].dropna()
    df["Label"] = df["Label"].astype(int)
    return df
