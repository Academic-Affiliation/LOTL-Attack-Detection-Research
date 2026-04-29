"""
All hyperparameters sourced directly from the paper and existing notebooks.
Do not change values here without updating the corresponding paper section.
"""

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR = "."
TRAIN_CSV      = "balanced_combined_lotl_dataset.csv"   # 7049 rows, Command/Label
VOLT_CSV       = "complete_volttyphoon_dataset.csv"     # 102 rows, Command/Label
OBFUSC_CSV     = "enhanced_lotl_obfuscated_dataset.csv" # 250 rows, Command/Label

# ── Reproducibility ───────────────────────────────────────────────────────────
RANDOM_SEED = 42

# ── Data splits (paper Section 4.1: stratified 70/15/15) ─────────────────────
TRAIN_RATIO = 0.70   # → 4934 samples
VAL_RATIO   = 0.15   # → 1058 samples
TEST_RATIO  = 0.15   # → 1057 samples

# Near-duplicate removal threshold (paper Section 4.1)
LEVENSHTEIN_THRESHOLD = 0.90  # remove pairs with >90% Levenshtein similarity

# ── BERT / tokeniser (paper Section 3.2; BERT LOLWTC notebook) ───────────────
BERT_MODEL_NAME = "bert-base-uncased"
BERT_MAX_LENGTH = 128   # max subword tokens; from BERT notebook

# ── Character CNN (paper Section 3.4) ────────────────────────────────────────
CHAR_MAX_LENGTH    = 512   # fixed ASCII sequence length
CHAR_VOCAB_SIZE    = 256   # ASCII range [0, 255]
CHAR_EMBED_DIM     = 32    # W_char ∈ R^{256×32}
CHAR_FILTER_SIZES  = (3, 4, 5)   # three parallel convolutions
CHAR_NUM_FILTERS   = 64          # 64 filters each → 64*3 = 192-dim output

# ── Inter-command attention (paper Section 3.3) ───────────────────────────────
TEMPORAL_NUM_HEADS = 4    # H=4 heads
TEMPORAL_D_K       = 192  # d_k = 768/4 = 192

# ── Anomaly autoencoder (paper Section 3.5) ──────────────────────────────────
AE_HIDDEN_DIM        = 256   # encoder step 1: 768 → 256
AE_LATENT_DIM        = 128   # encoder step 2: 256 → 128 (z_anomaly)
ANOMALY_THRESHOLD    = 0.11  # τ: 95th-percentile of benign-val recon errors

# ── Fusion dimensions (paper Section 3.6) ────────────────────────────────────
# h_concat = [h_BERT(768) ; h_char(192) ; h_temporal(768) ; z_anomaly(128)]
CONCAT_DIM     = 768 + 192 + 768 + 128  # = 1856
FUSION_DIM     = 768   # W_fuse ∈ R^{768×1856}
CLASSIFIER_DIM = 384   # 768 → 384 → 2

# ── Training (paper Section 4.2) ─────────────────────────────────────────────
LEARNING_RATE           = 2e-5   # AdamW
BATCH_SIZE              = 32
NUM_EPOCHS              = 10
EARLY_STOPPING_PATIENCE = 3
DROPOUT                 = 0.3
CLASS_WEIGHT_BENIGN     = 0.53   # inverse-frequency weight for label=0
CLASS_WEIGHT_MALICIOUS  = 8.08   # inverse-frequency weight for label=1
ANOMALY_LAMBDA          = 0.1    # λ: weight of reconstruction loss
ALPHA                   = 0.6    # fraction of single-command batches

# ── Sequence training (paper Section 3.3) ────────────────────────────────────
MAX_SEQ_LEN    = 5    # T ≤ 5 commands per sequence
SEQ_BATCH_FRAC = 0.4  # 40% of batches are multi-command sequences

# ── Baseline hyperparameters ──────────────────────────────────────────────────
# Classical ML (paper Section 4.2; BERT LOLWTC notebook)
BOW_MAX_FEATURES   = 5000
TFIDF_MAX_FEATURES = 5000
TFIDF_NGRAM_RANGE  = (1, 3)   # paper Section 4.2
RF_N_ESTIMATORS    = 100
RF_MAX_DEPTH       = None
XGB_N_ESTIMATORS   = 100
XGB_MAX_DEPTH      = 10
XGB_LR             = 0.1
GB_N_ESTIMATORS    = 100
GB_MAX_DEPTH       = 10
GB_LR              = 0.1

# BERT baseline (identical config to HALO-BERT BERT pathway; from notebook)
BERT_BASELINE_HIDDEN = 128   # FC: 770→128→64→2 (768 BERT + 2 manual features)
BERT_BASELINE_EPOCHS = 10
BERT_BASELINE_LR     = 2e-5

# CharCNN standalone baseline (paper Section 4.2: "128 filters each")
CHAR_NUM_FILTERS_BASELINE = 128

# LOLWTC (Word2Vec + TextCNN; from BERT LOLWTC notebook)
W2V_VECTOR_SIZE        = 400            # paper Section 3.2 (LOLWTC)
W2V_WINDOW             = 5
CNN_FILTER_SIZES_LOLWTC = [2, 3, 4, 5, 6]  # from notebook
CNN_NUM_FILTERS_LOLWTC  = 100               # from notebook
CNN_MAX_LENGTH          = 100               # 100×400 matrix per notebook
LOLWTC_EPOCHS           = 30
LOLWTC_LR               = 0.001

# BERT + Augmentation (paper Section 4.2 augmentation fractions)
AUG_FRACTIONS = {
    "base64":        0.20,
    "concatenation": 0.15,
    "alias":         0.15,
    "case":          0.10,
    "char_encoding": 0.10,
    "format_op":     0.10,
    "backtick":      0.10,
    "whitespace":    0.10,
}
