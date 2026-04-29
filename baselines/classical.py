"""
Classical ML baselines (paper Section 4.2 and BERT LOLWTC notebook):
  BoW   + Random Forest
  TF-IDF + Random Forest  (ngram_range=(1,3) per paper)
  Random Forest  (standalone, with 20-feature engineering from LOTL Attack Detection notebook)
  XGBoost
  Gradient Boosting
  Token–Char Fusion (TF-IDF + char n-grams 3-5, paper Section 4.2)

All models share the same sklearn Pipeline interface:
  .fit(commands, labels)
  .predict(commands) -> np.ndarray
  .predict_proba(commands) -> np.ndarray  (for ROC/FPR computation)
"""

import re
import numpy as np
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import FunctionTransformer
import xgboost as xgb

from halo_bert.config import (
    BOW_MAX_FEATURES, TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE,
    RF_N_ESTIMATORS, RF_MAX_DEPTH,
    XGB_N_ESTIMATORS, XGB_MAX_DEPTH, XGB_LR,
    GB_N_ESTIMATORS, GB_MAX_DEPTH, GB_LR,
    RANDOM_SEED,
)
from halo_bert.dataset import preprocess_command


# ── 20-feature extractor (LOTL Attack Detection notebook, Section 4 cell 3) ───

def extract_command_features(command: str) -> dict:
    """
    20 complexity/obfuscation features exactly as implemented in the
    LOTL Attack Detection notebook.
    """
    if not command:
        return {f"feature_{i}": 0.0 for i in range(20)}

    cmd = str(command)
    cmd_lower = cmd.lower()
    features = {}

    # Syntactic complexity
    features["cmd_length"]      = min(len(cmd) / 1000.0, 1.0)
    features["token_count"]     = min(len(cmd.split()) / 50.0, 1.0)
    features["pipe_count"]      = min(cmd.count("|") / 5.0, 1.0)
    features["redirect_count"]  = min((cmd.count(">") + cmd.count("<")) / 3.0, 1.0)
    features["semicolon_count"] = min(cmd.count(";") / 3.0, 1.0)
    features["ampersand_count"] = min(cmd.count("&") / 3.0, 1.0)

    # Encoding / obfuscation
    features["has_base64"]    = float(bool(re.search(r"[A-Za-z0-9+/]{40,}={0,2}", cmd)))
    features["has_encoding"]  = float(bool(re.search(r"-enc|-en|-e\s+", cmd_lower)))
    features["has_hex"]       = float(bool(re.search(r"0x[0-9a-f]{4,}", cmd_lower)))
    features["special_char_ratio"] = sum(
        not c.isalnum() and not c.isspace() for c in cmd
    ) / max(len(cmd), 1)

    # Quotes and nesting
    features["nested_quotes"]  = min(cmd.count('"') / 4.0, 1.0)
    features["single_quotes"]  = min(cmd.count("'") / 4.0, 1.0)
    features["backtick_count"] = min(cmd.count("`") / 3.0, 1.0)

    # Suspicious patterns
    features["has_network_path"] = float(bool(re.search(r"\\\\[\d\.]+", cmd)))
    features["has_localhost"]    = float(bool(re.search(r"(127\.0\.0\.1|localhost)", cmd_lower)))
    features["has_admin_share"]  = float(bool(re.search(r"\\\\.*\\(ADMIN|C)\$", cmd, re.I)))
    features["has_powershell"]   = float(bool(re.search(r"powershell|pwsh", cmd_lower)))
    features["has_wmi"]          = float(bool(re.search(r"wmic|Get-WmiObject", cmd, re.I)))

    # Command chaining
    features["command_chain_length"] = min(
        cmd_lower.count("|") + cmd_lower.count(";") + cmd_lower.count("&"), 5.0
    ) / 5.0

    # Silent mode
    features["has_quiet_mode"] = float(bool(re.search(r"/[qQ]|-quiet|-silent", cmd)))

    return features


def _feature_matrix(commands):
    rows = [list(extract_command_features(c).values()) for c in commands]
    return np.array(rows, dtype=np.float32)


# ── Preprocessing helper for sklearn pipelines ─────────────────────────────────

def _preprocess(commands):
    return [preprocess_command(c) for c in commands]


_preproc = FunctionTransformer(_preprocess, validate=False)


# ── Model factories ────────────────────────────────────────────────────────────

def build_bow_rf() -> Pipeline:
    """Bag-of-Words + Random Forest (BERT LOLWTC notebook, Section 6)."""
    return Pipeline([
        ("preproc", _preproc),
        ("bow",     CountVectorizer(max_features=BOW_MAX_FEATURES, token_pattern=r"\b\w+\b")),
        ("clf",     RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS, max_depth=RF_MAX_DEPTH,
            random_state=RANDOM_SEED, n_jobs=-1,
        )),
    ])


def build_tfidf_rf() -> Pipeline:
    """TF-IDF (1-3 grams) + Random Forest (paper Section 4.2)."""
    return Pipeline([
        ("preproc", _preproc),
        ("tfidf",   TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES, ngram_range=TFIDF_NGRAM_RANGE,
            token_pattern=r"\b\w+\b",
        )),
        ("clf",     RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS, max_depth=RF_MAX_DEPTH,
            random_state=RANDOM_SEED, n_jobs=-1,
        )),
    ])


def build_random_forest():
    """RF with 20-feature engineering (LOTL Attack Detection notebook)."""
    class RF20:
        def __init__(self):
            self.clf = RandomForestClassifier(
                n_estimators=RF_N_ESTIMATORS, max_depth=RF_MAX_DEPTH,
                random_state=RANDOM_SEED, n_jobs=-1,
            )
        def fit(self, commands, labels):
            self.clf.fit(_feature_matrix(commands), labels)
            return self
        def predict(self, commands):
            return self.clf.predict(_feature_matrix(commands))
        def predict_proba(self, commands):
            return self.clf.predict_proba(_feature_matrix(commands))
    return RF20()


def build_xgboost():
    """XGBoost with 20-feature engineering (paper Section 4.2)."""
    class XGB20:
        def __init__(self):
            self.clf = xgb.XGBClassifier(
                n_estimators=XGB_N_ESTIMATORS, max_depth=XGB_MAX_DEPTH,
                learning_rate=XGB_LR, random_state=RANDOM_SEED,
                eval_metric="logloss",
            )
        def fit(self, commands, labels):
            self.clf.fit(_feature_matrix(commands), labels)
            return self
        def predict(self, commands):
            return self.clf.predict(_feature_matrix(commands))
        def predict_proba(self, commands):
            return self.clf.predict_proba(_feature_matrix(commands))
    return XGB20()


def build_gradient_boosting():
    """Gradient Boosting with 20-feature engineering (paper Section 4.2)."""
    class GB20:
        def __init__(self):
            self.clf = GradientBoostingClassifier(
                n_estimators=GB_N_ESTIMATORS, max_depth=GB_MAX_DEPTH,
                learning_rate=GB_LR, random_state=RANDOM_SEED,
            )
        def fit(self, commands, labels):
            self.clf.fit(_feature_matrix(commands), labels)
            return self
        def predict(self, commands):
            return self.clf.predict(_feature_matrix(commands))
        def predict_proba(self, commands):
            return self.clf.predict_proba(_feature_matrix(commands))
    return GB20()


def build_token_char_fusion():
    """
    TF-IDF (token) + character n-grams (3-5) fused via FeatureUnion + RF.
    Paper Section 4.2: "combining TF-IDF with character n-grams (3-5 grams)".
    """
    return Pipeline([
        ("preproc", _preproc),
        ("features", FeatureUnion([
            ("token", TfidfVectorizer(
                max_features=TFIDF_MAX_FEATURES, ngram_range=TFIDF_NGRAM_RANGE,
                token_pattern=r"\b\w+\b",
            )),
            ("char", TfidfVectorizer(
                max_features=2000, analyzer="char", ngram_range=(3, 5),
            )),
        ])),
        ("clf", RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS, max_depth=RF_MAX_DEPTH,
            random_state=RANDOM_SEED, n_jobs=-1,
        )),
    ])


# ── Registry used by run_full_experiment.py ───────────────────────────────────

CLASSICAL_BASELINES = {
    "BoW":               build_bow_rf,
    "TF-IDF":            build_tfidf_rf,
    "Random Forest":     build_random_forest,
    "XGBoost":           build_xgboost,
    "Gradient Boosting": build_gradient_boosting,
    "Token-Char Fusion": build_token_char_fusion,
}
