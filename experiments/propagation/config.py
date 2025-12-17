"""
Configuration constants for the emotion-engagement prediction pipeline.

Defines feature dimensions, index ranges, model identifiers, and training hyperparameters.
"""

# =============================================================================
# Feature Dimensions
# =============================================================================

EMBEDDING_DIM = 384      # sentence-transformers/all-MiniLM-L6-v2 output
METADATA_DIM = 11        # contextual tweet/user features (includes cyclical hour encoding)
AFFECT_DIM = 15          # sentiment (1) + VAD (3) + emotions (11)
FULL_DIM = EMBEDDING_DIM + METADATA_DIM + AFFECT_DIM  # 410

# =============================================================================
# Index Slices (zero-based, end-exclusive for Python slicing)
# =============================================================================

INDEX_SLICES = {
    "text": (0, 384),
    "metadata": (384, 395),
    "affect": (395, 410),
}

# =============================================================================
# Probe Configuration (from README_v1 Section 5.1.1: Linearity Check)
# =============================================================================

# Minimum R² threshold for including an affect dimension in the subspace.
# Dimensions with R² below this are excluded from the decomposition.
PROBE_R2_THRESHOLD = 0.05

# Minimum VAD lexicon coverage: fraction of tweets that must have >= this many
# matched tokens for VAD scores to be considered reliable (README_v2 Section 2.2.1)
VAD_MIN_MATCHED_TOKENS = 3

# Input dimensions for each model variant
MODEL_INPUT_DIMS = {
    "full": FULL_DIM,                          # 410
    "text_meta": EMBEDDING_DIM + METADATA_DIM, # 395
    "shuffled": FULL_DIM,                      # 410 (same structure, permuted affect)
    "affect_only": AFFECT_DIM + METADATA_DIM,  # 26
    "non_affect": EMBEDDING_DIM + METADATA_DIM # 395
}

# =============================================================================
# Feature Names (in canonical order)
# =============================================================================

METADATA_FEATURES = [
    "log_followers",
    "log_friends",
    "log_statuses",
    "user_verified",
    "tweet_length_tokens",
    "has_url",
    "has_hashtag",
    "has_mention",
    "has_media",
    "hour_sin",    # sin(2π·hour/24) - cyclical encoding
    "hour_cos",    # cos(2π·hour/24) - cyclical encoding
]

# Affect block: polarity (1) + VAD (3) + 11 emotion probabilities
AFFECT_FEATURES = [
    "sentiment_polarity",
    "valence",
    "arousal",
    "dominance",
    # Emotion probabilities (order from twitter-roberta-base-emotion-latest id2label)
    "anger",
    "anticipation",
    "disgust",
    "fear",
    "joy",
    "love",
    "optimism",
    "pessimism",
    "sadness",
    "surprise",
    "trust",
]

# =============================================================================
# Pretrained Model Identifiers
# =============================================================================

PRETRAINED_MODELS = {
    "text_encoder": "sentence-transformers/all-MiniLM-L6-v2",
    "sentiment": "cardiffnlp/twitter-roberta-base-sentiment-latest",
    "emotion": "cardiffnlp/twitter-roberta-base-emotion-latest",
}

# =============================================================================
# Training Defaults
# =============================================================================

TRAINING_DEFAULTS = {
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "batch_size": 512,
    "epochs": 15,
    "patience": 5,
    "dropout": 0.1,
    "hidden_dims": (512, 256),
}

# Ridge regression probe
PROBE_DEFAULTS = {
    "alpha": 1.0,
    "fit_intercept": True,
}

# Data split ratios
SPLIT_RATIOS = {
    "train": 0.70,
    "val": 0.15,
    "test": 0.15,
}

RANDOM_SEED = 562