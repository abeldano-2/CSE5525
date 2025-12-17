"""
Emotion-Engagement Prediction Pipeline

A representation learning approach to analyze how emotional and sentiment
content of tweets affects retweet and favorite counts.

Modules:
    config      - Constants, dimensions, and hyperparameters
    data        - Data loading, splitting, and PyTorch datasets
    features    - Feature extraction (text embeddings, sentiment, emotion, VAD)
    models      - Engagement MLP and affect probe definitions
    subspace    - Affect subspace projection and decomposition
    training    - Training loop, evaluation, and model comparison
    analysis    - SHAP feature importance and result summarization
    utils       - I/O utilities and reproducibility helpers

Scripts:
    scripts/run_pipeline.py   - Full end-to-end execution
    scripts/run_ablations.py  - Focused ablation experiments
"""

from .config import (
    EMBEDDING_DIM,
    METADATA_DIM,
    AFFECT_DIM,
    FULL_DIM,
    INDEX_SLICES,
    TRAINING_DEFAULTS,
    PROBE_R2_THRESHOLD,
    VAD_MIN_MATCHED_TOKENS,
)

from .models import ModelVariant, EngagementMLP, AffectProbe
from .subspace import AffectSubspace
from .features import FeatureBuilder, VADScorer

__all__ = [
    "EMBEDDING_DIM",
    "METADATA_DIM", 
    "AFFECT_DIM",
    "FULL_DIM",
    "INDEX_SLICES",
    "TRAINING_DEFAULTS",
    "PROBE_R2_THRESHOLD",
    "VAD_MIN_MATCHED_TOKENS",
    "ModelVariant",
    "EngagementMLP",
    "AffectProbe",
    "AffectSubspace",
    "FeatureBuilder",
    "VADScorer",
]