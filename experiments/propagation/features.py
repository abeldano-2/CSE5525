"""
Feature extraction components for the emotion-engagement pipeline.

Includes wrappers for pretrained models (text encoder, sentiment, emotion)
and lexicon-based VAD scoring, plus orchestration for building the full
feature matrix Z = [H; C; A].
"""

import math
import numpy as np
import pandas as pd
import re
import warnings
from collections import Counter
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import torch
from sentence_transformers import SentenceTransformer
from torch.nn.functional import softmax
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import (
    PRETRAINED_MODELS, 
    EMBEDDING_DIM, 
    METADATA_DIM, 
    AFFECT_DIM
)


# =============================================================================
# Text Encoder
# =============================================================================

class TextEncoder:
    """
    Wrapper for SentenceTransformer model. Uses pretrained model from config.
    Default model is sentence-transformers/all-MiniLM-L6-v2.
    
    Encodes tweet text into dense vectors.
    """
    
    def __init__(self, model_name: str = None, device: str = None):
        """
        Initialize the TextEncoder with a pretrained model.

        Args:
            model_name: HuggingFace model identifier (default from config)
            device: 'cuda' or 'cpu' (auto-detected if None)
        """
        if model_name is None:
            model_name = PRETRAINED_MODELS["text_encoder"]
        
        self.model = SentenceTransformer(model_name, device=device)
        self.dim = EMBEDDING_DIM
    
    def encode(self, texts: List[str], batch_size: int = 64, show_progress: bool = True) -> np.ndarray:
        """
        Encode text strings into dense embeddings.
        
        Args:
            texts: List of tweet strings
            batch_size: Encoding batch size
            show_progress: Show progress bar
            
        Returns:
            Array of shape (N, 384)
        """
        embeddings = self.model.encode(
            texts, 
            batch_size=batch_size, 
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        return embeddings.astype(np.float32)


# =============================================================================
# Sentiment Scorer
# =============================================================================

class SentimentScorer:
    """
    Wrapper for RoBERTa-base sentiment model. Uses pretrained model from config.
    Default model is cardiffnlp/twitter-roberta-base-sentiment-latest.
    
    Returns polarity score: p(positive) - p(negative).
    """
    
    def __init__(self, model_name: str = None, device: str = None):
        """
        Args:
            model_name: HuggingFace model identifier
            device: Device for inference
        """
        if model_name is None:
            model_name = PRETRAINED_MODELS["sentiment"]
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        # Move to device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model.to(device)
        self.model.eval()
        
        # Get label mapping (negative=0, neutral=1, positive=2)
        self.id2label = self.model.config.id2label
    
    def score(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        """
        Compute sentiment polarity for each text.
        
        Polarity = p(positive) - p(negative), range [-1, 1].
        
        Args:
            texts: List of tweet strings
            batch_size: Processing batch size
            
        Returns:
            Array of shape (N,) with polarity scores
        """
        polarities = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            inputs = self.tokenizer(
                batch, 
                return_tensors="pt", 
                truncation=True, 
                padding=True, 
                max_length=128
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = softmax(outputs.logits, dim=-1).cpu().numpy()  # numpy has to be on CPU
            
            # Find positive and negative indices from id2label
            pos_idx = [k for k, v in self.id2label.items() if "pos" in v.lower()][0]
            neg_idx = [k for k, v in self.id2label.items() if "neg" in v.lower()][0]
            
            batch_polarity = probs[:, pos_idx] - probs[:, neg_idx]
            polarities.append(batch_polarity)
        
        return np.concatenate(polarities).astype(np.float32)


# =============================================================================
# Emotion Classifier
# =============================================================================

class EmotionClassifier:
    """
    Wrapper for RoBERTa-base emotion model. Uses pretrained model from config.
    Default model is cardiffnlp/twitter-roberta-base-emotion-latest.
    
    Emotions include: anger, anticipation, disgust, fear, joy, love, optimism, 
                      pessimism, sadness, surprise, trust

    Returns probability distribution over 11 emotion categories.
    """
    
    def __init__(self, model_name: str = None, device: str = None):
        """
        Args:
            model_name: HuggingFace model identifier
            device: Device for inference
        """
        if model_name is None:
            model_name = PRETRAINED_MODELS["emotion"]
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model.to(device)
        self.model.eval()
        
        # Store label order for reference
        self.id2label = self.model.config.id2label
        self.num_labels = len(self.id2label)
    
    def predict(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        """
        Predict emotion probabilities for each text.
        
        Args:
            texts: List of tweet strings
            batch_size: Processing batch size
            
        Returns:
            Array of shape (N, 11) with emotion probabilities in id2label order
        """
        all_probs = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = softmax(outputs.logits, dim=-1).cpu().numpy()  # numpy has to be on CPU
            
            all_probs.append(probs)
        
        return np.concatenate(all_probs).astype(np.float32)
    
    def get_label_names(self) -> List[str]:
        """Return emotion labels in prediction order."""
        return [self.id2label[i] for i in range(self.num_labels)]


# =============================================================================
# VAD Lexicon Scorer
# =============================================================================

class VADScorer:
    """
    Lexicon-based Valence-Arousal-Dominance scorer using NRC VAD Lexicon.
    
    Aggregates word-level VAD scores via mean or TF-IDF weighted mean.
    """
    
    def __init__(self, lexicon_path: Optional[str] = None, use_tfidf: bool = False):
        """
        Args:
            lexicon_path: Path to NRC VAD lexicon file (tab-separated: word, V, A, D)
                         If None, attempts to load from default location.
            use_tfidf: If True, weight token contributions by inverse document frequency
        """
        self.lexicon: Dict[str, Tuple[float, float, float]] = {}
        self.use_tfidf = use_tfidf
        self.idf_weights: Optional[Dict[str, float]] = None
        
        # Coverage tracking
        self._last_coverage_stats: Optional[Dict] = None
        
        if lexicon_path is not None:
            self._load_lexicon(lexicon_path)
    
    def _load_lexicon(self, path: str):
        """
        Load NRC VAD lexicon from file.
        
        Args:
            path: Path to lexicon file
        """
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i == 0:  # Skip header row
                    continue
                parts = line.strip().split("\t")
                if len(parts) >= 4:
                    word = parts[0].lower()
                    v, a, d = float(parts[1]), float(parts[2]), float(parts[3])
                    self.lexicon[word] = (v, a, d)
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Simple whitespace + punctuation tokenization.
        
        Args:
            text: Text string to tokenize
        
        Returns:
            List of tokens
        """
        text = text.lower()
        tokens = re.findall(r'\b[a-z]+\b', text)
        return tokens
    
    def compute_idf(self, texts: List[str]) -> None:
        """
        Compute IDF weights from a corpus for TF-IDF weighting.
        
        Args:
            texts: List of documents to compute IDF from
        """
        n_docs = len(texts)
        doc_freq = Counter()
        
        for text in texts:
            tokens = set(self._tokenize(text))
            for token in tokens:
                if token in self.lexicon:
                    doc_freq[token] += 1
        
        self.idf_weights = {
            word: math.log(n_docs / (1 + freq))
            for word, freq in doc_freq.items()
        }
    
    def score_tweet(self, text: str) -> Tuple[Tuple[float, float, float], int]:
        """
        Compute VAD scores for a single tweet.
        
        Returns mean V, A, D over matched tokens (0.5 default if no matches),
        along with the count of matched tokens.
        
        Args:
            text: Tweet string
            
        Returns:
            ((valence, arousal, dominance), n_matched) tuple
        """
        tokens = self._tokenize(text)
        
        v_scores, a_scores, d_scores = [], [], []
        weights = []
        
        for token in tokens:
            if token in self.lexicon:
                v, a, d = self.lexicon[token]
                v_scores.append(v)
                a_scores.append(a)
                d_scores.append(d)
                
                # TF-IDF weight or uniform
                if self.use_tfidf and self.idf_weights is not None:
                    weights.append(self.idf_weights.get(token, 1.0))
                else:
                    weights.append(1.0)
        
        n_matched = len(v_scores)
        
        # Default to neutral (0.5) if no matches
        if n_matched == 0:
            return ((0.5, 0.5, 0.5), 0)
        
        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize
        
        return (
            (float(np.dot(weights, v_scores)),
             float(np.dot(weights, a_scores)),
             float(np.dot(weights, d_scores))),
            n_matched
        )
    
    def score(self, texts: List[str], min_matches: int = 3) -> Tuple[np.ndarray, Dict]:
        """
        Compute VAD scores for multiple texts with coverage statistics.
        
        Args:
            texts: List of tweet strings
            min_matches: Minimum matched tokens for "adequate coverage"
            
        Returns:
            Tuple of:
                - Array of shape (N, 3) with [valence, arousal, dominance] columns
                - Coverage statistics dict
        """
        scores = []
        match_counts = []
        
        for t in texts:
            (vad, n_matched) = self.score_tweet(t)
            scores.append(vad)
            match_counts.append(n_matched)
        
        match_counts = np.array(match_counts)
        
        # Compute coverage statistics
        coverage_stats = {
            "total_tweets": len(texts),
            "tweets_with_any_match": int(np.sum(match_counts > 0)),
            "tweets_with_adequate_coverage": int(np.sum(match_counts >= min_matches)),
            "coverage_fraction": float(np.mean(match_counts >= min_matches)),
            "mean_matches_per_tweet": float(np.mean(match_counts)),
            "median_matches_per_tweet": float(np.median(match_counts)),
            "min_matches_threshold": min_matches,
        }
        
        self._last_coverage_stats = coverage_stats
        
        return np.array(scores, dtype=np.float32), coverage_stats
    
    def get_last_coverage_stats(self) -> Optional[Dict]:
        """Return coverage statistics from the last score() call."""
        return self._last_coverage_stats


# =============================================================================
# Metadata Extraction
# =============================================================================

def extract_metadata(df: pd.DataFrame, verbose: bool = True) -> np.ndarray:
    """
    Extract the 11 metadata features from a DataFrame.
    
    Will attempt to derive features from available columns if standard columns
    are not present (e.g., derive has_hashtag from 'hashtags' column).
    
    Expected columns (or derivable alternatives):
        - followers_count, friends_count, statuses_count (user metrics)
        - user_verified (or verified)
        - text/clean_tweet/original_text (for token length)
        - has_url/has_hashtag/has_mention/has_media (or hashtags/user_mentions to derive)
        - created_at (for hour extraction)
    
    Args:
        df: DataFrame with tweet data
        verbose: Print warnings about missing columns
        
    Returns:
        Array of shape (N, 11) with metadata features in canonical order
    """
    n = len(df)
    C = np.zeros((n, METADATA_DIM), dtype=np.float32)
    
    # Track which features we successfully extracted
    extracted = []
    missing = []
    
    # 0: log_followers
    if "followers_count" in df.columns:
        C[:, 0] = np.log1p(pd.to_numeric(df["followers_count"], errors='coerce').fillna(0).values)
        extracted.append("log_followers")
    else:
        missing.append("followers_count")
    
    # 1: log_friends
    if "friends_count" in df.columns:
        C[:, 1] = np.log1p(pd.to_numeric(df["friends_count"], errors='coerce').fillna(0).values)
        extracted.append("log_friends")
    else:
        missing.append("friends_count")
    
    # 2: log_statuses
    if "statuses_count" in df.columns:
        C[:, 2] = np.log1p(pd.to_numeric(df["statuses_count"], errors='coerce').fillna(0).values)
        extracted.append("log_statuses")
    else:
        missing.append("statuses_count")
    
    # 3: user_verified (binary)
    verified_col = None
    for col in ["user_verified", "verified"]:
        if col in df.columns:
            verified_col = col
            break
    if verified_col:
        C[:, 3] = df[verified_col].fillna(0).astype(float).values
        extracted.append("user_verified")
    else:
        missing.append("user_verified")
    
    # 4: tweet_length_tokens (derive from text if needed)
    text_col = None
    for col in ["tweet_length_tokens", "clean_tweet", "original_text", "text"]:
        if col in df.columns:
            text_col = col
            break
    if text_col:
        if text_col == "tweet_length_tokens":
            C[:, 4] = df[text_col].values
        else:
            C[:, 4] = df[text_col].fillna("").astype(str).str.split().str.len().fillna(0).values
        extracted.append(f"tweet_length (from {text_col})")
    else:
        missing.append("tweet_length")
    
    # 5: has_url (derive from original_text which preserves URLs, not clean_tweet)
    if "has_url" in df.columns:
        C[:, 5] = df["has_url"].fillna(0).astype(float).values
        extracted.append("has_url")
    else:
        # Check original_text first (clean_tweet often has URLs removed)
        url_pattern = r'https?://|www\.|t\.co/'
        url_col = None
        for col in ["original_text", "text", "clean_tweet"]:
            if col in df.columns:
                url_col = col
                break
        if url_col:
            C[:, 5] = df[url_col].fillna("").astype(str).str.contains(url_pattern, regex=True, na=False).astype(float).values
            extracted.append(f"has_url (from {url_col})")
        else:
            missing.append("has_url")
    
    # 6: has_hashtag (derive from 'hashtags' column if available)
    if "has_hashtag" in df.columns:
        C[:, 6] = df["has_hashtag"].fillna(0).astype(float).values
        extracted.append("has_hashtag")
    elif "hashtags" in df.columns:
        # Non-empty and not 'nan' string means has hashtag
        C[:, 6] = (df["hashtags"].fillna("").astype(str).str.len() > 0).astype(float).values
        C[:, 6] = np.where(df["hashtags"].astype(str) == "nan", 0.0, C[:, 6])
        extracted.append("has_hashtag (from hashtags)")
    elif text_col:
        # Detect # in text
        C[:, 6] = df[text_col].fillna("").astype(str).str.contains(r'#\w+', regex=True, na=False).astype(float).values
        extracted.append("has_hashtag (derived)")
    else:
        missing.append("has_hashtag")
    
    # 7: has_mention (derive from 'user_mentions' column if available)
    if "has_mention" in df.columns:
        C[:, 7] = df["has_mention"].fillna(0).astype(float).values
        extracted.append("has_mention")
    elif "user_mentions" in df.columns:
        # Non-empty and not 'nan' string means has mention
        C[:, 7] = (df["user_mentions"].fillna("").astype(str).str.len() > 0).astype(float).values
        C[:, 7] = np.where(df["user_mentions"].astype(str) == "nan", 0.0, C[:, 7])
        extracted.append("has_mention (from user_mentions)")
    elif text_col:
        # Detect @ in text
        C[:, 7] = df[text_col].fillna("").astype(str).str.contains(r'@\w+', regex=True, na=False).astype(float).values
        extracted.append("has_mention (derived)")
    else:
        missing.append("has_mention")
    
    # 8: has_media (often not available in CSV exports)
    if "has_media" in df.columns:
        C[:, 8] = df["has_media"].fillna(0).astype(float).values
        extracted.append("has_media")
    elif "media" in df.columns:
        C[:, 8] = (df["media"].fillna("").astype(str).str.len() > 0).astype(float).values
        extracted.append("has_media (from media)")
    else:
        # Leave as 0 - media info often not in CSV exports
        missing.append("has_media (defaulting to 0)")
    
    # 9-10: Cyclical hour encoding
    # Using sin/cos respects circular structure: hour 23 and hour 0 are adjacent
    hour = np.zeros(n)
    if "created_at" in df.columns:
        # Use errors='coerce' and format='mixed' to handle varied/invalid date formats
        timestamps = pd.to_datetime(df["created_at"], errors='coerce', format='mixed')
        # Check if we have hour information (not just dates)
        if timestamps.dt.hour.notna().any() and timestamps.dt.hour.max() > 0:
            hour = timestamps.dt.hour.fillna(12).values  # Default to noon if missing
            extracted.append("hour_sin/cos (from created_at)")
        else:
            # Only date available, no time - use random or constant
            hour = np.full(n, 12.0)  # Default to noon
            extracted.append("hour_sin/cos (no time info, defaulting to noon)")
    elif "hour_of_day" in df.columns:
        hour = df["hour_of_day"].values * 24  # Assume normalized 0-1
        extracted.append("hour_sin/cos (from hour_of_day)")
    else:
        missing.append("hour (defaulting to noon)")
        hour = np.full(n, 12.0)
    
    C[:, 9] = np.sin(2 * np.pi * hour / 24)   # hour_sin
    C[:, 10] = np.cos(2 * np.pi * hour / 24)  # hour_cos
    
    # Report what was successfully extracted
    if verbose:
        print(f"  Metadata extraction: {len(extracted)}/11 features from data")
        if missing:
            warnings.warn(f"Missing columns (using defaults): {missing}")
    
    return C


# =============================================================================
# Affect Vector Assembly
# =============================================================================

def build_affect_vector(
    polarity: np.ndarray,
    vad: np.ndarray,
    emotions: np.ndarray
) -> np.ndarray:
    """
    Assemble the 15-dimensional affect vector in canonical order.
    
    Order: [polarity (1), V, A, D (3), emotion_probs (11)]
    
    Args:
        polarity: Shape (N,) sentiment polarity scores
        vad: Shape (N, 3) valence, arousal, dominance scores
        emotions: Shape (N, 11) emotion probabilities
        
    Returns:
        Array of shape (N, 15)
    """
    n = len(polarity)
    A = np.zeros((n, AFFECT_DIM), dtype=np.float32)
    
    A[:, 0] = polarity   # index 0: polarity
    A[:, 1:4] = vad      # indices 1-3: VAD
    A[:, 4:] = emotions  # indices 4-14: emotions
    
    return A


# =============================================================================
# Feature Builder
# =============================================================================

class FeatureBuilder:
    """
    Orchestrates all feature extraction components.
    
    Combines text encoder, sentiment scorer, emotion classifier, and VAD scorer
    to produce the full feature matrix Z = [H; C; A].
    """
    
    def __init__(
        self,
        text_encoder: Optional[TextEncoder] = None,
        sentiment_scorer: Optional[SentimentScorer] = None,
        emotion_classifier: Optional[EmotionClassifier] = None,
        vad_scorer: Optional[VADScorer] = None,
        device: str = None
    ):
        """
        Initialize the FeatureBuilder with the necessary components.
        
        Args:
            text_encoder: TextEncoder instance (created if None)
            sentiment_scorer: SentimentScorer instance (created if None)
            emotion_classifier: EmotionClassifier instance (created if None)
            vad_scorer: VADScorer instance (must be provided with lexicon loaded)
            device: Device for neural models
        """
        self.text_encoder = text_encoder or TextEncoder(device=device)
        self.sentiment_scorer = sentiment_scorer or SentimentScorer(device=device)
        self.emotion_classifier = emotion_classifier or EmotionClassifier(device=device)
        self.vad_scorer = vad_scorer  # May be None if lexicon unavailable
        
        # Track VAD coverage for reporting
        self._vad_coverage_stats: Optional[Dict] = None
        
        # Verify emotion label count
        self._verify_emotion_labels()
    
    def _verify_emotion_labels(self):
        """Verify and report emotion label configuration."""
        labels = self.emotion_classifier.get_label_names()
        expected_count = 11  # From AFFECT_DIM specification
        actual_count = len(labels)
        
        if actual_count != expected_count:
            warnings.warn(
                f"Emotion classifier has {actual_count} labels (expected {expected_count}). "
                f"Labels: {labels}. Affect vector dimension may need adjustment."
            )
        
        self._emotion_labels = labels
    
    def get_emotion_labels(self) -> List[str]:
        """Return the emotion labels from the classifier's id2label mapping."""
        return self._emotion_labels
    
    def get_vad_coverage_stats(self) -> Optional[Dict]:
        """Return VAD coverage statistics from last feature extraction."""
        return self._vad_coverage_stats
    
    def build_features(
        self,
        df: pd.DataFrame,
        text_col: str = "text"
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract all feature blocks from a DataFrame.
        
        Args:
            df: DataFrame with tweet data
            text_col: Column name containing tweet text
            
        Returns:
            (H, C, A) tuple:
                H: Text embeddings (N, 384)
                C: Metadata features (N, 11)
                A: Affect vectors (N, 15)
        """
        # Handle NaN values by replacing with empty string
        texts = df[text_col].fillna("").astype(str).tolist()
        
        # Text embeddings
        H = self.text_encoder.encode(texts)
        
        # Metadata (now includes cyclical hour encoding)
        C = extract_metadata(df)
        
        # Affect components
        polarity = self.sentiment_scorer.score(texts)
        
        if self.vad_scorer is not None:
            vad, coverage_stats = self.vad_scorer.score(texts)
            self._vad_coverage_stats = coverage_stats
        else:
            # Default to neutral if no lexicon
            vad = np.full((len(texts), 3), 0.5, dtype=np.float32)
            self._vad_coverage_stats = None
        
        emotions = self.emotion_classifier.predict(texts)
        
        # Assemble affect vector
        A = build_affect_vector(polarity, vad, emotions)
        
        return H, C, A
    
    def build_full_z(
        self,
        df: pd.DataFrame,
        text_col: str = "text"
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Build the complete feature matrix Z by concatenating [H; C; A].
        
        Args:
            df: DataFrame with tweet data
            text_col: Column name containing tweet text
            
        Returns:
            (H, C, A, Z) tuple where Z is (N, 410) concatenated features
        """
        H, C, A = self.build_features(df, text_col)
        Z = np.concatenate([H, C, A], axis=1)
        return H, C, A, Z