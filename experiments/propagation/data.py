"""
Data loading, preprocessing, and PyTorch dataset utilities.

Handles tweet data I/O, train/val/test splitting, log-transforms for targets,
and DataLoader construction.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, Optional

from config import SPLIT_RATIOS, RANDOM_SEED


# =============================================================================
# Data Loading
# =============================================================================

def load_tweets(path: str) -> pd.DataFrame:
    """
    Load raw tweet data from a CSV or parquet file.
    
    Expected columns (at minimum):
        - text: raw tweet text
        - retweet_count: integer
        - favorite_count: integer
        - followers_count, friends_count, statuses_count: user metrics
        - user_verified: boolean
        - created_at: timestamp (for hour extraction)
    
    Args:
        path: Path to data file (.csv, .parquet, or .json)
        
    Returns:
        DataFrame with tweet data
    """
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    elif path.endswith(".json") or path.endswith(".jsonl"):
        df = pd.read_json(path, lines=path.endswith(".jsonl"))
    else:
        df = pd.read_csv(path)
    
    return df


# =============================================================================
# Data Splitting
# =============================================================================

def split_data(
    df: pd.DataFrame,
    ratios: Dict[str, float] = SPLIT_RATIOS,
    seed: int = RANDOM_SEED
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train/val/test sets.
    
    Args:
        df: Input DataFrame
        ratios: Dict with 'train', 'val', 'test' keys summing to 1.0 (default: SPLIT_RATIOS)
        seed: Random seed for reproducibility (default: RANDOM_SEED)
        
    Returns:
        Tuple of (train_df, val_df, test_df) DataFrames
    """
    # First split: separate test set
    test_size = ratios["test"]
    train_val, test = train_test_split(
        df, test_size=test_size, random_state=seed
    )
    
    # Second split: separate train and val from remaining
    val_ratio_adjusted = ratios["val"] / (ratios["train"] + ratios["val"])
    train, val = train_test_split(
        train_val, test_size=val_ratio_adjusted, random_state=seed
    )
    
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


# =============================================================================
# Target Transforms
# =============================================================================

def log_transform(y: np.ndarray) -> np.ndarray:
    """Apply log(1 + y) transform to engagement counts."""
    return np.log1p(y)


def inverse_log_transform(t: np.ndarray) -> np.ndarray:
    """Inverse of log_transform: exp(t) - 1."""
    return np.expm1(t)


def prepare_targets(df: pd.DataFrame) -> np.ndarray:
    """
    Extract and log-transform engagement targets.
    
    Args:
        df: DataFrame with 'retweet_count' and 'favorite_count' columns
        
    Returns:
        Array of shape (N, 2) with [log_retweets, log_favorites]
    """
    # Use pd.to_numeric to handle invalid/non-numeric values (e.g., duplicate header rows)
    retweets = pd.to_numeric(df["retweet_count"], errors='coerce').fillna(0).values.astype(np.float32)
    favorites = pd.to_numeric(df["favorite_count"], errors='coerce').fillna(0).values.astype(np.float32)
    
    y = np.stack([log_transform(retweets), log_transform(favorites)], axis=1)
    return y


# =============================================================================
# Feature Scaling
# =============================================================================

class FeatureScaler:
    """
    Wrapper for standardizing metadata and affect features.
    
    Text embeddings are left unscaled (as produced by the encoder).
    Metadata and affect blocks are standardized to zero mean, unit variance.
    """
    
    def __init__(self, metadata_slice: Tuple[int, int], affect_slice: Tuple[int, int]):
        """
        Args:
            metadata_slice: (start, end) indices for metadata block
            affect_slice: (start, end) indices for affect block
        """
        self.metadata_slice = metadata_slice
        self.affect_slice = affect_slice
        self.metadata_scaler = StandardScaler()  # Zeros mean, unit variance
        self.affect_scaler = StandardScaler()    # Zeros mean, unit variance
        self._fitted = False                     # Flag to check if scalers have been fit
    
    def fit(self, Z: np.ndarray) -> "FeatureScaler":
        """
        Fit scalers on training data.
        
        Args:
            Z: Feature matrix of shape (N, D) where D is input dimension
        
        Returns:
            Self
        """
        meta_start, meta_end = self.metadata_slice
        aff_start, aff_end = self.affect_slice
        
        self.metadata_scaler.fit(Z[:, meta_start:meta_end])
        self.affect_scaler.fit(Z[:, aff_start:aff_end])
        self._fitted = True
        return self
    
    def transform(self, Z: np.ndarray) -> np.ndarray:
        """
        Apply fitted scalers, returning a copy with scaled blocks.
        
        Args:
            Z: Feature matrix of shape (N, D) where D is input dimension
        
        Returns:
            Feature matrix with scaled metadata and affect blocks
        """
        if not self._fitted:
            raise RuntimeError("Scaler must be fit before transform")
        
        Z_scaled = Z.copy()
        meta_start, meta_end = self.metadata_slice
        aff_start, aff_end = self.affect_slice
        
        Z_scaled[:, meta_start:meta_end] = self.metadata_scaler.transform(
            Z[:, meta_start:meta_end]
        )
        Z_scaled[:, aff_start:aff_end] = self.affect_scaler.transform(
            Z[:, aff_start:aff_end]
        )
        return Z_scaled
    
    def fit_transform(self, Z: np.ndarray) -> np.ndarray:
        """Fit and transform in one call."""
        return self.fit(Z).transform(Z)


# =============================================================================
# PyTorch Dataset
# =============================================================================

class EngagementDataset(Dataset):
    """
    PyTorch Dataset for engagement prediction.
    
    Holds feature matrix Z and target matrix Y, returning (z_i, y_i) tuples.
    """
    
    def __init__(self, Z: np.ndarray, Y: np.ndarray):
        """
        Args:
            Z: Feature matrix of shape (N, D) where D is input dimension
            Y: Target matrix of shape (N, 2) with [log_retweets, log_favorites]
        """
        self.Z = torch.from_numpy(Z.astype(np.float32))
        self.Y = torch.from_numpy(Y.astype(np.float32))
        
        assert len(self.Z) == len(self.Y), "Z and Y must have same length"
    
    def __len__(self) -> int:
        return len(self.Z)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.Z[idx], self.Y[idx]


# =============================================================================
# DataLoader Construction
# =============================================================================

def build_dataloaders(
    Z_train: np.ndarray,
    Y_train: np.ndarray,
    Z_val: np.ndarray,
    Y_val: np.ndarray,
    Z_test: np.ndarray,
    Y_test: np.ndarray,
    batch_size: int = 512,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Construct DataLoaders for train/val/test sets.
    
    Args:
        Z_train, Y_train: Training features and targets
        Z_val, Y_val: Validation features and targets
        Z_test, Y_test: Test features and targets
        batch_size: Batch size for all loaders
        num_workers: Number of worker processes for data loading
        
    Returns:
        (train_loader, val_loader, test_loader) tuple
    """
    train_ds = EngagementDataset(Z_train, Y_train)
    val_ds = EngagementDataset(Z_val, Y_val)
    test_ds = EngagementDataset(Z_test, Y_test)
    
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader