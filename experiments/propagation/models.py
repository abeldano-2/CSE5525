"""
Model definitions for engagement prediction.

Includes the MLP architecture, affect probe (ridge regression), model variant
enumeration, and factory functions for constructing correctly-sized models.
"""

import numpy as np
import torch
import torch.nn as nn
from enum import Enum
from typing import Tuple, Optional, List
from sklearn.linear_model import Ridge

from config import (
    MODEL_INPUT_DIMS,
    TRAINING_DEFAULTS,
    PROBE_DEFAULTS,
    PROBE_R2_THRESHOLD
)


# =============================================================================
# Model Variants
# =============================================================================

class ModelVariant(Enum):
    """
    Enumeration of engagement model variants.

    Notational convention:
    - H: Text embeddings
    - C: Metadata
    - A: Affect vector (explicit affect)
    - A_shuffled: Shuffled affect vector (affect misaligned wrt text)
    - H_aff: Affect subspace embeddings (implicit affect)
    - H_non: Non-affect subspace embeddings

    """
    FULL = "full"                # H + C + A          - baseline model (explicit affect)
    TEXT_META = "text_meta"      # H + C              - no explicit affect
    SHUFFLED = "shuffled"        # H + C + A_shuffled - control experiment
    AFFECT_ONLY = "affect_only"  # H_aff + C          - affect subspace coords only (implicit affect)
    NON_AFFECT = "non_affect"    # H_non + C          - non-affect subspace coords only


# =============================================================================
# Engagement MLP
# =============================================================================

class EngagementMLP(nn.Module):
    """
    N-hidden-layer MLP for engagement prediction. N should be between 2 and 4.
    Hidden layer sizes are specified in the config.

    Architecture: input -> hidden1 (ReLU) -> ... -> hiddenN (ReLU) -> 2 outputs
    
    Outputs predict log(1 + retweets) and log(1 + favorites).
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Tuple[int, int] = None,
        dropout: float = None
    ):
        """
        Args:
            input_dim: Dimension of input feature vector
            hidden_dims: Tuple of (hidden1, hidden2) sizes (default: 512, 256)
            dropout: Dropout probability between layers (default: 0.1)
        """
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = TRAINING_DEFAULTS["hidden_dims"]
        if dropout is None:
            dropout = TRAINING_DEFAULTS["dropout"]
        
        h1, h2 = hidden_dims
        
        # Build network layers
        self.net = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h2, 2)  # 2 outputs: log_retweets, log_favorites
        )
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Xavier initialization for linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            z: Input tensor of shape (batch, input_dim)
            
        Returns:
            Predictions of shape (batch, 2) for [log_retweets, log_favorites]
        """
        return self.net(z)
    
    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# Model Factory (generates model variants)
# =============================================================================

def build_engagement_model(
    variant: ModelVariant,
    hidden_dims: Tuple[int, int] = None,
    dropout: float = None,
    input_dim_override: int = None
) -> EngagementMLP:
    """
    Factory function to create an EngagementMLP with correct input dimension.
    
    Args:
        variant: ModelVariant specifying which input configuration to use
        hidden_dims: Optional override for hidden layer sizes
        dropout: Optional override for dropout rate
        input_dim_override: Optional override for input dimension (useful when
                            subspace dimension varies due to linearity filtering)
        
    Returns:
        Configured EngagementMLP instance
    """
    if input_dim_override is not None:
        input_dim = input_dim_override
    else:
        input_dim = MODEL_INPUT_DIMS[variant.value]
    return EngagementMLP(input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout)


# =============================================================================
# Affect Shuffling (for control experiment)
# =============================================================================

def shuffle_affect(A: np.ndarray, seed: int = None) -> np.ndarray:
    """
    Row-permute the affect matrix to break text-affect alignment.
    
    Used for the M_shuffled control experiment to test whether the model
    genuinely uses text-emotion alignment or treats affect as noise.
    
    Args:
        A: Affect matrix of shape (N, K)
        seed: Random seed for reproducibility
        
    Returns:
        Permuted affect matrix A_tilde with same shape
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(A))
    return A[idx].copy()


# =============================================================================
# Affect Probe (Ridge Regression)
# =============================================================================

class AffectProbe:
    """
    Linear probe from text embeddings to affect vectors.
    
    Fits multi-output ridge regression: A_hat = H @ W_affect.T + b
    The weight matrix W_affect (K x 384) defines directions in embedding space
    corresponding to each affect dimension, spanning the "affect subspace".
    
    Linearity Check:
        After fitting, computes R^2 for each affect dimension. Dimensions with
        R^2 below a threshold are flagged as poorly linearly separable in the
        embedding space and can be excluded from the subspace decomposition.
    """
    
    def __init__(
        self,
        alpha: float = None,
        fit_intercept: bool = None,
        r2_threshold: float = None
    ):
        """
        Args:
            alpha: Ridge regularization strength (default from config)
            fit_intercept: Whether to fit bias term (default: True)
            r2_threshold: Minimum R^2 for including dimension (default from config)
        """
        if alpha is None:
            alpha = PROBE_DEFAULTS["alpha"]
        if fit_intercept is None:
            fit_intercept = PROBE_DEFAULTS["fit_intercept"]
        if r2_threshold is None:
            r2_threshold = PROBE_R2_THRESHOLD
        
        self.model = Ridge(alpha=alpha, fit_intercept=fit_intercept)
        self.alpha = alpha
        self.r2_threshold = r2_threshold
        self._fitted = False
        
        # Will be set after fitting
        self.W_affect: Optional[np.ndarray] = None  # Shape (K, 384)
        self.b: Optional[np.ndarray] = None         # Shape (K,)
        
        # Per-dimension R^2 scores
        self.r2_per_dimension: Optional[np.ndarray] = None
        self.valid_dimensions: Optional[np.ndarray] = None  # Boolean mask
    
    def fit(self, H: np.ndarray, A: np.ndarray) -> "AffectProbe":
        """
        Fit the probe to predict affect from embeddings.
        
        Args:
            H: Text embeddings of shape (N, 384)
            A: Affect vectors of shape (N, K)
            
        Returns:
            Self for method chaining
        """
        self.model.fit(H, A)
        
        # Extract learned parameters
        # sklearn Ridge with multi-output: coef_ has shape (K, 384)
        self.W_affect = self.model.coef_.astype(np.float32)
        self.b = self.model.intercept_.astype(np.float32)
        
        # Compute per-dimension R^2
        A_pred = self.model.predict(H)
        K = A.shape[1]
        self.r2_per_dimension = np.zeros(K)
        
        for k in range(K):
            ss_res = np.sum((A[:, k] - A_pred[:, k]) ** 2)
            ss_tot = np.sum((A[:, k] - np.mean(A[:, k])) ** 2)
            self.r2_per_dimension[k] = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        
        # Flag dimensions with sufficient R^2
        self.valid_dimensions = self.r2_per_dimension >= self.r2_threshold
        
        self._fitted = True
        return self
    
    def predict(self, H: np.ndarray) -> np.ndarray:
        """
        Predict affect vectors from embeddings.
        
        Args:
            H: Text embeddings of shape (N, 384)
            
        Returns:
            Predicted affect of shape (N, K)
        """
        if not self._fitted:
            raise RuntimeError("Probe must be fit before predict")
        return self.model.predict(H).astype(np.float32)
    
    def get_weights(self, valid_only: bool = False) -> np.ndarray:
        """
        Return the learned weight matrix W_affect.
        
        Each row i of W_affect is the direction in embedding space most
        correlated with affect dimension i.
        
        Args:
            valid_only: If True, return only rows for dimensions with R^2 >= threshold
        
        Returns:
            Weight matrix of shape (K, 384) or (K_valid, 384)
        """
        if not self._fitted:
            raise RuntimeError("Probe must be fit first")
        
        if valid_only and self.valid_dimensions is not None:
            return self.W_affect[self.valid_dimensions]
        return self.W_affect
    
    def get_valid_dimension_mask(self) -> np.ndarray:
        """
        Return boolean mask indicating which dimensions passed the linearity check.
        
        Returns:
            Boolean array of shape (K,) where True = R^2 >= threshold
        """
        if not self._fitted:
            raise RuntimeError("Probe must be fit first")
        return self.valid_dimensions
    
    def get_r2_scores(self) -> np.ndarray:
        """
        Return per-dimension R^2 scores from the linearity check.
        
        Returns:
            Array of shape (K,) with R^2 for each affect dimension
        """
        if not self._fitted:
            raise RuntimeError("Probe must be fit first")
        return self.r2_per_dimension
    
    def score(self, H: np.ndarray, A: np.ndarray) -> float:
        """
        Compute R^2 score on given data.
        
        Args:
            H: Text embeddings of shape (N, 384)
            A: True affect vectors of shape (N, K)
            
        Returns:
            R^2 coefficient of determination (averaged across outputs)
        """
        if not self._fitted:
            raise RuntimeError("Probe must be fit first")
        return self.model.score(H, A)
    
    def summary(self, affect_names: List[str] = None) -> str:
        """
        Generate a summary of the probe fit quality per dimension.
        
        Args:
            affect_names: Optional list of affect dimension names
            
        Returns:
            Formatted string summary
        """
        if not self._fitted:
            raise RuntimeError("Probe must be fit first")
        
        K = len(self.r2_per_dimension)
        if affect_names is None:
            affect_names = [f"dim_{i}" for i in range(K)]
        
        lines = ["Affect Probe Linearity Check (R^2 per dimension):"]
        lines.append("-" * 50)
        
        for i, (name, r2, valid) in enumerate(zip(
            affect_names, self.r2_per_dimension, self.valid_dimensions
        )):
            status = "✓" if valid else "✗"
            lines.append(f"  {status} {name:20s}: R^2 = {r2:.4f}")
        
        n_valid = np.sum(self.valid_dimensions)
        lines.append("-" * 50)
        lines.append(f"Valid dimensions: {n_valid}/{K} (threshold: R^2 >= {self.r2_threshold})")
        
        return "\n".join(lines)


# =============================================================================
# Feature Assembly Utilities (for model variants)
# =============================================================================

def assemble_full_input(H: np.ndarray, C: np.ndarray, A: np.ndarray) -> np.ndarray:
    """
    Assemble input for M_full: [H; C; A].
    
    Args:
        H: Text embeddings (N, 384)
        C: Metadata (N, 10)
        A: Affect vectors (N, 15)
        
    Returns:
        Z of shape (N, 409)
    """
    return np.concatenate([H, C, A], axis=1)


def assemble_text_meta_input(H: np.ndarray, C: np.ndarray) -> np.ndarray:
    """
    Assemble input for M_text_meta: [H; C].
    
    Args:
        H: Text embeddings (N, 384)
        C: Metadata (N, 10)
        
    Returns:
        Z of shape (N, 394)
    """
    return np.concatenate([H, C], axis=1)


def assemble_shuffled_input(
    H: np.ndarray, 
    C: np.ndarray, 
    A: np.ndarray, 
    seed: int = None
) -> np.ndarray:
    """
    Assemble input for M_shuffled: [H; C; A_shuffled].
    
    Args:
        H: Text embeddings (N, 384)
        C: Metadata (N, 10)
        A: Affect vectors (N, 15)
        seed: Random seed for shuffling
        
    Returns:
        Z of shape (N, 409) with permuted affect
    """
    A_shuffled = shuffle_affect(A, seed=seed)
    return np.concatenate([H, C, A_shuffled], axis=1)