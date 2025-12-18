"""
Training loop and evaluation utilities for engagement models.

Provides a Trainer class handling the train/validation loop with early stopping,
plus evaluation metrics (MSE, R^2) and comparison utilities across model variants.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import time

from config import TRAINING_DEFAULTS


# =============================================================================
# Trainer Class
# =============================================================================

class Trainer:
    """
    Handles the training loop for engagement prediction models.
    
    Features:
        - Train/validation epoch execution
        - Early stopping with patience
        - Training history tracking
        - Best model checkpoint restoration
    """
    
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module = None,
        optimizer: torch.optim.Optimizer = None,
        device: str = None,
        lr: float = None,
        weight_decay: float = None
    ):
        """
        Args:
            model: PyTorch model to train
            criterion: Loss function (default: MSELoss)
            optimizer: Optimizer (default: Adam with config defaults)
            device: 'cuda' or 'cpu' (auto-detected if None)
            lr: Learning rate override
            weight_decay: Weight decay override
        """
        # Device setup
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        
        # Model
        self.model = model.to(self.device)
        
        # Loss function
        self.criterion = criterion if criterion is not None else nn.MSELoss()
        
        # Optimizer
        if optimizer is None:
            lr = lr if lr is not None else TRAINING_DEFAULTS["lr"]
            weight_decay = weight_decay if weight_decay is not None else TRAINING_DEFAULTS["weight_decay"]
            optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.optimizer = optimizer
        
        # Training state
        self.history: Dict[str, List[float]] = defaultdict(list)
        self.best_val_loss = float("inf")
        self.best_model_state = None
        self.epochs_trained = 0
    
    def train_epoch(self, loader: DataLoader) -> float:
        """
        Execute one training epoch.
        
        Args:
            loader: Training DataLoader
            
        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        
        for z_batch, y_batch in loader:
            z_batch = z_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            y_pred = self.model(z_batch)
            loss = self.criterion(y_pred, y_batch)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        return total_loss / n_batches
    
    @torch.no_grad()
    def validate(self, loader: DataLoader) -> float:
        """
        Evaluate model on validation set.
        
        Args:
            loader: Validation DataLoader
            
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        
        for z_batch, y_batch in loader:
            z_batch = z_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            y_pred = self.model(z_batch)
            loss = self.criterion(y_pred, y_batch)
            
            total_loss += loss.item()
            n_batches += 1
        
        return total_loss / n_batches
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = None,
        patience: int = None,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Full training loop with early stopping.
        
        Args:
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            epochs: Maximum number of epochs (default from config)
            patience: Early stopping patience (default from config)
            verbose: Whether to print progress
            
        Returns:
            Training history dict with 'train_loss' and 'val_loss' lists
        """
        if epochs is None:
            epochs = TRAINING_DEFAULTS["epochs"]
        if patience is None:
            patience = TRAINING_DEFAULTS["patience"]
        
        epochs_without_improvement = 0
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # Training
            train_loss = self.train_epoch(train_loader)
            
            # Validation
            val_loss = self.validate(val_loader)
            
            # Record history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.epochs_trained += 1
            
            # Check for improvement
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            
            # Logging
            if verbose:
                elapsed = time.time() - start_time
                print(f"Epoch {epoch+1:3d}/{epochs} | "
                      f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                      f"Time: {elapsed:.1f}s" +
                      (" *" if epochs_without_improvement == 0 else ""))
            
            # Early stopping check
            if epochs_without_improvement >= patience:
                if verbose:
                    print(f"Early stopping after {epoch+1} epochs (patience={patience})")
                break
        
        # Restore best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            self.model.to(self.device)
        
        return dict(self.history)
    
    def get_best_val_loss(self) -> float:
        """Return the best validation loss achieved during training."""
        return self.best_val_loss


# =============================================================================
# Evaluation Functions
# =============================================================================

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: str = None
) -> Tuple[float, float, float]:
    """
    Evaluate model and compute MSE and R^2 metrics.
    
    Args:
        model: Trained model
        loader: Test DataLoader
        device: Device for evaluation
        
    Returns:
        Tuple of (mse, r2_retweets, r2_favorites)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    
    model = model.to(device)
    model.eval()
    
    all_preds = []
    all_targets = []
    
    for z_batch, y_batch in loader:
        z_batch = z_batch.to(device)
        y_pred = model(z_batch).cpu().numpy()
        all_preds.append(y_pred)
        all_targets.append(y_batch.numpy())
    
    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    
    # MSE (averaged over both outputs)
    mse = float(np.mean((preds - targets) ** 2))
    
    # R^2 for each output
    r2_rt = compute_r2(targets[:, 0], preds[:, 0])
    r2_fav = compute_r2(targets[:, 1], preds[:, 1])
    
    return mse, r2_rt, r2_fav


def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute coefficient of determination (R^2).
    
    R^2 = 1 - SS_res / SS_tot
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        
    Returns:
        R^2 score (can be negative for poor models)
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    if ss_tot == 0:
        return 0.0
    
    return float(1 - ss_res / ss_tot)


def compute_mse_per_output(
    model: nn.Module,
    loader: DataLoader,
    device: str = None
) -> Tuple[float, float]:
    """
    Compute MSE separately for retweets and favorites.
    
    Args:
        model: Trained model
        loader: DataLoader
        device: Device for evaluation
        
    Returns:
        (mse_retweets, mse_favorites) tuple
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    
    model = model.to(device)
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for z_batch, y_batch in loader:
            z_batch = z_batch.to(device)
            y_pred = model(z_batch).cpu().numpy()
            all_preds.append(y_pred)
            all_targets.append(y_batch.numpy())
    
    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    
    mse_rt = float(np.mean((preds[:, 0] - targets[:, 0]) ** 2))
    mse_fav = float(np.mean((preds[:, 1] - targets[:, 1]) ** 2))
    
    return mse_rt, mse_fav


# =============================================================================
# Model Comparison
# =============================================================================

def compare_variants(results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """
    Create a comparison table of model variant performance.
    
    Args:
        results: Dict mapping model name to metrics dict with keys:
                 'mse', 'r2_retweets', 'r2_favorites'
                 
    Returns:
        DataFrame with models as rows and metrics as columns
    """
    rows = []
    for model_name, metrics in results.items():
        rows.append({
            "Model": model_name,
            "MSE": metrics.get("mse", np.nan),
            "R^2 (Retweets)": metrics.get("r2_retweets", np.nan),
            "R^2 (Favorites)": metrics.get("r2_favorites", np.nan),
        })
    
    df = pd.DataFrame(rows)
    df = df.set_index("Model")
    
    return df


def compute_ablation_deltas(
    full_mse: float,
    text_meta_mse: float,
    shuffled_mse: float
) -> Dict[str, float]:
    """
    Compute performance deltas from ablation experiments.
    
    Positive delta means the ablated model performs worse (higher MSE),
    indicating the removed feature was helpful.
    
    Args:
        full_mse: MSE of M_full
        text_meta_mse: MSE of M_text+meta (no affect)
        shuffled_mse: MSE of M_shuffled (permuted affect)
        
    Returns:
        Dict with delta values and interpretations
    """
    # Delta = ablated_mse - full_mse
    # Positive means removing the feature hurts performance
    delta_no_affect = text_meta_mse - full_mse
    delta_shuffled = shuffled_mse - full_mse
    
    # Relative change (%)
    rel_delta_no_affect = 100 * delta_no_affect / full_mse if full_mse > 0 else 0
    rel_delta_shuffled = 100 * delta_shuffled / full_mse if full_mse > 0 else 0
    
    return {
        "full_mse": full_mse,
        "text_meta_mse": text_meta_mse,
        "shuffled_mse": shuffled_mse,
        "delta_no_affect": delta_no_affect,
        "delta_shuffled": delta_shuffled,
        "relative_delta_no_affect_pct": rel_delta_no_affect,
        "relative_delta_shuffled_pct": rel_delta_shuffled,
        "affect_helps": delta_no_affect > 0,
        "alignment_matters": delta_shuffled > 0
    }


def compute_subspace_deltas(
    affect_only_mse: float,
    non_affect_mse: float,
    full_mse: float
) -> Dict[str, float]:
    """
    Compute performance comparisons from subspace experiments.
    
    Addresses: How much engagement signal is in affect vs non-affect directions?
    
    Args:
        affect_only_mse: MSE of M_affect_only (affect coords + metadata)
        non_affect_mse: MSE of M_non_affect (residual embedding + metadata)
        full_mse: MSE of M_full (reference)
        
    Returns:
        Dict with comparison metrics
    """
    return {
        "full_mse": full_mse,
        "affect_only_mse": affect_only_mse,
        "non_affect_mse": non_affect_mse,
        "delta_affect_only": affect_only_mse - full_mse,
        "delta_non_affect": non_affect_mse - full_mse,
        "affect_captures_more": affect_only_mse < non_affect_mse,
        "affect_ratio": affect_only_mse / full_mse if full_mse > 0 else np.nan,
        "non_affect_ratio": non_affect_mse / full_mse if full_mse > 0 else np.nan
    }