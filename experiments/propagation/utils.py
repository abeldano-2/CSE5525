"""
Utility functions for I/O, reproducibility, and device management.

Provides helpers for saving/loading model checkpoints, serializing feature
arrays, setting random seeds, and detecting compute devices.
"""

import random
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, Optional, Union

from config import RANDOM_SEED


# =============================================================================
# Device Management
# =============================================================================

def get_device(prefer_cuda: bool = True) -> torch.device:
    """
    Get the appropriate compute device.
    
    Args:
        prefer_cuda: Whether to prefer CUDA if available
        
    Returns:
        torch.device object
    """
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_device_info() -> Dict[str, Any]:
    """
    Get information about available compute devices.
    
    Returns:
        Dict with device availability and properties
    """
    info = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "current_device": str(get_device()),
    }
    
    if torch.cuda.is_available():
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
        info["cuda_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    return info


# =============================================================================
# Reproducibility
# =============================================================================

def set_seed(seed: int = RANDOM_SEED) -> None:
    """
    Set random seeds for reproducibility across Python, NumPy, and PyTorch.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # For complete reproducibility (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# =============================================================================
# Model Checkpointing
# =============================================================================

def save_checkpoint(
    model: torch.nn.Module,
    path: Union[str, Path],
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: Optional[int] = None,
    metrics: Optional[Dict[str, float]] = None,
    extra: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save a model checkpoint with optional training state.
    
    Args:
        model: PyTorch model to save
        path: Output file path (.pt or .pth)
        optimizer: Optional optimizer state to save
        epoch: Optional epoch number
        metrics: Optional performance metrics
        extra: Optional additional data to include
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_class": model.__class__.__name__,
    }
    
    # Store model architecture info if available
    if hasattr(model, "input_dim"):
        checkpoint["input_dim"] = model.input_dim
    if hasattr(model, "hidden_dims"):
        checkpoint["hidden_dims"] = model.hidden_dims
    
    # Optional components
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if epoch is not None:
        checkpoint["epoch"] = epoch
    if metrics is not None:
        checkpoint["metrics"] = metrics
    if extra is not None:
        checkpoint["extra"] = extra
    
    torch.save(checkpoint, path)


def load_checkpoint(
    path: Union[str, Path],
    model: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[str] = None
) -> Dict[str, Any]:
    """
    Load a model checkpoint.
    
    If model is provided, loads state dict into it. Otherwise returns the
    checkpoint dict for manual handling.
    
    Args:
        path: Path to checkpoint file
        model: Optional model to load weights into
        optimizer: Optional optimizer to load state into
        device: Device to map tensors to
        
    Returns:
        Checkpoint dict (model is modified in-place if provided)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    checkpoint = torch.load(path, map_location=device)
    
    if model is not None:
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
    
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    return checkpoint


# =============================================================================
# Array Serialization
# =============================================================================

def save_arrays(path: Union[str, Path], **arrays: np.ndarray) -> None:
    """
    Save multiple numpy arrays to a single .npz file.
    
    Args:
        path: Output file path (.npz)
        **arrays: Keyword arguments mapping names to arrays
        
    Example:
        save_arrays("features.npz", H=embeddings, C=metadata, A=affect)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


def load_arrays(path: Union[str, Path]) -> Dict[str, np.ndarray]:
    """
    Load arrays from a .npz file.
    
    Args:
        path: Path to .npz file
        
    Returns:
        Dict mapping array names to numpy arrays
    """
    data = np.load(path)
    return {key: data[key] for key in data.files}


def save_probe_weights(
    path: Union[str, Path],
    W_affect: np.ndarray,
    b: np.ndarray,
    metadata: Optional[Dict] = None
) -> None:
    """
    Save affect probe weights for later use.
    
    Args:
        path: Output file path (.npz)
        W_affect: Weight matrix (K, 384)
        b: Bias vector (K,)
        metadata: Optional metadata dict (e.g., alpha, training info)
    """
    save_dict = {"W_affect": W_affect, "b": b}
    if metadata is not None:
        # Store metadata as a pickled object
        save_dict["metadata"] = np.array([metadata], dtype=object)
    
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **save_dict)


def load_probe_weights(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load saved probe weights.
    
    Args:
        path: Path to saved weights
        
    Returns:
        Dict with 'W_affect', 'b', and optionally 'metadata'
    """
    data = np.load(path, allow_pickle=True)
    result = {
        "W_affect": data["W_affect"],
        "b": data["b"]
    }
    if "metadata" in data.files:
        result["metadata"] = data["metadata"][0]
    return result


# =============================================================================
# Logging Helpers
# =============================================================================

def format_metrics(metrics: Dict[str, float], precision: int = 4) -> str:
    """
    Format a metrics dict as a readable string.
    
    Args:
        metrics: Dict of metric names to values
        precision: Decimal places for floats
        
    Returns:
        Formatted string
    """
    parts = []
    for key, value in metrics.items():
        if isinstance(value, float):
            parts.append(f"{key}={value:.{precision}f}")
        else:
            parts.append(f"{key}={value}")
    return " | ".join(parts)


def print_section(title: str, width: int = 60) -> None:
    """Print a formatted section header."""
    print("\n" + "=" * width)
    print(f" {title}")
    print("=" * width)


# =============================================================================
# Path Utilities
# =============================================================================

def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_project_root() -> Path:
    """
    Get the project root directory (where this utils.py lives).
    
    Returns:
        Path to project root
    """
    return Path(__file__).parent.resolve()