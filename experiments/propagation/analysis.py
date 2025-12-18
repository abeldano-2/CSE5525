"""
Feature importance analysis and result summarization.

Provides SHAP-based feature attribution for affect dimensions and utilities
for summarizing ablation and subspace experiment results.
"""

import numpy as np
import pandas as pd
import shap
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Any

from config import INDEX_SLICES, AFFECT_FEATURES


# =============================================================================
# SHAP Feature Importance
# =============================================================================

def run_shap(
    model: nn.Module,
    background_X: np.ndarray,
    explain_X: np.ndarray,
    device: str = None,
    max_evals: int = 500
) -> Any:
    """
    Run SHAP explainer on a trained model.
    
    Uses DeepExplainer for PyTorch models, falling back to KernelExplainer
    if issues arise.
    
    Args:
        model: Trained PyTorch model
        background_X: Background dataset for SHAP (subset of training data)
        explain_X: Samples to explain
        device: Device for model evaluation
        max_evals: Maximum evaluations for KernelExplainer
        
    Returns:
        SHAP Explanation object or array of SHAP values
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = model.to(device)
    model.eval()
    
    # Convert to tensors
    background_tensor = torch.from_numpy(background_X.astype(np.float32)).to(device)
    explain_tensor = torch.from_numpy(explain_X.astype(np.float32)).to(device)
    
    try:
        # Try DeepExplainer first (faster for neural networks)
        explainer = shap.DeepExplainer(model, background_tensor)
        shap_values = explainer.shap_values(explain_tensor)
    except Exception:
        # Fall back to KernelExplainer (model-agnostic)
        def model_predict(x):
            with torch.no_grad():
                t = torch.from_numpy(x.astype(np.float32)).to(device)
                return model(t).cpu().numpy()
        
        explainer = shap.KernelExplainer(model_predict, background_X)
        shap_values = explainer.shap_values(explain_X, nsamples=max_evals)
    
    return shap_values


def run_stratified_shap(
    model: nn.Module,
    X: np.ndarray,
    follower_col_idx: int,
    n_strata: int = 5,
    samples_per_stratum: int = 100,
    device: str = None
) -> Dict[str, Any]:
    """
    Run SHAP analysis stratified by follower count.
    
    This controls for metadata dominance by computing feature importance
    within bins of similar follower counts, revealing whether emotion matters
    more for low-reach vs high-reach accounts.
    
    Args:
        model: Trained PyTorch model
        X: Full feature matrix
        follower_col_idx: Column index for log_followers in X
        n_strata: Number of follower count bins
        samples_per_stratum: Samples to explain per stratum
        device: Device for model evaluation
        
    Returns:
        Dict with SHAP values and importance per stratum
    """
    import shap
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = model.to(device)
    model.eval()
    
    # Get follower values and compute quantile boundaries
    followers = X[:, follower_col_idx]
    quantiles = np.linspace(0, 1, n_strata + 1)
    boundaries = np.quantile(followers, quantiles)
    
    results = {
        "strata_boundaries": boundaries.tolist(),
        "per_stratum": [],
    }
    
    for i in range(n_strata):
        low, high = boundaries[i], boundaries[i + 1]
        mask = (followers >= low) & (followers <= high)
        X_stratum = X[mask]
        
        if len(X_stratum) < samples_per_stratum:
            continue
        
        # Sample from stratum
        idx = np.random.choice(len(X_stratum), min(samples_per_stratum, len(X_stratum)), replace=False)
        X_explain = X_stratum[idx]
        
        # Use full stratum as background
        X_bg = X_stratum[np.random.choice(len(X_stratum), min(200, len(X_stratum)), replace=False)]
        
        try:
            shap_values = run_shap(model, X_bg, X_explain, device=device)
            
            results["per_stratum"].append({
                "stratum_idx": i,
                "follower_range": (float(low), float(high)),
                "n_samples": len(X_explain),
                "shap_values": shap_values,
            })
        except Exception as e:
            results["per_stratum"].append({
                "stratum_idx": i,
                "error": str(e),
            })
    
    return results


def rank_affect_importance(
    shap_values: np.ndarray,
    affect_indices: Tuple[int, int] = None,
    affect_names: List[str] = None,
    output_idx: int = 0
) -> pd.DataFrame:
    """
    Extract and rank affect feature importance from SHAP values.
    
    Args:
        shap_values: SHAP values array, shape (N, D), (N, D, O), or list of arrays per output
        affect_indices: (start, end) indices for affect block (default from config)
        affect_names: Names for affect features (default from config)
        output_idx: Which output to analyze (0=retweets, 1=favorites)
        
    Returns:
        DataFrame with affect features ranked by mean |SHAP|
    """
    if affect_indices is None:
        affect_indices = INDEX_SLICES["affect"]
    if affect_names is None:
        affect_names = AFFECT_FEATURES
    
    # Handle different SHAP value formats
    if isinstance(shap_values, list):
        # List of arrays (one per output)
        sv = shap_values[output_idx]
    elif hasattr(shap_values, 'values'):
        # SHAP Explanation object
        sv = shap_values.values
        if sv.ndim == 3:
            sv = sv[:, :, output_idx]
    elif isinstance(shap_values, np.ndarray):
        if shap_values.ndim == 3:
            # 3D array: (N, D, O) - samples x features x outputs
            sv = shap_values[:, :, output_idx]
        else:
            sv = shap_values
    else:
        sv = np.array(shap_values)
    
    # Ensure 2D
    if sv.ndim != 2:
        raise ValueError(f"Expected 2D SHAP values, got shape {sv.shape}")
    
    start, end = affect_indices
    affect_shap = sv[:, start:end]  # (N, 15)
    
    # Compute mean absolute SHAP value per feature
    mean_abs_shap = np.mean(np.abs(affect_shap), axis=0)
    
    # Also compute mean signed SHAP (directional effect)
    mean_shap = np.mean(affect_shap, axis=0)
    
    # Build results DataFrame
    df = pd.DataFrame({
        "feature": affect_names[:len(mean_abs_shap)],
        "mean_abs_shap": mean_abs_shap,
        "mean_shap": mean_shap,
        "std_shap": np.std(affect_shap, axis=0)
    })
    
    # Sort by importance (descending)
    df = df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)
    
    return df[["rank", "feature", "mean_abs_shap", "mean_shap", "std_shap"]]


def compare_feature_blocks(
    shap_values: np.ndarray,
    output_idx: int = 0
) -> Dict[str, float]:
    """
    Compare total SHAP importance across feature blocks (text, metadata, affect).
    
    Note: Affect features may appear low importance
    when competing with dominant metadata signals like log_followers.
    Use stratified analysis for clearer interpretation.
    
    Args:
        shap_values: SHAP values array
        output_idx: Which output to analyze
        
    Returns:
        Dict with total importance per block
    """
    # Handle different SHAP value formats
    if isinstance(shap_values, list):
        sv = shap_values[output_idx]
    elif hasattr(shap_values, 'values'):
        sv = shap_values.values
        if sv.ndim == 3:
            sv = sv[:, :, output_idx]
    elif isinstance(shap_values, np.ndarray):
        if shap_values.ndim == 3:
            sv = shap_values[:, :, output_idx]
        else:
            sv = shap_values
    else:
        sv = np.array(shap_values)
    
    results = {}
    total_importance = np.sum(np.mean(np.abs(sv), axis=0))
    
    for block_name, (start, end) in INDEX_SLICES.items():
        block_shap = sv[:, start:end]
        block_importance = np.sum(np.mean(np.abs(block_shap), axis=0))
        results[f"{block_name}_importance"] = float(block_importance)
        results[f"{block_name}_pct"] = float(100 * block_importance / total_importance) if total_importance > 0 else 0
    
    return results


def conditional_affect_importance(
    model: nn.Module,
    X: np.ndarray,
    affect_indices: Tuple[int, int],
    device: str = None
) -> Dict[str, float]:
    """
    Compute affect importance conditional on metadata.
    
    Two-stage approach:
    1. Predict engagement from metadata only (baseline)
    2. Measure how much affect improves on the residual
    
    Args:
        model: Trained full model
        X: Feature matrix
        affect_indices: (start, end) for affect block
        device: Device for inference
        
    Returns:
        Dict with conditional importance metrics
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = model.to(device)
    model.eval()
    
    # Get predictions from full model
    with torch.no_grad():
        X_tensor = torch.from_numpy(X.astype(np.float32)).to(device)
        y_pred_full = model(X_tensor).cpu().numpy()
    
    # Create version with zeroed affect features
    X_no_affect = X.copy()
    aff_start, aff_end = affect_indices
    X_no_affect[:, aff_start:aff_end] = 0  # Zero out affect
    
    with torch.no_grad():
        X_tensor_no_aff = torch.from_numpy(X_no_affect.astype(np.float32)).to(device)
        y_pred_no_affect = model(X_tensor_no_aff).cpu().numpy()
    
    # Compare predictions
    diff = y_pred_full - y_pred_no_affect
    
    return {
        "mean_affect_contribution_retweets": float(np.mean(diff[:, 0])),
        "mean_affect_contribution_favorites": float(np.mean(diff[:, 1])),
        "std_affect_contribution_retweets": float(np.std(diff[:, 0])),
        "std_affect_contribution_favorites": float(np.std(diff[:, 1])),
        "mean_abs_affect_contribution": float(np.mean(np.abs(diff))),
    }


# =============================================================================
# Result Summarization
# =============================================================================

def summarize_ablation(
    mse_full: float,
    mse_text_meta: float,
    mse_shuffled: float
) -> Dict[str, Any]:
    """
    Summarize ablation experiment results.
    
    Core question: How much does explicit emotional information help?
    
    Args:
        mse_full: Test MSE of M_full (H + C + A)
        mse_text_meta: Test MSE of M_text+meta (H + C only)
        mse_shuffled: Test MSE of M_shuffled (H + C + A_permuted)
        
    Returns:
        Dict with summary metrics and interpretation
    """
    # Compute deltas (positive = ablated model is worse)
    delta_no_affect = mse_text_meta - mse_full
    delta_shuffled = mse_shuffled - mse_full
    
    # Relative improvements
    rel_improvement_affect = 100 * delta_no_affect / mse_text_meta if mse_text_meta > 0 else 0
    rel_improvement_align = 100 * delta_shuffled / mse_shuffled if mse_shuffled > 0 else 0
    
    # Interpretation
    affect_helps = delta_no_affect > 0
    alignment_matters = delta_shuffled > 0
    
    return {
        "metrics": {
            "mse_full": mse_full,
            "mse_text_meta": mse_text_meta,
            "mse_shuffled": mse_shuffled,
        },
        "deltas": {
            "delta_no_affect": delta_no_affect,
            "delta_shuffled": delta_shuffled,
        },
        "relative_improvement_pct": {
            "adding_affect": rel_improvement_affect,
            "proper_alignment": rel_improvement_align,
        },
        "interpretation": {
            "explicit_affect_helps": affect_helps,
            "text_affect_alignment_matters": alignment_matters,
        },
    }


def summarize_subspace(
    mse_affect_only: float,
    mse_non_affect: float,
    mse_full: float
) -> Dict[str, Any]:
    """
    Summarize subspace decomposition experiment results.
    
    Core question: How much engagement signal lives in affect vs non-affect
    directions of the embedding space?
    
    Args:
        mse_affect_only: Test MSE of M_affect_only (affect coords + metadata)
        mse_non_affect: Test MSE of M_non_affect (residual embedding + metadata)
        mse_full: Test MSE of M_full (reference)
        
    Returns:
        Dict with summary metrics and interpretation
    """
    # Which component captures more signal (lower MSE = better)
    affect_captures_more = mse_affect_only < mse_non_affect
    
    # How much worse than full model?
    gap_affect = mse_affect_only - mse_full
    gap_non_affect = mse_non_affect - mse_full
    
    # Relative to full model performance
    ratio_affect = mse_affect_only / mse_full if mse_full > 0 else float('inf')
    ratio_non_affect = mse_non_affect / mse_full if mse_full > 0 else float('inf')
    
    return {
        "metrics": {
            "mse_full": mse_full,
            "mse_affect_only": mse_affect_only,
            "mse_non_affect": mse_non_affect,
        },
        "gaps_from_full": {
            "gap_affect_only": gap_affect,
            "gap_non_affect": gap_non_affect,
        },
        "mse_ratios": {
            "affect_only_ratio": ratio_affect,
            "non_affect_ratio": ratio_non_affect,
        },
        "interpretation": {
            "affect_captures_more_signal": affect_captures_more,
            "components_are_complementary": gap_affect > 0 and gap_non_affect > 0,
        },
    }


# =============================================================================
# Report Generation (Raw Results Only)
# =============================================================================

def ablation_summary_text(ablation_results: Dict[str, Any]) -> str:
    """Generate raw ablation results text without interpretation."""
    m = ablation_results["metrics"]
    d = ablation_results["deltas"]
    rel = ablation_results["relative_improvement_pct"]
    
    lines = [
        "=== Ablation Experiment Results ===",
        f"M_full MSE:        {m['mse_full']:.6f}",
        f"M_text+meta MSE:   {m['mse_text_meta']:.6f}  (Δ = {d['delta_no_affect']:+.6f})",
        f"M_shuffled MSE:    {m['mse_shuffled']:.6f}  (Δ = {d['delta_shuffled']:+.6f})",
        "",
        f"Relative change (adding affect):     {rel['adding_affect']:+.4f}%",
        f"Relative change (proper alignment):  {rel['proper_alignment']:+.4f}%",
    ]
    
    return "\n".join(lines)


def subspace_summary_text(subspace_results: Dict[str, Any]) -> str:
    """Generate raw subspace results text without interpretation."""
    m = subspace_results["metrics"]
    g = subspace_results["gaps_from_full"]
    r = subspace_results["mse_ratios"]
    
    lines = [
        "=== Subspace Decomposition Results ===",
        f"M_full MSE:         {m['mse_full']:.6f}",
        f"M_affect_only MSE:  {m['mse_affect_only']:.6f}  (gap = {g['gap_affect_only']:+.6f}, ratio = {r['affect_only_ratio']:.4f})",
        f"M_non_affect MSE:   {m['mse_non_affect']:.6f}  (gap = {g['gap_non_affect']:+.6f}, ratio = {r['non_affect_ratio']:.4f})",
    ]
    
    return "\n".join(lines)


def generate_report(
    ablation_results: Dict[str, Any],
    subspace_results: Dict[str, Any],
    importance_df: pd.DataFrame = None,
    probe_summary: str = None,
    vad_coverage: Dict[str, Any] = None
) -> str:
    """
    Generate a markdown report of raw analysis results without interpretation.
    
    Args:
        ablation_results: Output from summarize_ablation()
        subspace_results: Output from summarize_subspace()
        importance_df: Output from rank_affect_importance() (optional)
        probe_summary: Output from AffectProbe.summary() (optional)
        vad_coverage: VAD lexicon coverage stats (optional)
        
    Returns:
        Markdown-formatted report string with raw results only
    """
    lines = [
        "# Emotion-Engagement Analysis Results",
        "",
    ]
    
    # Section 1: Ablation Results
    lines.extend([
        "## 1. Ablation Experiment Results",
        "",
        "| Model | MSE | Δ from Full |",
        "|-------|-----|-------------|",
    ])
    
    m = ablation_results["metrics"]
    d = ablation_results["deltas"]
    rel = ablation_results["relative_improvement_pct"]
    
    lines.append(f"| M_full | {m['mse_full']:.6f} | — |")
    lines.append(f"| M_text+meta | {m['mse_text_meta']:.6f} | {d['delta_no_affect']:+.6f} |")
    lines.append(f"| M_shuffled | {m['mse_shuffled']:.6f} | {d['delta_shuffled']:+.6f} |")
    
    lines.extend([
        "",
        f"Relative improvement (adding affect): {rel['adding_affect']:+.4f}%",
        "",
        f"Relative improvement (proper alignment): {rel['proper_alignment']:+.4f}%",
        "",
    ])
    
    # Section 2: Subspace Results
    lines.extend([
        "## 2. Subspace Decomposition Results",
        "",
        "| Model | MSE | Gap from Full | MSE Ratio |",
        "|-------|-----|---------------|-----------|",
    ])
    
    m2 = subspace_results["metrics"]
    g = subspace_results["gaps_from_full"]
    r = subspace_results["mse_ratios"]
    
    lines.append(f"| M_full | {m2['mse_full']:.6f} | — | 1.0000 |")
    lines.append(f"| M_affect_only | {m2['mse_affect_only']:.6f} | {g['gap_affect_only']:+.6f} | {r['affect_only_ratio']:.4f} |")
    lines.append(f"| M_non_affect | {m2['mse_non_affect']:.6f} | {g['gap_non_affect']:+.6f} | {r['non_affect_ratio']:.4f} |")
    lines.append("")
    
    # Section 3: Probe Linearity (if provided)
    section_num = 3
    if probe_summary:
        lines.extend([
            f"## {section_num}. Probe Linearity Results",
            "",
            "```",
            probe_summary,
            "```",
            "",
        ])
        section_num += 1
    
    # Section 4: VAD Coverage (if provided)
    if vad_coverage:
        lines.extend([
            f"## {section_num}. VAD Lexicon Coverage",
            "",
            f"Total tweets: {vad_coverage.get('total_tweets', 'N/A')}",
            "",
            f"Tweets with any VAD match: {vad_coverage.get('tweets_with_any_match', 'N/A')}",
            "",
            f"Min matches threshold: {vad_coverage.get('min_matches_threshold', 'N/A')}",
            "",
            f"Tweets with adequate coverage: {vad_coverage.get('tweets_with_adequate_coverage', 'N/A')}",
            "",
            f"Coverage fraction: {vad_coverage.get('coverage_fraction', 0):.6f}",
            "",
            f"Mean matches per tweet: {vad_coverage.get('mean_matches_per_tweet', 0):.4f}",
            "",
        ])
        section_num += 1
    
    # Section 5: Feature Importance (if provided)
    if importance_df is not None:
        lines.extend([
            f"## {section_num}. Affect Feature Importance (SHAP)",
            "",
            "| Rank | Feature | Mean |SHAP| | Mean SHAP | Std SHAP |",
            "|------|---------|-------------|-----------|----------|",
        ])
        
        for _, row in importance_df.iterrows():
            lines.append(
                f"| {row['rank']} | {row['feature']} | {row['mean_abs_shap']:.6f} | "
                f"{row['mean_shap']:+.6f} | {row['std_shap']:.6f} |"
            )
        
        lines.append("")
    
    return "\n".join(lines)