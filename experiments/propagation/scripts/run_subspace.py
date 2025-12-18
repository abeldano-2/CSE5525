#!/usr/bin/env python3
"""
Focused subspace decomposition experiment script.

Runs the subspace analysis (M_affect_only vs M_non_affect vs M_full)
without the ablation comparison or SHAP. Useful for analyzing how much
engagement signal lives in emotional vs non-emotional embedding directions.

Usage:
    python run_subspace.py --features features_train.npz --output subspace_results/
    python run_subspace.py --data tweets.csv --quick  # Subsample for fast testing

Disclaimer: This runner script was written with the help of Cursor.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config import TRAINING_DEFAULTS, RANDOM_SEED, AFFECT_FEATURES
from data import (
    load_tweets, prepare_targets, build_dataloaders
)
from features import FeatureBuilder
from models import (
    ModelVariant, build_engagement_model, AffectProbe,
    assemble_full_input
)
from subspace import (
    AffectSubspace, prepare_affect_only_input, prepare_non_affect_input
)
from training import Trainer, evaluate
from analysis import summarize_subspace, subspace_summary_text
from utils import (
    set_seed, get_device, save_checkpoint, load_arrays,
    save_probe_weights, ensure_dir, print_section, format_metrics
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run subspace decomposition experiments for affect vs non-affect signal"
    )
    
    # Input options (either raw data or pre-extracted features)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--data", type=str,
        help="Path to raw tweet data (will extract features)"
    )
    input_group.add_argument(
        "--features", type=str,
        help="Path to pre-extracted features (.npz file)"
    )
    
    parser.add_argument(
        "--output", type=str, default="subspace_results",
        help="Output directory"
    )
    parser.add_argument(
        "--batch-size", type=int, default=TRAINING_DEFAULTS["batch_size"]
    )
    parser.add_argument(
        "--epochs", type=int, default=TRAINING_DEFAULTS["epochs"]
    )
    parser.add_argument(
        "--seed", type=int, default=RANDOM_SEED
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode: subsample data for fast iteration"
    )
    parser.add_argument(
        "--subsample", type=int, default=5000,
        help="Number of samples to use in quick mode"
    )
    parser.add_argument(
        "--device", type=str, default=None
    )
    parser.add_argument(
        "--r2-threshold", type=float, default=0.05,
        help="R^2 threshold for including affect dimensions in subspace (default: 0.05)"
    )
    parser.add_argument(
        "--no-filter", action="store_true",
        help="Use all affect dimensions regardless of linearity check"
    )
    
    return parser.parse_args()


def load_or_extract_features(args):
    """Load pre-extracted features or extract from raw data."""
    
    if args.features:
        # Load pre-extracted features
        print(f"Loading features from {args.features}...")
        data = load_arrays(args.features)
        
        # Expect H, C, A, Y in the file
        H = data["H"]
        C = data["C"]
        A = data["A"]
        Y = data["Y"]
        
        return H, C, A, Y
    
    else:
        # Extract from raw data
        print(f"Loading raw data from {args.data}...")
        df = load_tweets(args.data)
        
        if args.quick:
            print(f"Quick mode: subsampling to {args.subsample} tweets")
            df = df.sample(n=min(args.subsample, len(df)), random_state=args.seed)
        
        print(f"Loaded {len(df)} tweets")
        
        print("Extracting features...")
        builder = FeatureBuilder(device=args.device)
        H, C, A, Z = builder.build_full_z(df)
        Y = prepare_targets(df)
        
        return H, C, A, Y


def run_subspace_experiment(
    H_train, C_train, A_train, Y_train,
    H_val, C_val, A_val, Y_val,
    H_test, C_test, A_test, Y_test,
    args,
    output_dir
):
    """
    Run the subspace decomposition experiment.
    
    Compares:
        - M_full: Full model (H + C + A) as reference
        - M_affect_only: Affect coordinates + metadata
        - M_non_affect: Residual embedding + metadata
    """
    device = args.device or str(get_device())
    results = {}
    
    # -------------------------------------------------------------------------
    # Step 1: Fit Affect Probe
    # -------------------------------------------------------------------------
    print_section("Step 1: Fitting Affect Probe")
    
    probe = AffectProbe(r2_threshold=args.r2_threshold)
    probe.fit(H_train, A_train)
    
    probe_r2 = probe.score(H_test, A_test)
    print(f"Probe R^2 on test set: {probe_r2:.4f}")
    
    # Per-dimension R^2 scores
    probe_summary = probe.summary(AFFECT_FEATURES)
    print("\n" + probe_summary)
    
    # Determine which dimensions to use
    if args.no_filter:
        valid_mask = None
        n_valid = len(AFFECT_FEATURES)
        print(f"\nUsing all {n_valid} affect dimensions (--no-filter)")
    else:
        valid_mask = probe.get_valid_dimension_mask()
        n_valid = np.sum(valid_mask)
        print(f"\nUsing {n_valid}/{len(AFFECT_FEATURES)} dimensions (R^2 >= {args.r2_threshold})")
    
    # Save probe weights
    W_affect = probe.get_weights()
    save_probe_weights(
        output_dir / "probe_weights.npz",
        W_affect, probe.b,
        metadata={"r2_per_dim": probe.get_r2_scores().tolist()}
    )
    print(f"Saved probe weights to {output_dir}/probe_weights.npz")
    
    # -------------------------------------------------------------------------
    # Step 2: Compute Subspace Decomposition
    # -------------------------------------------------------------------------
    print_section("Step 2: Computing Subspace Decomposition")
    
    subspace = AffectSubspace(W_affect, orthonormalize=True, valid_mask=valid_mask)
    
    # Verify subspace properties
    subspace_info = subspace.get_subspace_info()
    print(f"Subspace rank: {subspace_info['projection_rank']}")
    print(f"Orthonormalized: {subspace_info['orthonormalized']}")
    print(f"Valid projection: {subspace_info['is_valid_projection']}")
    
    # Decompose embeddings
    U_train, H_aff_train, H_non_train = subspace.decompose(H_train)
    U_val, H_aff_val, H_non_val = subspace.decompose(H_val)
    U_test, H_aff_test, H_non_test = subspace.decompose(H_test)
    
    # Report variance explained
    var_ratio = subspace.explained_variance_ratio(H_test)
    ortho_check = subspace.orthogonality_check(H_test)
    print(f"Affect subspace explains {100*var_ratio:.1f}% of embedding variance")
    print(f"Orthogonality check (mean |inner product|): {ortho_check:.2e}")
    
    # -------------------------------------------------------------------------
    # Step 3: Train Reference Model (M_full)
    # -------------------------------------------------------------------------
    print_section("Step 3: Training Reference Model (M_full)")
    
    # Scale metadata
    C_scaler = StandardScaler()
    C_train_scaled = C_scaler.fit_transform(C_train)
    C_val_scaled = C_scaler.transform(C_val)
    C_test_scaled = C_scaler.transform(C_test)
    
    # Build full input (H + C + A)
    Z_full_train = assemble_full_input(H_train, C_train, A_train)
    Z_full_val = assemble_full_input(H_val, C_val, A_val)
    Z_full_test = assemble_full_input(H_test, C_test, A_test)
    
    # Scale full input
    from data import FeatureScaler
    from config import INDEX_SLICES
    full_scaler = FeatureScaler(INDEX_SLICES["metadata"], INDEX_SLICES["affect"])
    Z_full_train = full_scaler.fit_transform(Z_full_train)
    Z_full_val = full_scaler.transform(Z_full_val)
    Z_full_test = full_scaler.transform(Z_full_test)
    
    model_full = build_engagement_model(ModelVariant.FULL)
    train_loader_full, val_loader_full, test_loader_full = build_dataloaders(
        Z_full_train, Y_train, Z_full_val, Y_val, Z_full_test, Y_test,
        batch_size=args.batch_size
    )
    
    trainer_full = Trainer(model_full, device=device)
    trainer_full.fit(train_loader_full, val_loader_full, epochs=args.epochs)
    
    mse_full, r2_rt_full, r2_fav_full = evaluate(model_full, test_loader_full, device)
    results["M_full"] = {"mse": mse_full, "r2_retweets": r2_rt_full, "r2_favorites": r2_fav_full}
    print(f"Result: {format_metrics(results['M_full'])}")
    
    save_checkpoint(model_full, output_dir / "model_full.pt", metrics=results["M_full"])
    
    # -------------------------------------------------------------------------
    # Step 4: Train M_affect_only
    # -------------------------------------------------------------------------
    print_section("Step 4: Training M_affect_only")
    
    affect_dim = U_train.shape[1]
    affect_only_input_dim = affect_dim + C_train_scaled.shape[1]
    
    print(f"Affect dimensions: {affect_dim}")
    print(f"Input dimension: {affect_only_input_dim}")
    
    Z_aff_train = prepare_affect_only_input(U_train, C_train_scaled)
    Z_aff_val = prepare_affect_only_input(U_val, C_val_scaled)
    Z_aff_test = prepare_affect_only_input(U_test, C_test_scaled)
    
    model_aff = build_engagement_model(ModelVariant.AFFECT_ONLY, input_dim_override=affect_only_input_dim)
    print(f"Model parameters: {model_aff.count_parameters():,}")
    
    train_loader_aff, val_loader_aff, test_loader_aff = build_dataloaders(
        Z_aff_train, Y_train, Z_aff_val, Y_val, Z_aff_test, Y_test,
        batch_size=args.batch_size
    )
    
    trainer_aff = Trainer(model_aff, device=device)
    trainer_aff.fit(train_loader_aff, val_loader_aff, epochs=args.epochs)
    
    mse_aff, r2_rt_aff, r2_fav_aff = evaluate(model_aff, test_loader_aff, device)
    results["M_affect_only"] = {"mse": mse_aff, "r2_retweets": r2_rt_aff, "r2_favorites": r2_fav_aff}
    print(f"Result: {format_metrics(results['M_affect_only'])}")
    
    save_checkpoint(model_aff, output_dir / "model_affect_only.pt", metrics=results["M_affect_only"])
    
    # -------------------------------------------------------------------------
    # Step 5: Train M_non_affect
    # -------------------------------------------------------------------------
    print_section("Step 5: Training M_non_affect")
    
    Z_non_train = prepare_non_affect_input(H_non_train, C_train_scaled)
    Z_non_val = prepare_non_affect_input(H_non_val, C_val_scaled)
    Z_non_test = prepare_non_affect_input(H_non_test, C_test_scaled)
    
    model_non = build_engagement_model(ModelVariant.NON_AFFECT)
    print(f"Input dimension: {model_non.input_dim}")
    print(f"Model parameters: {model_non.count_parameters():,}")
    
    train_loader_non, val_loader_non, test_loader_non = build_dataloaders(
        Z_non_train, Y_train, Z_non_val, Y_val, Z_non_test, Y_test,
        batch_size=args.batch_size
    )
    
    trainer_non = Trainer(model_non, device=device)
    trainer_non.fit(train_loader_non, val_loader_non, epochs=args.epochs)
    
    mse_non, r2_rt_non, r2_fav_non = evaluate(model_non, test_loader_non, device)
    results["M_non_affect"] = {"mse": mse_non, "r2_retweets": r2_rt_non, "r2_favorites": r2_fav_non}
    print(f"Result: {format_metrics(results['M_non_affect'])}")
    
    save_checkpoint(model_non, output_dir / "model_non_affect.pt", metrics=results["M_non_affect"])
    
    return results, {
        "probe": probe,
        "probe_summary": probe_summary,
        "subspace": subspace,
        "subspace_info": subspace_info,
        "variance_ratio": var_ratio,
        "orthogonality_check": ortho_check,
    }


def main():
    args = parse_args()
    set_seed(args.seed)
    output_dir = ensure_dir(args.output)
    
    print_section("Subspace Experiments: Affect vs Non-Affect Signal")
    
    # Load or extract features
    H, C, A, Y = load_or_extract_features(args)
    
    # Split into train/val/test
    n = len(H)
    idx = np.random.permutation(n)
    
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]
    
    H_train, H_val, H_test = H[train_idx], H[val_idx], H[test_idx]
    C_train, C_val, C_test = C[train_idx], C[val_idx], C[test_idx]
    A_train, A_val, A_test = A[train_idx], A[val_idx], A[test_idx]
    Y_train, Y_val, Y_test = Y[train_idx], Y[val_idx], Y[test_idx]
    
    print(f"Split: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")
    
    # Run experiments
    results, subspace_data = run_subspace_experiment(
        H_train, C_train, A_train, Y_train,
        H_val, C_val, A_val, Y_val,
        H_test, C_test, A_test, Y_test,
        args,
        output_dir
    )
    
    # Summarize results
    print_section("Subspace Results")
    
    summary = summarize_subspace(
        mse_affect_only=results["M_affect_only"]["mse"],
        mse_non_affect=results["M_non_affect"]["mse"],
        mse_full=results["M_full"]["mse"]
    )
    
    print(subspace_summary_text(summary))
    
    # Additional metrics
    print("\nSubspace Properties:")
    print(f"  Variance explained by affect subspace: {100*subspace_data['variance_ratio']:.2f}%")
    print(f"  Orthogonality check: {subspace_data['orthogonality_check']:.2e}")
    print(f"  Subspace rank: {subspace_data['subspace_info']['projection_rank']}")
    
    # Save results
    results_df = pd.DataFrame([
        {"Model": k, **v} for k, v in results.items()
    ]).set_index("Model")
    
    results_df.to_csv(output_dir / "subspace_results.csv")
    print(f"\nSaved results to {output_dir}/subspace_results.csv")
    
    # Save probe summary
    with open(output_dir / "probe_summary.txt", "w") as f:
        f.write(subspace_data["probe_summary"])
    print(f"Saved probe summary to {output_dir}/probe_summary.txt")
    
    print_section("Complete")
    
    # Return summary for programmatic use
    return {
        "results": results,
        "summary": summary,
        "subspace_data": subspace_data
    }


if __name__ == "__main__":
    main()

