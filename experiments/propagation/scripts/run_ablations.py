#!/usr/bin/env python3
"""
Focused ablation experiment script.

Runs just the core ablation comparison (M_full vs M_text+meta vs M_shuffled)
without the subspace analysis or SHAP. Useful for quick iteration when
testing the impact of explicit affect features.

Usage:
    python run_ablations.py --features features_train.npz --output ablation_results/
    python run_ablations.py --data tweets.csv --quick  # Subsample for fast testing

Disclaimer: This runner script was written with the help of Cursor.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from config import TRAINING_DEFAULTS, RANDOM_SEED, INDEX_SLICES
from data import (
    load_tweets, prepare_targets,
    FeatureScaler, build_dataloaders
)
from features import FeatureBuilder
from models import (
    ModelVariant, build_engagement_model,
    assemble_full_input, assemble_text_meta_input, assemble_shuffled_input
)
from training import Trainer, evaluate, compute_ablation_deltas
from analysis import summarize_ablation, ablation_summary_text
from utils import (
    set_seed, get_device, save_checkpoint, load_arrays,
    ensure_dir, print_section, format_metrics
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run ablation experiments for affect feature importance"
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
        "--output", type=str, default="ablation_results",
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
    
    return parser.parse_args()


def load_or_extract_features(args):
    """Load pre-extracted features or extract from raw data."""
    
    if args.features:
        # Load pre-extracted features
        print(f"Loading features from {args.features}...")
        data = load_arrays(args.features)
        
        # Expect H, C, A, Y in the file; reconstruct Z
        H = data["H"]
        C = data["C"]
        A = data["A"]
        Y = data["Y"]
        Z = data.get("Z", np.concatenate([H, C, A], axis=1))
        
        return H, C, A, Z, Y
    
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
        
        return H, C, A, Z, Y


def run_ablation_experiment(
    H_train, C_train, A_train, Y_train,
    H_val, C_val, A_val, Y_val,
    H_test, C_test, A_test, Y_test,
    args
):
    """
    Run the three-way ablation experiment.
    
    Compares:
        - M_full: H + C + A
        - M_text+meta: H + C only
        - M_shuffled: H + C + A_permuted
    """
    device = args.device or str(get_device())
    results = {}
    
    # Scale features
    Z_full_train = assemble_full_input(H_train, C_train, A_train)
    Z_full_val = assemble_full_input(H_val, C_val, A_val)
    Z_full_test = assemble_full_input(H_test, C_test, A_test)
    
    scaler = FeatureScaler(INDEX_SLICES["metadata"], INDEX_SLICES["affect"])
    Z_full_train = scaler.fit_transform(Z_full_train)
    Z_full_val = scaler.transform(Z_full_val)
    Z_full_test = scaler.transform(Z_full_test)
    
    # -------------------------------------------------------------------------
    # Model 1: Full (H + C + A)
    # -------------------------------------------------------------------------
    print("\n[1/3] Training M_full (text + metadata + affect)...")
    
    model_full = build_engagement_model(ModelVariant.FULL)
    train_loader, val_loader, test_loader = build_dataloaders(
        Z_full_train, Y_train, Z_full_val, Y_val, Z_full_test, Y_test,
        batch_size=args.batch_size
    )
    
    trainer = Trainer(model_full, device=device)
    history = trainer.fit(train_loader, val_loader, epochs=args.epochs)
    
    mse, r2_rt, r2_fav = evaluate(model_full, test_loader, device)
    results["M_full"] = {"mse": mse, "r2_retweets": r2_rt, "r2_favorites": r2_fav}
    print(f"  Result: {format_metrics(results['M_full'])}")
    
    # -------------------------------------------------------------------------
    # Model 2: Text + Metadata (no affect)
    # -------------------------------------------------------------------------
    print("\n[2/3] Training M_text+meta (no affect vector)...")
    
    Z_tm_train = assemble_text_meta_input(H_train, C_train)
    Z_tm_val = assemble_text_meta_input(H_val, C_val)
    Z_tm_test = assemble_text_meta_input(H_test, C_test)
    
    # Scale metadata portion
    from sklearn.preprocessing import StandardScaler
    meta_scaler = StandardScaler()
    Z_tm_train[:, 384:] = meta_scaler.fit_transform(Z_tm_train[:, 384:])
    Z_tm_val[:, 384:] = meta_scaler.transform(Z_tm_val[:, 384:])
    Z_tm_test[:, 384:] = meta_scaler.transform(Z_tm_test[:, 384:])
    
    model_tm = build_engagement_model(ModelVariant.TEXT_META)
    train_loader_tm, val_loader_tm, test_loader_tm = build_dataloaders(
        Z_tm_train, Y_train, Z_tm_val, Y_val, Z_tm_test, Y_test,
        batch_size=args.batch_size
    )
    
    trainer_tm = Trainer(model_tm, device=device)
    trainer_tm.fit(train_loader_tm, val_loader_tm, epochs=args.epochs)
    
    mse_tm, r2_rt_tm, r2_fav_tm = evaluate(model_tm, test_loader_tm, device)
    results["M_text+meta"] = {"mse": mse_tm, "r2_retweets": r2_rt_tm, "r2_favorites": r2_fav_tm}
    print(f"  Result: {format_metrics(results['M_text+meta'])}")
    
    # -------------------------------------------------------------------------
    # Model 3: Shuffled Affect
    # -------------------------------------------------------------------------
    print("\n[3/3] Training M_shuffled (permuted affect control)...")
    
    Z_shuf_train = assemble_shuffled_input(H_train, C_train, A_train, seed=args.seed)
    Z_shuf_val = assemble_shuffled_input(H_val, C_val, A_val, seed=args.seed + 1)
    Z_shuf_test = assemble_shuffled_input(H_test, C_test, A_test, seed=args.seed + 2)
    
    Z_shuf_train = scaler.transform(Z_shuf_train)
    Z_shuf_val = scaler.transform(Z_shuf_val)
    Z_shuf_test = scaler.transform(Z_shuf_test)
    
    model_shuf = build_engagement_model(ModelVariant.SHUFFLED)
    train_loader_shuf, val_loader_shuf, test_loader_shuf = build_dataloaders(
        Z_shuf_train, Y_train, Z_shuf_val, Y_val, Z_shuf_test, Y_test,
        batch_size=args.batch_size
    )
    
    trainer_shuf = Trainer(model_shuf, device=device)
    trainer_shuf.fit(train_loader_shuf, val_loader_shuf, epochs=args.epochs)
    
    mse_shuf, r2_rt_shuf, r2_fav_shuf = evaluate(model_shuf, test_loader_shuf, device)
    results["M_shuffled"] = {"mse": mse_shuf, "r2_retweets": r2_rt_shuf, "r2_favorites": r2_fav_shuf}
    print(f"  Result: {format_metrics(results['M_shuffled'])}")
    
    return results, model_full, model_tm, model_shuf


def main():
    args = parse_args()
    set_seed(args.seed)
    output_dir = ensure_dir(args.output)
    
    print_section("Ablation Experiments: Does Emotion Help?")
    
    # Load or extract features
    H, C, A, Z, Y = load_or_extract_features(args)
    
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
    results, model_full, model_tm, model_shuf = run_ablation_experiment(
        H_train, C_train, A_train, Y_train,
        H_val, C_val, A_val, Y_val,
        H_test, C_test, A_test, Y_test,
        args
    )
    
    # Summarize results
    print_section("Ablation Results")
    
    summary = summarize_ablation(
        mse_full=results["M_full"]["mse"],
        mse_text_meta=results["M_text+meta"]["mse"],
        mse_shuffled=results["M_shuffled"]["mse"]
    )
    
    print(ablation_summary_text(summary))
    
    # Compute deltas
    deltas = compute_ablation_deltas(
        results["M_full"]["mse"],
        results["M_text+meta"]["mse"],
        results["M_shuffled"]["mse"]
    )
    
    print("\nDetailed Deltas:")
    print(f"  Δ(no affect):  {deltas['delta_no_affect']:+.4f} ({deltas['relative_delta_no_affect_pct']:+.1f}%)")
    print(f"  Δ(shuffled):   {deltas['delta_shuffled']:+.4f} ({deltas['relative_delta_shuffled_pct']:+.1f}%)")
    
    # Save results
    results_df = pd.DataFrame([
        {"Model": k, **v} for k, v in results.items()
    ]).set_index("Model")
    
    results_df.to_csv(output_dir / "ablation_results.csv")
    print(f"\nSaved results to {output_dir}/ablation_results.csv")
    
    # Save models
    save_checkpoint(model_full, output_dir / "model_full.pt", metrics=results["M_full"])
    save_checkpoint(model_tm, output_dir / "model_text_meta.pt", metrics=results["M_text+meta"])
    save_checkpoint(model_shuf, output_dir / "model_shuffled.pt", metrics=results["M_shuffled"])
    
    print_section("Complete")
    
    # Return summary for programmatic use
    return {
        "results": results,
        "summary": summary,
        "deltas": deltas
    }


if __name__ == "__main__":
    main()