#!/usr/bin/env python3
"""
End-to-end pipeline for emotion-engagement prediction analysis.

This script executes the complete workflow:
    1. Load data and extract features (H, C, A)
    2. Train main engagement models (M_full, M_text+meta, M_shuffled)
    3. Fit affect probe and compute subspace decomposition
    4. Train subspace models (M_affect_only, M_non_affect)
    5. Run SHAP analysis and generate final report

Usage:
    python run_pipeline.py --data path/to/tweets.csv --output results/
    python run_pipeline.py --data tweets.parquet --vad-lexicon nrc_vad.txt

Disclaimer: This runner script was primarily written with the help of Cursor.
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from sklearn.preprocessing import StandardScaler

from config import (
    TRAINING_DEFAULTS, RANDOM_SEED, INDEX_SLICES,
    AFFECT_FEATURES
)
from data import (
    load_tweets, split_data, prepare_targets,
    FeatureScaler, build_dataloaders
)
from features import FeatureBuilder, VADScorer
from models import (
    ModelVariant, build_engagement_model, AffectProbe,
    assemble_text_meta_input, assemble_shuffled_input
)
from subspace import (
    AffectSubspace, prepare_affect_only_input, prepare_non_affect_input
)
from training import Trainer, evaluate, compare_variants
from analysis import (
    run_shap, rank_affect_importance, summarize_ablation,
    summarize_subspace, generate_report, ablation_summary_text,
    subspace_summary_text
)
from utils import (
    set_seed, get_device, save_checkpoint, save_arrays,
    save_probe_weights, ensure_dir, print_section, format_metrics
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the emotion-engagement prediction pipeline"
    )
    parser.add_argument(
        "--data", type=str, required=True,
        help="Path to tweet data file (CSV, Parquet, or JSON)"
    )
    parser.add_argument(
        "--output", type=str, default="results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--vad-lexicon", type=str, default=None,
        help="Path to NRC VAD lexicon file (optional)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=TRAINING_DEFAULTS["batch_size"],
        help="Training batch size"
    )
    parser.add_argument(
        "--epochs", type=int, default=TRAINING_DEFAULTS["epochs"],
        help="Maximum training epochs"
    )
    parser.add_argument(
        "--seed", type=int, default=RANDOM_SEED,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--skip-shap", action="store_true",
        help="Skip SHAP analysis (faster)"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device to use (cuda/cpu)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup
    set_seed(args.seed)
    device = args.device or str(get_device())
    output_dir = ensure_dir(args.output)
    
    print_section("Emotion-Engagement Prediction Pipeline")
    print(f"Data: {args.data}")
    print(f"Output: {output_dir}")
    print(f"Device: {device}")
    
    # =========================================================================
    # Step 1: Load Data and Extract Features
    # =========================================================================
    print_section("Step 1: Data Loading and Feature Extraction")
    
    # Load raw data
    print("Loading tweet data...")
    df = load_tweets(args.data)
    print(f"  Loaded {len(df)} tweets")
    
    # Split data
    print("Splitting into train/val/test...")
    train_df, val_df, test_df = split_data(df, seed=args.seed)
    print(f"  Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Initialize feature extractors
    print("Initializing feature extractors...")
    vad_scorer = None
    if args.vad_lexicon:
        vad_scorer = VADScorer(args.vad_lexicon)
        print(f"  Loaded VAD lexicon with {len(vad_scorer.lexicon)} words")
    
    builder = FeatureBuilder(vad_scorer=vad_scorer, device=device)
    
    # Extract features for each split
    print("Extracting features (this may take a while)...")
    
    # Use 'clean_tweet' column from the Kaggle COVID-19 dataset
    text_col = "clean_tweet"
    
    print("  Processing training set...")
    H_train, C_train, A_train, Z_train = builder.build_full_z(train_df, text_col=text_col)
    
    print("  Processing validation set...")
    H_val, C_val, A_val, Z_val = builder.build_full_z(val_df, text_col=text_col)
    
    print("  Processing test set...")
    H_test, C_test, A_test, Z_test = builder.build_full_z(test_df, text_col=text_col)
    
    # Prepare targets
    Y_train = prepare_targets(train_df)
    Y_val = prepare_targets(val_df)
    Y_test = prepare_targets(test_df)
    
    print(f"  Feature shapes: H={H_train.shape}, C={C_train.shape}, A={A_train.shape}")
    print(f"  Full Z shape: {Z_train.shape}")
    
    # Scale metadata and affect features
    print("Scaling features...")
    scaler = FeatureScaler(INDEX_SLICES["metadata"], INDEX_SLICES["affect"])
    Z_train = scaler.fit_transform(Z_train)
    Z_val = scaler.transform(Z_val)
    Z_test = scaler.transform(Z_test)
    
    # Save features
    save_arrays(
        output_dir / "features_train.npz",
        H=H_train, C=C_train, A=A_train, Z=Z_train, Y=Y_train
    )
    print(f"  Saved features to {output_dir}/features_train.npz")
    
    # =========================================================================
    # Step 2: Train Main Engagement Models
    # =========================================================================
    print_section("Step 2: Training Engagement Models")
    
    results = {}
    
    # 2a. Full model (H + C + A)
    print("\nTraining M_full (text + metadata + affect)...")
    model_full = build_engagement_model(ModelVariant.FULL)
    print(f"  Parameters: {model_full.count_parameters():,}")
    
    train_loader, val_loader, test_loader = build_dataloaders(
        Z_train, Y_train, Z_val, Y_val, Z_test, Y_test,
        batch_size=args.batch_size
    )
    
    trainer = Trainer(model_full, device=device)
    trainer.fit(train_loader, val_loader, epochs=args.epochs)
    
    mse_full, r2_rt_full, r2_fav_full = evaluate(model_full, test_loader, device)
    results["M_full"] = {"mse": mse_full, "r2_retweets": r2_rt_full, "r2_favorites": r2_fav_full}
    print(f"  Test: {format_metrics(results['M_full'])}")
    
    save_checkpoint(model_full, output_dir / "model_full.pt", metrics=results["M_full"])
    
    # 2b. Text + Metadata model (no affect)
    print("\nTraining M_text+meta (text + metadata only)...")
    Z_tm_train = assemble_text_meta_input(H_train, C_train)
    Z_tm_val = assemble_text_meta_input(H_val, C_val)
    Z_tm_test = assemble_text_meta_input(H_test, C_test)
    
    # Re-scale just metadata for this variant
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
    print(f"  Test: {format_metrics(results['M_text+meta'])}")
    
    save_checkpoint(model_tm, output_dir / "model_text_meta.pt", metrics=results["M_text+meta"])
    
    # 2c. Shuffled model (permuted affect)
    print("\nTraining M_shuffled (shuffled affect control)...")
    Z_shuf_train = assemble_shuffled_input(H_train, C_train, A_train, seed=args.seed)
    Z_shuf_val = assemble_shuffled_input(H_val, C_val, A_val, seed=args.seed + 1)
    Z_shuf_test = assemble_shuffled_input(H_test, C_test, A_test, seed=args.seed + 2)
    
    # Apply same scaling as full model
    Z_shuf_train = scaler.transform(np.concatenate([H_train, C_train, Z_shuf_train[:, 394:]], axis=1))
    # Simpler: just use pre-scaled components
    Z_shuf_train = scaler.transform(assemble_shuffled_input(H_train, C_train, A_train, seed=args.seed))
    Z_shuf_val = scaler.transform(assemble_shuffled_input(H_val, C_val, A_val, seed=args.seed + 1))
    Z_shuf_test = scaler.transform(assemble_shuffled_input(H_test, C_test, A_test, seed=args.seed + 2))
    
    model_shuf = build_engagement_model(ModelVariant.SHUFFLED)
    train_loader_shuf, val_loader_shuf, test_loader_shuf = build_dataloaders(
        Z_shuf_train, Y_train, Z_shuf_val, Y_val, Z_shuf_test, Y_test,
        batch_size=args.batch_size
    )
    
    trainer_shuf = Trainer(model_shuf, device=device)
    trainer_shuf.fit(train_loader_shuf, val_loader_shuf, epochs=args.epochs)
    
    mse_shuf, r2_rt_shuf, r2_fav_shuf = evaluate(model_shuf, test_loader_shuf, device)
    results["M_shuffled"] = {"mse": mse_shuf, "r2_retweets": r2_rt_shuf, "r2_favorites": r2_fav_shuf}
    print(f"  Test: {format_metrics(results['M_shuffled'])}")
    
    save_checkpoint(model_shuf, output_dir / "model_shuffled.pt", metrics=results["M_shuffled"])
    
    # =========================================================================
    # Step 3: Fit Affect Probe and Compute Subspace
    # =========================================================================
    print_section("Step 3: Affect Subspace Analysis")
    
    print("Fitting affect probe (ridge regression)...")
    probe = AffectProbe()
    probe.fit(H_train, A_train)
    
    probe_r2 = probe.score(H_test, A_test)
    print(f"  Probe R^2 on test set: {probe_r2:.4f}")
    
    # Linearity check: per-dimension R^2
    probe_summary = probe.summary(AFFECT_FEATURES)
    print("\n" + probe_summary)
    
    n_valid = np.sum(probe.get_valid_dimension_mask())
    print(f"\n  Using {n_valid}/{len(AFFECT_FEATURES)} dimensions for subspace (R^2 >= {probe.r2_threshold})")
    
    W_affect = probe.get_weights()
    save_probe_weights(output_dir / "probe_weights.npz", W_affect, probe.b, 
                       metadata={"r2_per_dim": probe.get_r2_scores().tolist()})
    print(f"  Saved probe weights to {output_dir}/probe_weights.npz")
    
    # Build subspace projections with QR orthonormalization
    # Optionally filter to valid dimensions only
    print("\nComputing subspace decomposition (with QR orthonormalization)...")
    valid_mask = probe.get_valid_dimension_mask()
    subspace = AffectSubspace(W_affect, orthonormalize=True, valid_mask=valid_mask)
    
    # Verify orthogonality
    subspace_info = subspace.get_subspace_info()
    print(f"  Subspace rank: {subspace_info['projection_rank']}")
    print(f"  Orthonormalized: {subspace_info['orthonormalized']}")
    print(f"  Valid projection: {subspace_info['is_valid_projection']}")
    
    U_train, _, H_non_train = subspace.decompose(H_train)
    U_val, _, H_non_val = subspace.decompose(H_val)
    U_test, _, H_non_test = subspace.decompose(H_test)
    
    var_ratio = subspace.explained_variance_ratio(H_test)
    ortho_check = subspace.orthogonality_check(H_test)
    print(f"  Affect subspace explains {100*var_ratio:.1f}% of embedding variance")
    print(f"  Orthogonality check (mean |inner product|): {ortho_check:.2e}")
    
    # =========================================================================
    # Step 4: Train Subspace Models
    # =========================================================================
    print_section("Step 4: Training Subspace Models")
    
    # Scale metadata for subspace models
    C_scaler = StandardScaler()
    C_train_scaled = C_scaler.fit_transform(C_train)
    C_val_scaled = C_scaler.transform(C_val)
    C_test_scaled = C_scaler.transform(C_test)
    
    # 4a. Affect-only model
    # Note: dimension may be reduced if some affect dimensions failed linearity check
    affect_dim_actual = U_train.shape[1]  # May be < 15 if valid_mask filtered some out
    affect_only_input_dim = affect_dim_actual + C_train_scaled.shape[1]
    
    print(f"\nTraining M_affect_only (affect coords + metadata)...")
    print(f"  Affect dimensions used: {affect_dim_actual} (after linearity filtering)")
    
    Z_aff_train = prepare_affect_only_input(U_train, C_train_scaled)
    Z_aff_val = prepare_affect_only_input(U_val, C_val_scaled)
    Z_aff_test = prepare_affect_only_input(U_test, C_test_scaled)
    
    model_aff = build_engagement_model(ModelVariant.AFFECT_ONLY, input_dim_override=affect_only_input_dim)
    print(f"  Input dim: {model_aff.input_dim}, Parameters: {model_aff.count_parameters():,}")
    
    train_loader_aff, val_loader_aff, test_loader_aff = build_dataloaders(
        Z_aff_train, Y_train, Z_aff_val, Y_val, Z_aff_test, Y_test,
        batch_size=args.batch_size
    )
    
    trainer_aff = Trainer(model_aff, device=device)
    trainer_aff.fit(train_loader_aff, val_loader_aff, epochs=args.epochs)
    
    mse_aff, r2_rt_aff, r2_fav_aff = evaluate(model_aff, test_loader_aff, device)
    results["M_affect_only"] = {"mse": mse_aff, "r2_retweets": r2_rt_aff, "r2_favorites": r2_fav_aff}
    print(f"  Test: {format_metrics(results['M_affect_only'])}")
    
    save_checkpoint(model_aff, output_dir / "model_affect_only.pt", metrics=results["M_affect_only"])
    
    # 4b. Non-affect model
    print("\nTraining M_non_affect (residual embedding + metadata)...")
    Z_non_train = prepare_non_affect_input(H_non_train, C_train_scaled)
    Z_non_val = prepare_non_affect_input(H_non_val, C_val_scaled)
    Z_non_test = prepare_non_affect_input(H_non_test, C_test_scaled)
    
    model_non = build_engagement_model(ModelVariant.NON_AFFECT)
    train_loader_non, val_loader_non, test_loader_non = build_dataloaders(
        Z_non_train, Y_train, Z_non_val, Y_val, Z_non_test, Y_test,
        batch_size=args.batch_size
    )
    
    trainer_non = Trainer(model_non, device=device)
    trainer_non.fit(train_loader_non, val_loader_non, epochs=args.epochs)
    
    mse_non, r2_rt_non, r2_fav_non = evaluate(model_non, test_loader_non, device)
    results["M_non_affect"] = {"mse": mse_non, "r2_retweets": r2_rt_non, "r2_favorites": r2_fav_non}
    print(f"  Test: {format_metrics(results['M_non_affect'])}")
    
    save_checkpoint(model_non, output_dir / "model_non_affect.pt", metrics=results["M_non_affect"])
    
    # =========================================================================
    # Step 5: SHAP Analysis and Report Generation
    # =========================================================================
    print_section("Step 5: Analysis and Report")
    
    # Compute ablation and subspace summaries
    ablation_summary = summarize_ablation(
        mse_full=results["M_full"]["mse"],
        mse_text_meta=results["M_text+meta"]["mse"],
        mse_shuffled=results["M_shuffled"]["mse"]
    )
    
    subspace_summary = summarize_subspace(
        mse_affect_only=results["M_affect_only"]["mse"],
        mse_non_affect=results["M_non_affect"]["mse"],
        mse_full=results["M_full"]["mse"]
    )
    
    print(ablation_summary_text(ablation_summary))
    print()
    print(subspace_summary_text(subspace_summary))
    
    # Get VAD coverage stats
    vad_coverage = builder.get_vad_coverage_stats()
    if vad_coverage:
        print(f"\nVAD lexicon coverage: {100*vad_coverage['coverage_fraction']:.1f}% tweets with >= {vad_coverage['min_matches_threshold']} matches")
    
    # SHAP analysis
    importance_df = None
    if not args.skip_shap:
        print("\nRunning SHAP analysis (this may take a while)...")
        
        # Sample background and explanation sets
        n_background = min(500, len(Z_train))
        n_explain = min(200, len(Z_test))
        
        idx_bg = np.random.choice(len(Z_train), n_background, replace=False)
        idx_ex = np.random.choice(len(Z_test), n_explain, replace=False)
        
        try:
            shap_values = run_shap(
                model_full,
                Z_train[idx_bg],
                Z_test[idx_ex],
                device=device
            )
            
            importance_df = rank_affect_importance(shap_values, output_idx=0)
            print("\nTop affect features by importance:")
            print(importance_df.head(10).to_string(index=False))
            
            importance_df.to_csv(output_dir / "affect_importance.csv", index=False)
        except Exception as e:
            print(f"  SHAP analysis failed: {e}")
            print("  Skipping SHAP (install 'shap' package for this feature)")
    
    # Generate and save final report (now includes probe summary, VAD coverage, and causal caveat)
    report = generate_report(
        ablation_summary, 
        subspace_summary, 
        importance_df,
        probe_summary=probe_summary,
        vad_coverage=vad_coverage
    )
    
    report_path = output_dir / "report.md"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\nSaved report to {report_path}")
    
    # Save comparison table
    comparison_df = compare_variants(results)
    comparison_df.to_csv(output_dir / "model_comparison.csv")
    print(f"Saved model comparison to {output_dir}/model_comparison.csv")
    
    print_section("Pipeline Complete")
    print(f"All results saved to: {output_dir}/")


if __name__ == "__main__":
    main()