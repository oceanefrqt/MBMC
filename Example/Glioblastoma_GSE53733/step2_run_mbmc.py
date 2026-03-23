"""
Step 2 — Running the MBMC Pipeline (Glioblastoma GSE53733)
==========================================================
This script runs the Multi-class Bivariate Monotonic Classifier (MBMC)
pipeline on the train / test CSVs produced by step1_prepare_data.py.

Pipeline overview:
  1. Load the train and test datasets.
  2. Find the optimal ensemble size k (k_misclassification): runs the
     preselection algorithm once with m = max_k, computes the k-fold CV
     error matrix, and evaluates ensemble sizes k = 1 .. max_k.
  3. Build the final ensemble model with k_opt disjoint gene pairs.
  4. Predict on the test set with majority-vote ensemble prediction.
  5. Save predictions to CSV.

Usage:
  python step2_run_mbmc.py [--train FILE] [--test FILE]
                            [--nbcpus N] [--max-k K] [--kfold F]

Examples:
  # Default settings (4 CPUs, max_k=5, 5-fold CV):
  python step2_run_mbmc.py

  # Explicit data paths:
  python step2_run_mbmc.py --train DATA/GSE53733_MAD_train.csv \
                            --test  DATA/GSE53733_MAD_test.csv

  # Custom settings:
  python step2_run_mbmc.py --nbcpus 8 --max-k 10 --kfold 10

The script reads:
  - GSE53733_MAD_train.csv  (from step1_prepare_data.py)
  - GSE53733_MAD_test.csv   (from step1_prepare_data.py)

And writes:
  - ensemble_predictions.csv — test-set predictions from the ensemble model

Dependencies:
  - pandas, numpy
  - The Module package (automatically located relative to this script)
"""

import argparse
import os
import sys

import pandas as pd

# ---------------------------------------------------------------------------
# PATH SETUP
# ---------------------------------------------------------------------------
# Add the MBMC root directory to sys.path so that `Module` can be imported.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import Module.mbmc as mmck


# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------

def load_data(train_path: str, test_path: str) -> tuple:
    """Load train and test CSV files produced by step1_prepare_data.py.

    Both DataFrames are expected to have gene columns and a 'target' column
    with integer class labels {0, 1, 2}.  NaN rows are dropped.

    Parameters
    ----------
    train_path : str
    test_path  : str

    Returns
    -------
    (df_train, df_test)
    """
    df_train = pd.read_csv(train_path, index_col=0, low_memory=False)
    df_test  = pd.read_csv(test_path,  index_col=0, low_memory=False)

    df_train.dropna(inplace=True)
    df_test.dropna(inplace=True)

    print(f"Train: {df_train.shape[0]} samples × {df_train.shape[1]-1} genes")
    print(f"Test:  {df_test.shape[0]} samples × {df_test.shape[1]-1} genes")
    print(f"Train class distribution: {dict(sorted(df_train['target'].value_counts().items()))}")
    print(f"Test  class distribution: {dict(sorted(df_test['target'].value_counts().items()))}")

    return df_train, df_test


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Run the MBMC pipeline on the Glioblastoma GSE53733 dataset."
    )
    parser.add_argument(
        "--train", default="DATA/GSE53733_MAD_train.csv",
        help="Path to the training CSV (default: DATA/GSE53733_MAD_train.csv)"
    )
    parser.add_argument(
        "--test", default="DATA/GSE53733_MAD_test.csv",
        help="Path to the test CSV (default: DATA/GSE53733_MAD_test.csv)"
    )
    parser.add_argument(
        "--nbcpus", type=int, default=4,
        help="Number of parallel CPU cores to use (default: 4)"
    )
    parser.add_argument(
        "--max-k", type=int, default=5,
        help="Maximum ensemble size to evaluate (default: 5)"
    )
    parser.add_argument(
        "--kfold", type=int, default=5,
        help="Number of cross-validation folds (default: 5)"
    )
    args = parser.parse_args()

    nbcpus = args.nbcpus
    max_k  = args.max_k
    kfold  = args.kfold

    print("=" * 60)
    print("STEP 2: MBMC PIPELINE — GLIOBLASTOMA GSE53733")
    print("=" * 60)
    print(f"Parameters: nbcpus={nbcpus}, max_k={max_k}, kfold={kfold}\n")

    # 1. Load data
    df_train, df_test = load_data(args.train, args.test)

    # 2. Find the optimal ensemble size
    k_mae, k_opt = mmck.k_misclassification(df_train, nbcpus, 1, max_k, kfold)
    print(f"\nMAE-CVE per ensemble size: {k_mae}")
    print(f"Optimal k: {k_opt}")

    # 3. Build the ensemble model
    pairs = mmck.ensemble_model(df_train, k_opt, nbcpus, kfold)
    print(f"\nSelected pairs ({k_opt} disjoint classifiers):")
    for i, p in enumerate(pairs, 1):
        g1, g2, key = p.split("/")
        print(f"  {i}. {g1} / {g2}  (key={key})")

    # 4. Predict on the test set
    preds, probas = mmck.create_and_predict_ensemble_model(df_train, df_test, pairs, nbcpus)

    # 5. Evaluate and save
    true_labels = df_test["target"].tolist()
    mae = mmck.mean_absolute_error(true_labels, preds)
    print(f"\nTest MAE: {mae:.4f}")

    out = pd.DataFrame({"true": true_labels, "pred": preds}, index=df_test.index)
    out.to_csv("ensemble_predictions.csv")
    print("Predictions saved to ensemble_predictions.csv")
    print("\nNext step: run step3_evaluate.py for detailed performance metrics.")
