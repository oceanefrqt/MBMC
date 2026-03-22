"""
Step 2 — Running the MBMC Pipeline (Individual Pairs + Ensemble)
================================================================
This script runs the Multi-class Bivariate Monotonic Classifier (MBMC)
pipeline described in the paper:
  "Identification of Monotonically Classifying Pairs of Genes for Ordinal
   Disease Outcomes", Section 2.2.3.

It reproduces the logic of:
  - Cluster/Breast_Cancer/individual_BMC.py  (preselection + error matrix)
  - Cluster/Breast_Cancer/ensemble_BMC.py    (ensemble model + prediction)

Pipeline overview:
  1. Generate all possible gene-pair configurations (4 monotonicity directions
     per pair: increasing/decreasing on each axis).
  2. Preselect promising pairs using Algorithm 2 (based on full-data MAE as a
     lower bound for cross-validation error). This avoids computing k-fold CV
     for every pair in the large search space.
  3. Compute the k-fold CV error matrix for the preselected pairs.
  4. Build the ensemble model: select the best k disjoint pairs (no shared
     gene) and combine their predictions by majority vote.
  5. Save predictions and the error matrix to CSV files.

Usage:
  python step2_run_mbmc.py

The script reads:
  - breast_cancer_train.csv  (from step1_prepare_data.py)
  - breast_cancer_test.csv   (from step1_prepare_data.py)

And writes:
  - error_matrix_m{m}.csv   — per-sample CV errors for each preselected pair
  - ensemble_predictions.csv — test-set predictions from the ensemble model

Note on computation time:
  The gene-pair space grows as O(n_genes^2). With ~1,708 genes there are
  ~1.46 million pairs × 4 configurations = ~5.8 million classifiers. The
  preselection algorithm (Algorithm 2) drastically reduces this to a few
  hundred pairs before running the expensive k-fold CV. Use nbcpus to
  parallelize across CPU cores.

Dependencies:
  - pandas, numpy
  - The Clean_Module package (path must be added to sys.path below)
"""

import sys
import os
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# PATH SETUP
# ---------------------------------------------------------------------------
# Add the directory containing Clean_Module to the Python path.
# Adjust this path to wherever Clean_Module lives on your system.
CLEAN_MODULE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "Cluster", "Breast_Cancer"
)
sys.path.insert(0, CLEAN_MODULE_PATH)

import Module.multiclass_monotonic_classifiers_Kfold as mmck


# ---------------------------------------------------------------------------
# PARAMETERS
# ---------------------------------------------------------------------------

# Number of parallel CPU cores to use.
# On the cluster the authors used 23 CPUs. Adjust to your machine.
NBCPUS = 4

# K-fold cross-validation: 5 folds (as in the paper)
KFOLD = 5

# Minimum number of disjoint pairs to find at each preselection step.
# The paper tests m ∈ {5, 10, 20}, which define scenarios MBMC-5, MBMC-10,
# MBMC-20 respectively.
M_VALUES = [5, 10, 20]

# Maximum ensemble size to evaluate
MAX_K = 10


# ---------------------------------------------------------------------------
# 1. LOAD DATASETS
# ---------------------------------------------------------------------------

def load_data(train_path: str, test_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load train and test CSV files produced by step1_prepare_data.py.

    Both DataFrames are expected to have:
      - gene columns (all columns except 'target')
      - a 'target' column with integer class labels {0, 1, 2}

    NaN rows are dropped to avoid issues during monotonic regression.

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
# 2. GENERATE ALL PAIR CONFIGURATIONS
# ---------------------------------------------------------------------------

def generate_configurations(df: pd.DataFrame) -> list[str]:
    """
    Enumerate all possible gene-pair configurations.

    Each pair of genes (g1, g2) is associated with 4 monotonicity directions:
        key=1: g1 increasing, g2 increasing  (up-up)
        key=2: g1 decreasing, g2 increasing  (down-up)
        key=3: g1 increasing, g2 decreasing  (up-down) — encoded as (False,False)
        key=4: g1 decreasing, g2 decreasing  (down-down)

    The internal mapping is:
        equiv_to_key = {1: (False, True), 2: (True, True),
                        3: (False, False), 4: (True, False)}
    where the first bool is 'rev' (reverse axis 1) and the second is 'up'
    (upper boundary used for classification).

    Pairs are encoded as the string "gene1/gene2/key" to keep them hashable.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with gene columns and 'target' column.

    Returns
    -------
    list[str] — all configurations, e.g. ['BRCA1/TP53/1', 'BRCA1/TP53/2', ...]
    """
    configs = mmck.all_configurations(df)
    n_genes = df.shape[1] - 1  # exclude 'target'
    n_pairs = len(configs) // 4
    print(f"\nGenerated {len(configs):,} configurations "
          f"({n_genes} genes × {n_pairs:,} pairs × 4 directions)")
    return configs


# ---------------------------------------------------------------------------
# 3. PRESELECTION (ALGORITHM 2)
# ---------------------------------------------------------------------------

def run_preselection(configs: list[str],
                     df_train: pd.DataFrame,
                     m: int) -> list[str]:
    """
    Apply Algorithm 2 to select a compact set of promising gene pairs.

    The key insight (proven in the companion paper Fourquet et al. 2025) is
    that the MAE on the full training set (MAE_full) is a lower bound on the
    k-fold cross-validation MAE (MAE_CV). Algorithm 2 exploits this bound to
    avoid computing the expensive CV for poorly-performing pairs:

      1. Sort all pairs by MAE_full (cheap, computed in O(n log n) per pair).
      2. Process pairs in increasing order of MAE_full.
      3. For each batch, compute MAE_CV (expensive).
      4. Stop as soon as the set of processed pairs contains at least m
         disjoint pairs (pairs sharing no gene).

    Parameters
    ----------
    configs  : list[str]   — all pair configurations
    df_train : pd.DataFrame
    m        : int         — target number of disjoint pairs

    Returns
    -------
    list[str] — preselected pair configurations
    """
    print(f"\nRunning preselection (Algorithm 2) with m={m} disjoint pairs ...")
    pairs = mmck.preselection_multiclass(configs, df_train, m, NBCPUS, KFOLD)
    print(f"  → {len(pairs)} pairs selected")
    return pairs


# ---------------------------------------------------------------------------
# 4. ERROR MATRIX
# ---------------------------------------------------------------------------

def compute_error_matrix(df_train: pd.DataFrame,
                         pairs: list[str],
                         m: int) -> pd.DataFrame:
    """
    Compute the k-fold cross-validation error matrix for the preselected pairs.

    The error matrix has shape (n_samples + 2, n_pairs):
      - Rows 0..n_samples-1 : per-sample absolute errors |y_i - ŷ_i|
      - Row 'MAE'            : full-data MAE (no CV) for each pair
      - Row 'MAE-CVE'        : mean k-fold CV error for each pair

    Columns are sorted by MAE-CVE then MAE, so the best classifier is the
    leftmost column. When two configurations of the same gene pair appear,
    only the best one (by MAE-CVE) is kept — this deduplication is handled
    inside error_matrix_multiclass.

    Parameters
    ----------
    df_train : pd.DataFrame
    pairs    : list[str]   — preselected configurations
    m        : int         — used only for the output filename

    Returns
    -------
    pd.DataFrame — the sorted error matrix
    """
    print(f"\nComputing error matrix for {len(pairs)} pairs ...")
    er = mmck.error_matrix_multiclass(df_train, pairs, NBCPUS, KFOLD)
    print(f"  → Error matrix shape: {er.shape}")
    print(f"  → Best pairs (MAE-CVE):\n{er.loc['MAE-CVE'].head(5).to_string()}")

    output_path = f"error_matrix_m{m}.csv"
    er.to_csv(output_path)
    print(f"  → Saved to {output_path}")

    return er


# ---------------------------------------------------------------------------
# 5. BUILD ENSEMBLE AND PREDICT
# ---------------------------------------------------------------------------

def build_ensemble_and_predict(df_train: pd.DataFrame,
                                df_test:  pd.DataFrame,
                                error_matrix: pd.DataFrame,
                                k: int) -> tuple[list, list]:
    """
    Build an ensembleMBMC of k disjoint pairs and predict test-set labels.

    Ensemble construction (Section 2.2.3 of the paper):
      1. Sort pairs by MAE-CVE (ascending).
      2. Greedily pick the next best pair that shares no gene with already
         selected pairs — this ensures disjointness.
      3. Repeat until k pairs are chosen.

    Prediction:
      - Each MBMC pair outputs a class label for each test sample.
      - The ensemble prediction is the majority vote across the k classifiers.
      - In case of a tie, the prediction is biased towards the worst outcome
        (class 0 = short relapse), following the paper's convention.

    Parameters
    ----------
    df_train     : pd.DataFrame — training data (used to fit the ensemble)
    df_test      : pd.DataFrame — test data
    error_matrix : pd.DataFrame — from compute_error_matrix()
    k            : int          — ensemble size

    Returns
    -------
    (predictions, true_labels)
        predictions  : list[int] — predicted class for each test sample
        true_labels  : list[int] — ground-truth class for each test sample
    """
    print(f"\nBuilding ensemble with k={k} pairs ...")
    pairs = mmck.find_k_ensemble_model(df_train, error_matrix, k, NBCPUS)
    print(f"  → Selected pairs: {pairs}")

    preds, probas = mmck.create_and_predict_ensemble_model(
        df_train, df_test, pairs, NBCPUS
    )

    true_labels = df_test["target"].tolist()
    print(f"  → Predictions: {preds}")
    print(f"  → True labels: {true_labels}")

    return preds, true_labels


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    print("=" * 60)
    print("STEP 2: MBMC PIPELINE")
    print("=" * 60)

    # 1. Load data
    df_train, df_test = load_data("breast_cancer_train.csv", "breast_cancer_test.csv")

    # 2. Generate all pair configurations
    configs = generate_configurations(df_train)

    # 3. Run for each value of m (MBMC-5, MBMC-10, MBMC-20)
    all_results = {}

    for m in M_VALUES:
        print(f"\n{'='*40}")
        print(f"MBMC-{m} scenario")
        print(f"{'='*40}")

        # Preselection: find pairs worth evaluating with full CV
        pairs = run_preselection(configs, df_train, m)

        # Error matrix: k-fold CV MAE for each preselected pair
        er = compute_error_matrix(df_train, pairs, m)

        # For each ensemble size, build ensemble and record predictions
        scenario_results = {}
        for k in range(1, MAX_K + 1):
            preds, true_labels = build_ensemble_and_predict(
                df_train, df_test, er, k
            )
            scenario_results[k] = {"predictions": preds, "true_labels": true_labels}

        all_results[m] = scenario_results

    # 4. Save predictions to CSV
    rows = []
    for m, scenario in all_results.items():
        for k, result in scenario.items():
            for pred, true in zip(result["predictions"], result["true_labels"]):
                rows.append({"m": m, "k": k, "pred": pred, "true": true})

    pd.DataFrame(rows).to_csv("ensemble_predictions.csv", index=False)
    print("\nSaved all predictions to ensemble_predictions.csv")
    print("\nDone. Proceed to step3_evaluate.py")
