"""
Step 3 — Evaluation, Comparison and Kaplan–Meier Validation
============================================================
This script reproduces the evaluation pipeline described in Section 3.3 of
the paper:
  "Identification of Monotonically Classifying Pairs of Genes for Ordinal
   Disease Outcomes"

It covers:
  1. Performance metrics: MAE, Accuracy, MCC, Cohen's Kappa
  2. Comparison against standard classifiers (RF, DT, LR, GP, SVM)
     transformed for ordinal classification (Frank & Hall 2001)
  3. Bootstrap uncertainty estimation (1,000 resamples)
  4. Kaplan–Meier curves and log-rank tests for the ensembleMBMC predictions
  5. Confusion matrix

Usage:
  python step3_evaluate.py

Reads:
  - breast_cancer_train.csv  (from step1)
  - breast_cancer_test.csv   (from step1)
  - ensemble_predictions.csv (from step2)

Writes:
  - metrics_comparison.csv   — performance table for all classifiers
  - kaplan_meier.png         — Kaplan–Meier survival curves

Dependencies: pandas, numpy, scikit-learn, scipy, matplotlib, lifelines
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")   # non-interactive backend (safe on cluster/headless)

from collections import Counter
from scipy.stats import chi2_contingency
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    matthews_corrcoef,
    confusion_matrix,
)
from sklearn.preprocessing import label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier


from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test


# ---------------------------------------------------------------------------
# 1. PERFORMANCE METRICS
# ---------------------------------------------------------------------------

def mean_absolute_error(y_true: list, y_pred: list) -> float:
    """
    Mean Absolute Error for ordinal classification.

    MAE is particularly suited for ordinal data because it respects the
    ordering of class labels: predicting class 2 when the truth is 0 is
    penalized twice as much as predicting class 1 (Section 2.2.1).

    Equal spacing (distance = 1) between consecutive classes is assumed.
    """
    return float(np.mean(np.abs(np.array(y_true) - np.array(y_pred))))


def compute_all_metrics(y_true: list, y_pred: list) -> dict:
    """
    Compute the four metrics reported in Tables 4–7 of the paper:
      - MAE   : Mean Absolute Error (main metric for ordinal data)
      - Acc   : Accuracy (exact match)
      - MCC   : Matthews Correlation Coefficient (multi-class version)
      - CK    : Cohen's Kappa (linearly weighted to respect label ordering)

    Parameters
    ----------
    y_true, y_pred : lists of integer class labels

    Returns
    -------
    dict with keys 'MAE', 'Acc', 'MCC', 'CK'
    """
    return {
        "MAE": round(mean_absolute_error(y_true, y_pred), 3),
        "Acc": round(accuracy_score(y_true, y_pred), 3),
        "MCC": round(matthews_corrcoef(y_true, y_pred), 3),
        "CK":  round(cohen_kappa_score(y_true, y_pred, weights="linear"), 3),
    }


# ---------------------------------------------------------------------------
# 2. ORDINAL TRANSFORMATION OF STANDARD CLASSIFIERS (Frank & Hall 2001)
# ---------------------------------------------------------------------------

def ordinal_transform(X_train, y_train, X_test,
                      base_classifier, n_classes: int = 3):
    """
    Convert any binary classifier into an ordinal classifier using the
    Frank & Hall (2001) decomposition.

    For k ordinal classes the problem is split into k-1 binary problems:
      - Problem 1: P(y > 0)  — predicts whether the label exceeds class 0
      - Problem 2: P(y > 1)  — predicts whether the label exceeds class 1
      ...
      - Problem k-1: P(y > k-2)

    The final prediction is derived from the k-1 binary outputs by finding
    the largest threshold exceeded by the classifier.

    Parameters
    ----------
    X_train, X_test      : array-like features
    y_train              : array-like integer labels
    base_classifier      : fitted or unfitted sklearn estimator (will be cloned)
    n_classes            : number of ordinal classes

    Returns
    -------
    list[int] — predicted class labels
    """
    from sklearn.base import clone

    binary_preds = []

    for threshold in range(n_classes - 1):
        # Binary label: 1 if y > threshold, 0 otherwise
        y_binary = (np.array(y_train) > threshold).astype(int)
        clf = clone(base_classifier).fit(X_train, y_binary)
        binary_preds.append(clf.predict(X_test))

    # Reconstruct ordinal label: count how many thresholds are exceeded
    preds = np.sum(np.column_stack(binary_preds), axis=1).tolist()
    return preds


# ---------------------------------------------------------------------------
# 3. HYPERPARAMETER GRIDS FOR COMPETING CLASSIFIERS
# ---------------------------------------------------------------------------

PARAM_GRIDS = {
    "RF": {
        "n_estimators": [50, 100, 200],
        "max_depth":    [None, 5, 10],
        "min_samples_split": [2, 5],
    },
    "DT": {
        "max_depth":        [None, 5, 10],
        "min_samples_split": [2, 5],
    },
    "LR": {
        "C":          [0.01, 0.1, 1, 10],
        "max_iter":   [500],
    },
    "SVMlinear": {
        "C":          [0.01, 0.1, 1, 10],
        "kernel":     ["linear"],
    },
    "SVMrbf": {
        "C":          [0.1, 1, 10],
        "gamma":      ["scale", "auto"],
        "kernel":     ["rbf"],
    },
}


def run_competing_classifiers(X_train, y_train, X_test, y_test,
                              n_classes: int = 3,
                              n_splits: int = 5) -> pd.DataFrame:
    """
    Train and evaluate 6 competing classifiers with grid-search tuning.

    Each base classifier is adapted for ordinal classification via the
    Frank & Hall (2001) binary decomposition (except GP, which is used
    directly for comparison purposes as a weak baseline).

    Classifiers evaluated (matching the paper):
      RF         — Random Forest
      DT         — Decision Tree
      LR         — Logistic Regression
      GP         — Gaussian Process Classifier (baseline)
      SVMlinear  — SVM with linear kernel
      SVMrbf     — SVM with RBF kernel

    Parameters
    ----------
    X_train, X_test : feature arrays
    y_train, y_test : integer label arrays
    n_classes       : number of ordinal classes
    n_splits        : number of CV folds for grid search

    Returns
    -------
    pd.DataFrame — metrics for each classifier (rows=classifiers, cols=metrics)
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    classifiers = {
        "RF":        RandomForestClassifier(random_state=42),
        "DT":        DecisionTreeClassifier(random_state=42),
        "LR":        LogisticRegression(random_state=42, solver="lbfgs"),
        "GP":        GaussianProcessClassifier(random_state=42),
        "SVMlinear": SVC(random_state=42),
        "SVMrbf":    SVC(random_state=42),
    }

    results = {}

    for name, clf in classifiers.items():
        print(f"  Training {name} ...")

        if name in PARAM_GRIDS:
            # Grid search with 5-fold CV to find best hyperparameters
            gs = GridSearchCV(clf, PARAM_GRIDS[name], cv=cv,
                              scoring="neg_mean_absolute_error", n_jobs=-1)
            gs.fit(X_train, y_train)
            best_clf = gs.best_estimator_
        else:
            best_clf = clf.fit(X_train, y_train)

        if name == "GP":
            # GP used directly (no ordinal decomposition) — serves as a
            # near-chance baseline in high-dimensional, small-sample settings
            y_pred = best_clf.predict(X_test).tolist()
        else:
            # Apply Frank & Hall ordinal decomposition
            y_pred = ordinal_transform(X_train, y_train, X_test,
                                       best_clf, n_classes)

        results[name] = compute_all_metrics(y_test, y_pred)
        print(f"    {results[name]}")

    return pd.DataFrame(results).T


# ---------------------------------------------------------------------------
# 4. BOOTSTRAP UNCERTAINTY ESTIMATION
# ---------------------------------------------------------------------------

def bootstrap_metrics(df_train: pd.DataFrame,
                      df_test:  pd.DataFrame,
                      pairs_fn,
                      n_bootstrap: int = 1000,
                      random_state: int = 0) -> pd.DataFrame:
    """
    Estimate the variability of MBMC performance via bootstrap resampling
    of the training set with a fixed test set (Section 2.3 of the paper).

    For each bootstrap resample:
      1. Resample the training data (with replacement).
      2. Re-select MBMC pairs on the resample.
      3. Train on the resample and predict the fixed test set.
      4. Record all metrics.

    Reports: median and 95% percentile confidence interval across resamples.

    Parameters
    ----------
    df_train    : pd.DataFrame  — full training set
    df_test     : pd.DataFrame  — fixed test set
    pairs_fn    : callable      — function(df_resample) → (preds, true_labels)
                                  should return predictions on df_test
    n_bootstrap : int           — number of bootstrap resamples (paper: 1000)
    random_state : int

    Returns
    -------
    pd.DataFrame with rows [median, ci_lower, ci_upper] and metric columns
    """
    rng = np.random.default_rng(random_state)
    records = []

    for b in range(n_bootstrap):
        if b % 100 == 0:
            print(f"  Bootstrap {b}/{n_bootstrap} ...")

        # Resample training data with replacement
        idx = rng.integers(0, len(df_train), size=len(df_train))
        df_resample = df_train.iloc[idx]

        try:
            preds, true_labels = pairs_fn(df_resample)
            metrics = compute_all_metrics(true_labels, preds)
            records.append(metrics)
        except Exception:
            # Skip resamples that fail (e.g. degenerate cases)
            continue

    df_boot = pd.DataFrame(records)
    summary = pd.DataFrame({
        "median":   df_boot.median(),
        "ci_lower": df_boot.quantile(0.025),
        "ci_upper": df_boot.quantile(0.975),
    }).T

    return summary


# ---------------------------------------------------------------------------
# 5. KAPLAN–MEIER CURVES AND LOG-RANK TEST
# ---------------------------------------------------------------------------

def kaplan_meier_validation(df_test:    pd.DataFrame,
                             y_pred:     list,
                             clinical:   pd.DataFrame,
                             rfs_col:    str = "Relapse Free Status (Months)",
                             event_col:  str = "Relapse Free Status",
                             output_path: str = "kaplan_meier.png") -> None:
    """
    Validate ensemble predictions using Kaplan–Meier survival curves.

    The ensembleMBMC produces an ordinal prediction (0/1/2) for each test
    patient. We group test patients by their predicted class and plot separate
    Kaplan–Meier curves for each group. Significant separation between curves
    (log-rank p < α = 0.05/3 ≈ 0.0167 after Bonferroni correction for 3
    pairwise tests) validates that the predicted groups have genuinely
    different relapse dynamics (Section 3.3.1 of the paper).

    Parameters
    ----------
    df_test      : pd.DataFrame  — test dataset (index = patient IDs)
    y_pred       : list[int]     — predicted class labels for test patients
    clinical     : pd.DataFrame  — full clinical table (used for RFS time/event)
    rfs_col      : str           — column name for RFS time (months)
    event_col    : str           — column name for RFS event indicator
    output_path  : str           — path where the figure is saved

    Notes
    -----
    If lifelines is not installed this function prints a warning and returns.
    """
    if not LIFELINES_AVAILABLE:
        print("Skipping Kaplan–Meier (lifelines not installed).")
        return

    # Align clinical data with test set predictions
    test_ids = df_test.index.tolist()
    clin_test = clinical.loc[test_ids].copy()
    clin_test["predicted_class"] = y_pred

    # Encode event indicator (1:Recurred → 1)
    clin_test["event"] = clin_test[event_col].replace(
        {"1:Recurred": 1, "0:Not Recurred": 0}
    ).astype(int)

    # ---- Kaplan–Meier curves -----------------------------------------------
    class_labels = {0: "Predicted Short RFS", 1: "Predicted Mid RFS", 2: "Predicted Long RFS"}
    colors = {0: "#e74c3c", 1: "#2ecc71", 2: "#3498db"}

    fig, ax = plt.subplots(figsize=(8, 5))

    kmf = KaplanMeierFitter()
    for cls in sorted(clin_test["predicted_class"].unique()):
        mask = clin_test["predicted_class"] == cls
        group = clin_test.loc[mask]
        kmf.fit(
            group[rfs_col],
            event_observed=group["event"],
            label=class_labels.get(cls, f"Class {cls}"),
        )
        kmf.plot_survival_function(ax=ax, color=colors.get(cls), ci_show=True)

    ax.set_xlabel("Time (months)")
    ax.set_ylabel("Relapse-Free Status")
    ax.set_title("ensembleMBMC — Kaplan–Meier curves (test set)")

    # ---- Pairwise log-rank tests -------------------------------------------
    # Bonferroni-corrected significance level: α = 0.05 / 3 ≈ 0.0167
    alpha_corrected = 0.05 / 3
    class_pairs = [(0, 1), (0, 2), (1, 2)]
    pair_names = ["short vs mid", "short vs long", "mid vs long"]
    p_values = []

    for (c1, c2), pair_name in zip(class_pairs, pair_names):
        mask = clin_test["predicted_class"].isin([c1, c2])
        subset = clin_test.loc[mask]
        result = multivariate_logrank_test(
            durations=subset[rfs_col],
            groups=subset["predicted_class"],
            event_observed=subset["event"],
        )
        p = result.p_value
        p_values.append(p)
        sig = "✓" if p < alpha_corrected else "✗"
        print(f"  Log-rank {pair_name}: p = {p:.4f}  {sig} (α = {alpha_corrected:.4f})")

    # Annotate plot with p-values
    p_text = "\n".join(
        [f"p({n}) = {p:.4f}" for n, p in zip(pair_names, p_values)]
    )
    ax.text(0.02, 0.05, p_text, transform=ax.transAxes,
            fontsize=8, verticalalignment="bottom",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Kaplan–Meier figure saved to {output_path}")


# ---------------------------------------------------------------------------
# 6. CONFUSION MATRIX
# ---------------------------------------------------------------------------

def print_confusion_matrix(y_true: list, y_pred: list,
                            class_names: list = None) -> pd.DataFrame:
    """
    Display the confusion matrix as a labelled DataFrame.

    Rows represent true labels; columns represent predicted labels.
    The paper's Table 8 shows the METABRIC confusion matrix revealing a bias
    towards the intermediate class (class 1) due to majority voting in the
    ensemble architecture.

    Parameters
    ----------
    y_true, y_pred : lists of integer class labels
    class_names    : list[str] — optional label names for display

    Returns
    -------
    pd.DataFrame — confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    labels = class_names or [f"Class {i}" for i in range(cm.shape[0])]
    df_cm = pd.DataFrame(cm, index=labels, columns=labels)
    df_cm.index.name   = "True label"
    df_cm.columns.name = "Predicted label"
    print("\nConfusion Matrix:")
    print(df_cm.to_string())
    return df_cm


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    print("=" * 60)
    print("STEP 3: EVALUATION AND VALIDATION")
    print("=" * 60)

    # ---- 1. Load data ------------------------------------------------------
    df_train = pd.read_csv("breast_cancer_train.csv", index_col=0)
    df_test  = pd.read_csv("breast_cancer_test.csv",  index_col=0)
    df_train.dropna(inplace=True)
    df_test.dropna(inplace=True)

    X_train = df_train.drop(columns=["target"]).values
    y_train = df_train["target"].tolist()
    X_test  = df_test.drop(columns=["target"]).values
    y_test  = df_test["target"].tolist()

    # ---- 2. Load MBMC predictions from step2 -------------------------------
    preds_df = pd.read_csv("ensemble_predictions.csv")

    # Pick the ensembleMBMC scenario: m=5, k=3 (matching the paper)
    # The paper builds an ensemble of 3 MBMCs on the breast cancer dataset
    ensemble_preds = preds_df[(preds_df["m"] == 5) & (preds_df["k"] == 3)]
    y_pred_mbmc = ensemble_preds["pred"].tolist()
    y_true_mbmc = ensemble_preds["true"].tolist()

    print("\n--- ensembleMBMC performance ---")
    mbmc_metrics = compute_all_metrics(y_true_mbmc, y_pred_mbmc)
    for metric, val in mbmc_metrics.items():
        print(f"  {metric}: {val}")

    # ---- 3. Competing classifiers ------------------------------------------
    print("\n--- Competing classifiers ---")
    clf_metrics = run_competing_classifiers(
        X_train, y_train, X_test, y_test, n_classes=3
    )
    clf_metrics.loc["ensembleMBMC"] = mbmc_metrics

    print("\nPerformance summary:")
    print(clf_metrics.to_string())
    clf_metrics.to_csv("metrics_comparison.csv")
    print("\nSaved to metrics_comparison.csv")

    # ---- 4. Confusion matrix -----------------------------------------------
    print_confusion_matrix(
        y_true_mbmc, y_pred_mbmc,
        class_names=["Short RFS", "Mid RFS", "Long RFS"]
    )

    # ---- 5. Kaplan–Meier validation ----------------------------------------
    # We need the original clinical data to get RFS times
    # Adjust the path to wherever your clinical file is stored
    CLINICAL_FILE = "brca_metabric/data_clinical_patient.txt"
    if os.path.exists(CLINICAL_FILE):
        clinical = pd.read_csv(CLINICAL_FILE, sep="\t", index_col=0, comment="#")
        print("\n--- Kaplan–Meier validation ---")
        kaplan_meier_validation(
            df_test=df_test,
            y_pred=y_pred_mbmc,
            clinical=clinical,
            output_path="kaplan_meier.png",
        )
    else:
        print(f"\nSkipping Kaplan–Meier: clinical file not found at {CLINICAL_FILE}")

    print("\nAll done.")
