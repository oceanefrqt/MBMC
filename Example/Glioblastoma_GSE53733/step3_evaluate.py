"""
Step 3 — Evaluation: Glioblastoma GSE53733
==========================================

This script evaluates the ensembleMBMC model and compares it with
standard machine-learning competitors on the GSE53733 test set.

Pipeline
--------
  1. Load the train / test data and the precomputed error matrix
  2. Build the best ensemble MBMC model (k disjoint gene pairs)
  3. Compute performance metrics: MAE, Accuracy, MCC, Cohen's Kappa
  4. Compare with competing classifiers using the Frank & Hall (2001)
     ordinal transformation
  5. Bootstrap uncertainty (1,000 resamples) for the MBMC model
  6. Kaplan-Meier survival curves per predicted class with pairwise
     log-rank tests (Bonferroni-corrected α = 0.05 / 3 ≈ 0.0167)

Metrics
-------
  MAE  (Mean Absolute Error)   — primary metric for ordinal classification;
       penalises predictions proportionally to distance from the true class.
  Acc  (Accuracy)              — fraction of exactly correct predictions.
  MCC  (Matthews Correlation)  — balanced metric, robust to class imbalance.
  CK   (Cohen's Kappa, linear) — agreement above chance, weighted linearly.

Competing classifiers
---------------------
  Every competitor is wrapped in the OrdinalClassifier (Frank & Hall 2001),
  which decomposes the k-class ordinal problem into k-1 binary sub-problems:
    binary_y_i = (y > class_i)   for  i = 0, … , k-2.
  The k-1 binary probabilities are combined to give class probabilities and
  the class with the highest probability is predicted.

  Classifiers tested:
    Decision Tree, Random Forest, SVM (linear & rbf),
    Gaussian Process, Logistic Regression

Usage
-----
  python step3_evaluate.py DATA/GSE53733_MAD_train.csv \
                           DATA/GSE53733_MAD_test.csv  \
                           EM/ER_BMC_5_GSE53733_MAD_train.csv

Dependencies
------------
  pip install pandas numpy scikit-learn scipy lifelines statsmodels matplotlib
  Clean_Module must be on sys.path (handled automatically via CLUSTER_DIR).
"""

import argparse
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from collections import Counter
from sklearn.tree       import DecisionTreeClassifier
from sklearn.ensemble   import RandomForestClassifier
from sklearn.svm        import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model     import LogisticRegression
from sklearn.metrics    import (
    accuracy_score, matthews_corrcoef, cohen_kappa_score,
    mean_absolute_error as sklearn_mae,
)
from sklearn.base       import BaseEstimator, ClassifierMixin, clone
from scipy.special      import expit

# ── Clean_Module path ─────────────────────────────────────────────────────────
CLUSTER_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "Cluster", "Glioblastoma",
)
if os.path.isdir(CLUSTER_DIR):
    sys.path.insert(0, os.path.abspath(CLUSTER_DIR))

import Module.multiclass_monotonic_classifiers_Kfold as mmck

NBCPUS = 4    # increase to 23 on the cluster


# ── Performance metrics ───────────────────────────────────────────────────────

def mean_absolute_error(y_true, y_pred) -> float:
    """Ordinal MAE: mean of |y_true_i - y_pred_i|.

    Unlike accuracy, MAE takes into account the distance between predicted and
    true class.  For a 3-class ordinal problem (0, 1, 2), predicting class 2
    when the true class is 0 is penalised twice as much as predicting class 1.
    """
    return float(np.mean(np.abs(np.array(y_true) - np.array(y_pred))))


def compute_all_metrics(y_true, y_pred, label: str = "") -> dict:
    """Compute MAE, Accuracy, MCC and linear Cohen's Kappa.

    Parameters
    ----------
    y_true : array-like of int
    y_pred : array-like of int
    label  : str
        Optional prefix for print output.

    Returns
    -------
    metrics : dict with keys 'MAE', 'Acc', 'MCC', 'CK'
    """
    mae = mean_absolute_error(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    ck  = cohen_kappa_score(y_true, y_pred, weights="linear")

    if label:
        print(f"{label:<30}  MAE={mae:.4f}  Acc={acc:.4f}  MCC={mcc:.4f}  CK={ck:.4f}")

    return {"MAE": mae, "Acc": acc, "MCC": mcc, "CK": ck}


def print_confusion_matrix(y_true, y_pred,
                            class_names=("short OS", "intermediate OS", "long OS")):
    """Print a labelled confusion matrix."""
    from sklearn.metrics import confusion_matrix
    cm = pd.DataFrame(
        confusion_matrix(y_true, y_pred),
        index   =[f"true_{c}" for c in class_names],
        columns =[f"pred_{c}" for c in class_names],
    )
    print("\nConfusion matrix:")
    print(cm.to_string())


# ── Ordinal classifier wrapper (Frank & Hall 2001) ────────────────────────────

class OrdinalClassifier(BaseEstimator, ClassifierMixin):
    """Ordinal classifier following the Frank & Hall (2001) decomposition.

    Transforms a k-class ordinal problem into k-1 binary problems:
        P(y > class_i)   for  i = 0, … , k-2.

    Class probabilities are recovered as:
        P(y = 0)   = 1 - P(y > 0)
        P(y = i)   = P(y > i-1) - P(y > i)   for 0 < i < k-1
        P(y = k-1) = P(y > k-2)

    The predicted class is the one with the highest probability.
    """

    def __init__(self, clf):
        self.clf          = clf
        self.clfs         = {}
        self.unique_class = None

    def fit(self, X, y):
        self.unique_class = np.sort(np.unique(y))
        if len(self.unique_class) > 2:
            for i in range(len(self.unique_class) - 1):
                binary_y = (y > self.unique_class[i]).astype(np.uint8)
                clf = clone(self.clf)
                clf.fit(X, binary_y)
                self.clfs[i] = clf
        return self

    def _binary_prob(self, clf, X):
        """Return P(y=1) from predict_proba or sigmoid(decision_function)."""
        if hasattr(clf, "predict_proba"):
            return clf.predict_proba(X)[:, 1]
        return expit(clf.decision_function(X))

    def predict_proba(self, X):
        clfs_predict = {i: self._binary_prob(self.clfs[i], X) for i in self.clfs}
        k = len(self.unique_class) - 1
        predicted = []
        for i in range(len(self.unique_class)):
            if i == 0:
                predicted.append(1 - clfs_predict[0])
            elif i < k:
                predicted.append((1 - clfs_predict[i]) * clfs_predict[i - 1])
            else:
                predicted.append(clfs_predict[k - 1])
        return np.vstack(predicted).T

    def predict(self, X):
        return self.unique_class[np.argmax(self.predict_proba(X), axis=1)]


# ── Competing classifiers ──────────────────────────────────────────────────────

def get_competing_classifiers() -> dict:
    """Return a dict of {name: OrdinalClassifier} for all competitors."""
    base = {
        "Decision_Tree":      DecisionTreeClassifier(),
        "Random_Forest":      RandomForestClassifier(),
        "SVM_linear":         SVC(kernel="linear", probability=True),
        "SVM_rbf":            SVC(kernel="rbf",    probability=True),
        "Gaussian_Process":   GaussianProcessClassifier(),
        "Logistic_Regression": LogisticRegression(max_iter=500),
    }
    return {name: OrdinalClassifier(clf) for name, clf in base.items()}


def run_competing_classifiers(X_train, y_train, X_test, y_test) -> pd.DataFrame:
    """Train each competitor and evaluate on the test set.

    Parameters
    ----------
    X_train, y_train : training gene expression and labels
    X_test, y_test   : test gene expression and labels

    Returns
    -------
    results : pd.DataFrame
        One row per classifier, columns: MAE, Acc, MCC, CK.
    """
    classifiers = get_competing_classifiers()
    rows = []

    for name, clf in classifiers.items():
        try:
            clf.fit(X_train.values, y_train.values)
            y_pred = clf.predict(X_test.values)
            metrics = compute_all_metrics(y_test.values, y_pred, label=name)
        except Exception as e:
            print(f"  {name}: FAILED ({e})")
            metrics = {"MAE": np.nan, "Acc": np.nan, "MCC": np.nan, "CK": np.nan}

        rows.append({"Classifier": name, **metrics})

    return pd.DataFrame(rows).set_index("Classifier")


# ── Bootstrap uncertainty ──────────────────────────────────────────────────────

def bootstrap_metrics(df_train: pd.DataFrame, df_test: pd.DataFrame,
                      best_pairs: list, n_bootstrap: int = 1000) -> pd.DataFrame:
    """Estimate confidence intervals for the ensemble MBMC via bootstrap.

    Procedure (following the paper):
      - Resample the training set with replacement (n_bootstrap times)
      - Keep the test set fixed
      - For each resample, rebuild and evaluate the ensemble model
      - Report median ± 95% CI across bootstrap samples

    Parameters
    ----------
    df_train, df_test : DataFrames with 'target' column
    best_pairs        : ensemble gene-pair configurations to use
    n_bootstrap       : number of bootstrap resamples (default: 1000)

    Returns
    -------
    summary : pd.DataFrame
        Rows: MAE, Acc, MCC, CK.  Columns: median, ci_lower, ci_upper.
    """
    print(f"\nBootstrap uncertainty ({n_bootstrap} resamples) …")
    records = {"MAE": [], "Acc": [], "MCC": [], "CK": []}
    y_test  = df_test["target"].tolist()

    for i in range(n_bootstrap):
        # Resample training set with replacement
        boot = df_train.sample(n=len(df_train), replace=True, random_state=i)
        try:
            preds, _ = mmck.create_and_predict_ensemble_model(
                boot, df_test, best_pairs, NBCPUS
            )
            m = compute_all_metrics(y_test, preds)
            for k in records:
                records[k].append(m[k])
        except Exception:
            continue

    summary = {}
    for key, vals in records.items():
        arr = np.array(vals)
        summary[key] = {
            "median":   float(np.nanmedian(arr)),
            "ci_lower": float(np.nanpercentile(arr, 2.5)),
            "ci_upper": float(np.nanpercentile(arr, 97.5)),
        }

    summary_df = pd.DataFrame(summary).T
    print("\nBootstrap summary (median [95% CI]):")
    for metric, row in summary_df.iterrows():
        print(f"  {metric}: {row['median']:.4f} [{row['ci_lower']:.4f}, {row['ci_upper']:.4f}]")

    return summary_df


# ── Kaplan-Meier validation ───────────────────────────────────────────────────

def kaplan_meier_validation(df_test: pd.DataFrame, y_pred,
                             clinical: pd.DataFrame,
                             duration_col: str = "Overall Survival (Months)",
                             event_col: str    = "Overall Survival Status",
                             out_path: str     = "km_curves_glioblastoma.png",
                             alpha_bonferroni: float = 0.05 / 3):
    """Draw Kaplan-Meier curves stratified by predicted class and test for
    significant differences using pairwise log-rank tests.

    The survival outcome for glioblastoma is Overall Survival (OS).
    Predicted classes 0 (short OS) / 1 (intermediate) / 2 (long OS) should
    stratify patients into groups with significantly different survival.

    Bonferroni correction is applied over the 3 pairwise comparisons:
        (0 vs 1),  (0 vs 2),  (1 vs 2)
    so the corrected threshold is  α = 0.05 / 3 ≈ 0.0167.

    Parameters
    ----------
    df_test    : test expression DataFrame (index = sample IDs)
    y_pred     : predicted class list aligned with df_test
    clinical   : DataFrame with survival columns indexed by sample IDs
    duration_col, event_col : column names in `clinical`
    out_path   : figure file path
    alpha_bonferroni : significance threshold after Bonferroni correction

    Returns
    -------
    logrank_results : dict with p-values for each pair
    """
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test

    # Align predictions with clinical data
    test_ids = df_test.index
    pred_series = pd.Series(y_pred, index=test_ids, name="pred_class")

    # Join with clinical survival data
    surv = clinical.loc[
        clinical.index.intersection(test_ids),
        [duration_col, event_col]
    ].copy()
    surv["pred_class"] = pred_series

    # Map event string to binary (1:Deceased → 1, 0:Living → 0)
    if surv[event_col].dtype == object:
        surv[event_col] = surv[event_col].map(
            lambda x: 1 if "1" in str(x) or "Deceased" in str(x) else 0
        )

    class_labels = {0: "Short OS",  1: "Intermediate OS",  2: "Long OS"}
    colors       = {0: "firebrick", 1: "steelblue",        2: "seagreen"}

    fig, ax = plt.subplots(figsize=(6, 4))
    kmf = KaplanMeierFitter()

    for cls in sorted(surv["pred_class"].dropna().unique()):
        mask = surv["pred_class"] == cls
        if mask.sum() < 2:
            continue
        kmf.fit(
            durations=surv.loc[mask, duration_col],
            event_observed=surv.loc[mask, event_col],
            label=f"{class_labels.get(cls, cls)} (n={mask.sum()})",
        )
        kmf.plot_survival_function(ax=ax, ci_show=True, color=colors.get(cls, "grey"))

    ax.set_title("Kaplan-Meier — Predicted OS Classes\nGlioblastoma GSE53733")
    ax.set_xlabel("Overall Survival (months)")
    ax.set_ylabel("Survival probability")
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\nKM curves saved to {out_path}")

    # Pairwise log-rank tests with Bonferroni correction
    pairs_to_test = [(0, 1), (0, 2), (1, 2)]
    logrank_results = {}

    print(f"\nPairwise log-rank tests (Bonferroni α = {alpha_bonferroni:.4f}):")
    for c1, c2 in pairs_to_test:
        mask1 = surv["pred_class"] == c1
        mask2 = surv["pred_class"] == c2
        if mask1.sum() < 2 or mask2.sum() < 2:
            print(f"  class {c1} vs {c2}: not enough samples — skipped")
            continue

        result = logrank_test(
            surv.loc[mask1, duration_col], surv.loc[mask2, duration_col],
            event_observed_A=surv.loc[mask1, event_col],
            event_observed_B=surv.loc[mask2, event_col],
        )
        p = result.p_value
        sig = "*" if p < alpha_bonferroni else "ns"
        print(f"  class {c1} vs {c2}:  p = {p:.4f}  {sig}")
        logrank_results[f"{c1}_vs_{c2}"] = p

    return logrank_results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the ensembleMBMC model on the GSE53733 test set."
    )
    parser.add_argument("train_path",    help="Path to training CSV (from step1)")
    parser.add_argument("test_path",     help="Path to test CSV (from step1)")
    parser.add_argument("error_matrix",  help="Path to ER_BMC_*.csv (from step2)")
    parser.add_argument("--k",           type=int, default=5,
                        help="Ensemble size (default: 5)")
    parser.add_argument("--bootstrap",   action="store_true",
                        help="Run bootstrap uncertainty estimation (slow)")
    parser.add_argument("--clinical",    default=None,
                        help="Path to clinical CSV with OS data (for KM curves)")
    parser.add_argument("--out-dir",     default="./Results/",
                        help="Output directory for figures and tables (default: ./Results/)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────
    df_train = pd.read_csv(args.train_path,   index_col=0, low_memory=False).dropna()
    df_test  = pd.read_csv(args.test_path,    index_col=0, low_memory=False).dropna()
    er       = pd.read_csv(args.error_matrix, index_col=0, low_memory=False)

    y_train = df_train["target"]
    y_test  = df_test["target"]
    X_train = df_train.drop(columns=["target"])
    X_test  = df_test.drop(columns=["target"])

    print(f"Train: {df_train.shape}   Test: {df_test.shape}")
    print(f"Classes — train: {dict(Counter(y_train))}  test: {dict(Counter(y_test))}")

    # ── Build ensemble MBMC model ──────────────────────────────────────────
    print(f"\nBuilding ensemble MBMC (k={args.k}) …")
    best_pairs = mmck.find_k_ensemble_model(df_train, er, args.k, NBCPUS)
    print(f"Selected pairs: {best_pairs}")

    # ── Predict ────────────────────────────────────────────────────────────
    y_pred, probas = mmck.create_and_predict_ensemble_model(
        df_train, df_test, best_pairs, NBCPUS
    )

    # ── MBMC metrics ───────────────────────────────────────────────────────
    print("\n--- MBMC Performance ---")
    mbmc_metrics = compute_all_metrics(y_test.tolist(), y_pred, label="ensembleMBMC")
    print_confusion_matrix(y_test.tolist(), y_pred)

    # ── Competing classifiers ──────────────────────────────────────────────
    print("\n--- Competing Classifiers ---")
    comp_metrics = run_competing_classifiers(X_train, y_train, X_test, y_test)

    # Combine results
    mbmc_row = pd.DataFrame([{"Classifier": "ensembleMBMC", **mbmc_metrics}]).set_index("Classifier")
    all_metrics = pd.concat([mbmc_row, comp_metrics]).sort_values("MAE")
    print("\nFull comparison:")
    print(all_metrics.round(4).to_string())

    # Save comparison table
    comp_path = os.path.join(args.out_dir, "classifier_comparison_GSE53733.csv")
    all_metrics.to_csv(comp_path)
    print(f"\nComparison table saved to {comp_path}")

    # ── Bootstrap ─────────────────────────────────────────────────────────
    if args.bootstrap:
        boot_df = bootstrap_metrics(df_train, df_test, best_pairs, n_bootstrap=1000)
        boot_path = os.path.join(args.out_dir, "bootstrap_results_GSE53733.csv")
        boot_df.to_csv(boot_path)
        print(f"Bootstrap results saved to {boot_path}")

    # ── Kaplan-Meier curves ────────────────────────────────────────────────
    if args.clinical and os.path.isfile(args.clinical):
        clinical = pd.read_csv(args.clinical, index_col=0, low_memory=False)
        km_path  = os.path.join(args.out_dir, "km_curves_GSE53733.png")

        # Common OS column names in GEO clinical data
        duration_col = next(
            (c for c in clinical.columns if "survival" in c.lower() and "month" in c.lower()),
            clinical.columns[0],
        )
        event_col = next(
            (c for c in clinical.columns if "status" in c.lower() or "event" in c.lower()),
            clinical.columns[1],
        )
        print(f"\nKaplan-Meier using: duration='{duration_col}', event='{event_col}'")

        logrank_results = kaplan_meier_validation(
            df_test, y_pred, clinical,
            duration_col=duration_col,
            event_col=event_col,
            out_path=km_path,
        )

        lr_path = os.path.join(args.out_dir, "logrank_results_GSE53733.csv")
        pd.Series(logrank_results, name="p_value").to_csv(lr_path)
        print(f"Log-rank results saved to {lr_path}")
    else:
        print("\nNo clinical data provided — skipping Kaplan-Meier validation.")
        print("  (pass --clinical path/to/clinical.csv to enable it)")

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
