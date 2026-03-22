"""
Step 1 — Data Preparation for MBMC Breast Cancer Analysis (METABRIC)
=====================================================================
This script reproduces the preprocessing pipeline used in the paper:
  "Identification of Monotonically Classifying Pairs of Genes for Ordinal Disease Outcomes"
  Section 2.1.3 — Breast Cancer Dataset.

Pipeline overview:
  1. Load raw transcriptomics and clinical data from the METABRIC cohort (cBioPortal)
  2. Align samples: keep only patients present in both datasets
  3. Restrict to patients who experienced recurrence (RFS event = 1, n ≈ 630)
  4. Discretize the continuous RFS time into 3 ordered classes (short / mid / long)
  5. Apply MAD filtering on the training set to reduce dimensionality (~1,708 genes)
  6. Produce a stratified 80/20 train/test split with equal class sizes
  7. Save two ready-to-use CSV files: train and test

Expected input files (download from https://www.cbioportal.org/study/summary?id=brca_metabric):
  - data_mrna_illumina_microarray.txt   (gene expression, samples × genes after transposition)
  - data_clinical_patient.txt           (clinical covariates)

Output files (written to the current working directory):
  - breast_cancer_train.csv
  - breast_cancer_test.csv

Both files have shape (n_samples, n_genes + 1) where the last column is 'target'
with values 0 (short RFS), 1 (intermediate RFS), or 2 (long RFS).

Usage:
  python step1_prepare_data.py

Dependencies: pandas, numpy, scikit-learn, scipy
"""

import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import StratifiedShuffleSplit
from scipy.stats import median_abs_deviation


# ---------------------------------------------------------------------------
# 1. LOAD RAW DATA
# ---------------------------------------------------------------------------

def load_transcriptomics(path: str) -> pd.DataFrame:
    """
    Load the METABRIC microarray expression file.

    The raw file is formatted as genes × samples (with two leading annotation
    columns: Hugo_Symbol and Entrez_Gene_Id). After transposition the index
    becomes patient IDs and columns become gene symbols.

    Parameters
    ----------
    path : str
        Path to data_mrna_illumina_microarray.txt

    Returns
    -------
    pd.DataFrame, shape (n_patients, n_genes)
    """
    df = pd.read_csv(path, sep="\t", index_col=0, low_memory=False)

    # Drop the Entrez gene ID row and transpose so rows = patients
    df = df.T.drop("Entrez_Gene_Id")

    # Convert expression values to numeric (some may be strings after transposition)
    df = df.apply(pd.to_numeric, errors="coerce")

    print(f"Transcriptomics loaded: {df.shape[0]} patients × {df.shape[1]} genes")
    return df


def load_clinical(path: str) -> pd.DataFrame:
    """
    Load the METABRIC clinical patient file.

    The cBioPortal format has 4 metadata header rows before the actual data.
    We skip them and use the 5th row as the header.

    Parameters
    ----------
    path : str
        Path to data_clinical_patient.txt

    Returns
    -------
    pd.DataFrame with patient IDs as index
    """
    # Skip the 4 comment/metadata rows that start with '#'
    clin = pd.read_csv(path, sep="\t", index_col=0, comment="#")
    print(f"Clinical data loaded: {clin.shape[0]} patients × {clin.shape[1]} variables")
    return clin


# ---------------------------------------------------------------------------
# 2. ALIGN SAMPLES AND SELECT RECURRENT PATIENTS
# ---------------------------------------------------------------------------

def align_and_filter(transcriptomics: pd.DataFrame,
                     clinical: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Keep only the patients that are present in both the expression matrix and
    the clinical table, and that have complete values for the variables of
    interest (Relapse Free Status and Relapse Free Status (Months)).

    Parameters
    ----------
    transcriptomics : pd.DataFrame
    clinical        : pd.DataFrame

    Returns
    -------
    (transcriptomics_clean, clinical_clean) — aligned DataFrames
    """
    # Intersection of sample IDs
    common_ids = transcriptomics.index.intersection(clinical.index)
    transcriptomics = transcriptomics.loc[common_ids]
    clinical = clinical.loc[common_ids]

    # Drop samples with missing expression values
    transcriptomics = transcriptomics.dropna()
    clinical = clinical.loc[transcriptomics.index]

    # Drop samples with missing RFS information
    rfs_cols = ["Relapse Free Status", "Relapse Free Status (Months)"]
    clinical = clinical.dropna(subset=rfs_cols)
    transcriptomics = transcriptomics.loc[clinical.index]

    print(f"After alignment and NA removal: {transcriptomics.shape[0]} patients")
    return transcriptomics, clinical


def select_recurrent_patients(transcriptomics: pd.DataFrame,
                              clinical: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Restrict to patients who experienced a recurrence event.

    The paper (Section 2.1.3) focuses on the n = 630 patients for whom
    the relapse event was observed, because the ordinal MBMC method requires
    observed outcomes for all samples (censored data cannot be handled).

    Parameters
    ----------
    transcriptomics : pd.DataFrame
    clinical        : pd.DataFrame

    Returns
    -------
    (expr, rfs_months)
        expr       : expression DataFrame restricted to recurrent patients
        rfs_months : Series of RFS times (in months) for those patients
    """
    # Encode "1:Recurred" → 1, "0:Not Recurred" → 0
    rfs_event = clinical["Relapse Free Status"].replace(
        {"1:Recurred": 1, "0:Not Recurred": 0}
    )

    # Keep only patients who relapsed
    recurrent_mask = rfs_event == 1
    expr = transcriptomics.loc[recurrent_mask]
    rfs_months = clinical.loc[recurrent_mask, "Relapse Free Status (Months)"]

    print(f"Recurrent patients: {expr.shape[0]}")
    print(f"RFS (months) distribution:\n{rfs_months.describe().round(1)}")
    return expr, rfs_months


# ---------------------------------------------------------------------------
# 3. DISCRETIZE RFS INTO 3 ORDINAL CLASSES
# ---------------------------------------------------------------------------

def discretize_rfs(rfs_months: pd.Series, n_classes: int = 3) -> pd.Series:
    """
    Divide continuous RFS time into n_classes equally sized ordinal categories.

    The paper uses 3 classes (short / intermediate / long relapse) created by
    splitting patients into terciles of RFS time so that each group contains
    the same number of patients.

    Labels:
        0 → short-term relapse  (fastest relapsers)
        1 → intermediate-term relapse
        2 → long-term relapse   (latest relapsers)

    Parameters
    ----------
    rfs_months : pd.Series
        Continuous RFS time values.
    n_classes  : int
        Number of ordinal classes (default 3).

    Returns
    -------
    pd.Series of integer class labels {0, 1, ..., n_classes-1}
    """
    labels = pd.qcut(rfs_months, q=n_classes, labels=False)
    labels = labels.astype(int)

    print(f"Class distribution after discretization:")
    for cls, count in sorted(Counter(labels).items()):
        names = {0: "short", 1: "intermediate", 2: "long"}
        print(f"  Class {cls} ({names.get(cls, cls)}): {count} patients")

    return labels


# ---------------------------------------------------------------------------
# 4. STRATIFIED TRAIN / TEST SPLIT
# ---------------------------------------------------------------------------

def stratified_split(expr: pd.DataFrame,
                     target: pd.Series,
                     test_size: float = 0.20,
                     random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame,
                                                       pd.Series, pd.Series]:
    """
    Create an 80/20 stratified train/test split that preserves the class
    proportions across both sets (Section 2.1 of the paper).

    The paper reports that each class has 105 samples in both training and
    testing sets. With 3 classes × 105 × 2 = 630 total patients and a 50/50
    split per class this is consistent. Adjust test_size if needed to match
    your exact sample counts.

    Parameters
    ----------
    expr         : pd.DataFrame  — expression matrix (patients × genes)
    target       : pd.Series     — integer class labels
    test_size    : float         — fraction reserved for testing (default 0.20)
    random_state : int

    Returns
    -------
    (X_train, X_test, y_train, y_test)
    """
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    train_idx, test_idx = next(splitter.split(expr, target))

    X_train = expr.iloc[train_idx]
    X_test  = expr.iloc[test_idx]
    y_train = target.iloc[train_idx]
    y_test  = target.iloc[test_idx]

    print(f"\nTrain set: {X_train.shape[0]} samples")
    print(f"  Class distribution: {dict(sorted(Counter(y_train).items()))}")
    print(f"Test  set: {X_test.shape[0]} samples")
    print(f"  Class distribution: {dict(sorted(Counter(y_test).items()))}")

    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# 5. MAD-BASED GENE FILTERING
# ---------------------------------------------------------------------------

def mad_filter(X_train: pd.DataFrame,
               X_test: pd.DataFrame,
               mad_percentile: float = 0.92) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Retain only the most variably expressed genes using the Median Absolute
    Deviation (MAD) computed on the training set.

    MAD is preferred over variance because it is more robust to outliers,
    which are frequent in transcriptomic data. Gene selection is performed
    exclusively on the training set to avoid data leakage: the same set of
    genes is then applied to the test set.

    The paper (Section 2.1) retains 1,708 genes after filtering with a
    percentile threshold of ~0.92 on the METABRIC data.

    Parameters
    ----------
    X_train       : pd.DataFrame  — training expression matrix
    X_test        : pd.DataFrame  — test expression matrix
    mad_percentile : float        — keep genes above this MAD percentile
                                    (on the training set). Values in [0, 1].

    Returns
    -------
    (X_train_filtered, X_test_filtered) with the same selected gene columns.
    """
    # Compute MAD for every gene using the training set only
    mad_scores = X_train.apply(
        lambda col: median_abs_deviation(col.dropna()), axis=0
    )

    # Select genes whose MAD exceeds the threshold percentile
    threshold = np.percentile(mad_scores, mad_percentile * 100)
    selected_genes = mad_scores[mad_scores >= threshold].index

    X_train_f = X_train[selected_genes]
    X_test_f  = X_test[selected_genes]

    print(f"\nMAD filtering (percentile ≥ {mad_percentile:.0%}):")
    print(f"  Genes before filtering: {X_train.shape[1]}")
    print(f"  Genes after  filtering: {X_train_f.shape[1]}")

    return X_train_f, X_test_f


# ---------------------------------------------------------------------------
# 6. SAVE DATASETS
# ---------------------------------------------------------------------------

def save_datasets(X_train: pd.DataFrame, y_train: pd.Series,
                  X_test:  pd.DataFrame, y_test:  pd.Series,
                  train_path: str = "breast_cancer_train.csv",
                  test_path:  str = "breast_cancer_test.csv") -> None:
    """
    Save train and test datasets as CSV files ready for the MBMC pipeline.

    The MBMC code (individual_BMC.py, ensemble_BMC.py) expects a CSV where:
      - columns are gene names
      - the last column is named 'target' with integer class labels
      - the index contains patient IDs

    Parameters
    ----------
    X_train / X_test : expression DataFrames (after MAD filtering)
    y_train / y_test : target Series
    train_path / test_path : output file paths
    """
    df_train = X_train.copy()
    df_train["target"] = y_train

    df_test = X_test.copy()
    df_test["target"] = y_test

    df_train.to_csv(train_path)
    df_test.to_csv(test_path)

    print(f"\nSaved train → {train_path}  ({df_train.shape[0]} × {df_train.shape[1]} columns)")
    print(f"Saved test  → {test_path}   ({df_test.shape[0]} × {df_test.shape[1]} columns)")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # ---- Paths (edit to match your local download) -------------------------
    TRANSCRIPTOMICS_FILE = "brca_metabric/data_mrna_illumina_microarray.txt"
    CLINICAL_FILE        = "brca_metabric/data_clinical_patient.txt"

    # MAD percentile threshold: 0.917 ≈ top 8.3% ≈ 1,708 genes out of ~20,600
    MAD_PERCENTILE = 0.917

    # ---- Pipeline ----------------------------------------------------------
    print("=" * 60)
    print("STEP 1: DATA PREPARATION")
    print("=" * 60)

    # 1. Load raw data
    expr = load_transcriptomics(TRANSCRIPTOMICS_FILE)
    clin = load_clinical(CLINICAL_FILE)

    # 2. Align samples and remove incomplete cases
    expr, clin = align_and_filter(expr, clin)

    # 3. Select only patients with observed recurrence
    expr, rfs_months = select_recurrent_patients(expr, clin)

    # 4. Discretize RFS into 3 ordinal classes
    target = discretize_rfs(rfs_months, n_classes=3)

    # 5. 80/20 stratified split (train selection computed before filtering)
    X_train, X_test, y_train, y_test = stratified_split(
        expr, target, test_size=0.20, random_state=42
    )

    # 6. MAD filtering — fitted on training set, applied to both
    X_train, X_test = mad_filter(X_train, X_test, mad_percentile=MAD_PERCENTILE)

    # 7. Save ready-to-use datasets
    save_datasets(X_train, y_train, X_test, y_test,
                  train_path="breast_cancer_train.csv",
                  test_path="breast_cancer_test.csv")

    print("\nDone. Proceed to step2_run_mbmc.py")
