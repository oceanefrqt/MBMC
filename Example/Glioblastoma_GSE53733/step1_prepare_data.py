"""
Step 1 — Data Preparation: Glioblastoma GSE53733
=================================================

This script prepares the GSE53733 microarray dataset for MBMC (Multi-class
Bivariate Monotonic Classifier) analysis.

Pipeline:
  1. Download raw SOFT data from GEO using GEOparse
  2. Extract survival phenotypes (short / intermediate / long-term OS)
  3. Build the expression matrix and map probe IDs → gene symbols (GPL570)
  4. Average duplicate probes that map to the same gene
  5. Restrict to protein-coding genes (via pybiomart / saved mart file)
  6. Compute the MAD (Median Absolute Deviation) for every gene
  7. Keep genes above the MAD threshold (0.4)
  8. Stratified 80/20 train/test split
  9. Save train.csv and test.csv

Dataset characteristics
-----------------------
  - 70 glioblastoma samples from GEO series GSE53733
  - Platform: Affymetrix Human Genome U133 Plus 2.0 (GPL570)
  - Outcome: overall survival (OS) discretised into 3 ordinal classes
        0  →  short-term OS  (worst prognosis)
        1  →  intermediate OS
        2  →  long-term OS   (best prognosis)
  - After MAD filtering: ~1,837 informative genes remain

Usage
-----
  python step1_prepare_data.py [--destdir DATA_DIR] [--mart MART_CSV] [--mad-threshold 0.4]

Dependencies
------------
  pip install GEOparse pybiomart pandas numpy scikit-learn scipy matplotlib
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless backend for cluster environments
import matplotlib.pyplot as plt

from collections import Counter
from scipy.stats import median_abs_deviation
from sklearn.model_selection import train_test_split


# ── 1. Load raw GEO data ─────────────────────────────────────────────────────

def load_gse(geo_id: str = "GSE53733", destdir: str = "./Data/"):
    """Download (or load from cache) a GEO series using GEOparse.

    GEOparse stores the raw SOFT.gz file in `destdir` on the first call and
    reads it from disk on subsequent calls, so internet access is only needed
    once.

    Returns
    -------
    gse : GEOparse.GSE
        The parsed GEO series object.
    """
    import GEOparse
    print(f"Loading {geo_id} …")
    gse = GEOparse.get_GEO(geo=geo_id, destdir=destdir)
    print(f"  {len(gse.gsms)} samples loaded.")
    return gse


# ── 2. Extract survival phenotypes ───────────────────────────────────────────

def extract_targets(gse) -> pd.Series:
    """Map survival labels to ordinal integers.

    Label mapping
    -------------
      'short-term OS'        → 0   (worst outcome)
      'intermediate OS'      → 1
      'long-term OS'         → 2   (best outcome)

    The ordinal encoding is required by the MBMC algorithm: the classifier
    learns monotone decision boundaries in the feature space that respect the
    ordering  0 < 1 < 2.

    Returns
    -------
    target : pd.Series
        Index = GEO accession strings, values ∈ {0, 1, 2}.
    """
    phen = gse.phenotype_data
    label_col = "characteristics_ch1.0.survival"

    label_map = {
        "short-term OS":    0,
        "intermediate OS":  1,
        "long-term OS":     2,
    }

    target = (
        phen[label_col]
        .map(label_map)
        .rename("target")
    )

    if target.isna().any():
        unknown = phen.loc[target.isna(), label_col].unique()
        raise ValueError(f"Unexpected survival labels: {unknown}")

    print("Class distribution:")
    for cls, count in sorted(Counter(target).items()):
        print(f"  class {cls}: {count} samples")

    return target


# ── 3. Build expression matrix ───────────────────────────────────────────────

def build_expression_matrix(gse) -> pd.DataFrame:
    """Assemble the probe-level expression matrix (samples × probes).

    Each GSM object stores probe intensities in a `table` attribute.  We read
    the VALUE column for every sample and label probes with their ID_REF
    strings from the first sample's table.

    Returns
    -------
    expr : pd.DataFrame
        Shape (n_samples, n_probes), index = GEO accession strings.
    """
    print("Building expression matrix …")
    phen      = gse.phenotype_data
    samples   = phen["geo_accession"].tolist()
    first_gsm = gse.gsms[samples[0]]

    # Probe IDs from the first sample's table (identical across all samples)
    probe_ids = first_gsm.table["ID_REF"].tolist()

    expr = pd.DataFrame(
        {s: gse.gsms[s].table["VALUE"].values for s in samples},
        index=probe_ids,
    ).T                          # transpose → samples as rows

    print(f"  Expression matrix: {expr.shape[0]} samples × {expr.shape[1]} probes")
    return expr


# ── 4. Map probe IDs → gene symbols ──────────────────────────────────────────

def map_probes_to_genes(expr: pd.DataFrame, gse) -> pd.DataFrame:
    """Rename columns from Affymetrix probe IDs to HGNC gene symbols.

    The GPL570 platform table contains the mapping ID → Gene Symbol.  Some
    probes have an empty or NaN gene symbol; those are dropped.  When multiple
    probes map to the same gene, their expression values are averaged in the
    next step.

    Returns
    -------
    expr : pd.DataFrame
        Columns are now gene symbols (may still contain duplicates).
    """
    print("Mapping probe IDs to gene symbols …")
    gpl_table  = gse.gpls["GPL570"].table
    probe_map  = gpl_table.set_index("ID")["Gene Symbol"].to_dict()

    expr = expr.rename(columns=probe_map)

    # Drop probes that did not receive a gene symbol
    expr = expr.drop(columns=[np.nan], errors="ignore")
    expr = expr.loc[:, expr.columns.notna()]
    expr = expr.loc[:, expr.columns != ""]

    print(f"  {expr.shape[1]} gene columns after probe→symbol mapping")
    return expr


# ── 5. Restrict to protein-coding genes ──────────────────────────────────────

def filter_protein_coding(expr: pd.DataFrame, mart_path: str = "mart_export.txt") -> pd.DataFrame:
    """Keep only protein-coding genes listed in a BioMart export.

    The file `mart_export.txt` can be obtained from Ensembl BioMart
    (www.ensembl.org → BioMart) by selecting:
        Dataset  : Homo sapiens genes (GRCh38)
        Attributes: Gene name, Gene type
        Filter   : Gene type = protein_coding

    If the file is absent, pybiomart is used to query Ensembl live (requires
    internet access).

    Returns
    -------
    expr : pd.DataFrame
        Columns restricted to protein-coding genes present in the expression
        matrix.
    """
    if os.path.isfile(mart_path):
        dataset = pd.read_csv(mart_path, sep=",")
    else:
        print("  mart_export.txt not found — querying Ensembl BioMart …")
        from pybiomart import Server
        server  = Server(host="http://www.ensembl.org")
        dataset = (
            server.marts["ENSEMBL_MART_ENSEMBL"]
                  .datasets["hsapiens_gene_ensembl"]
                  .query(
                      attributes=["external_gene_name"],
                      filters={"biotype": "protein_coding"},
                  )
        )
        dataset.columns = ["Gene name"]
        dataset.to_csv(mart_path, index=False)
        print(f"  Saved protein-coding gene list to {mart_path}")

    coding_genes = set(dataset["Gene name"].dropna())
    keep         = [g for g in expr.columns if g in coding_genes]
    expr         = expr[keep]

    print(f"  {len(keep)} protein-coding genes retained")
    return expr


# ── 6. Average duplicate probes ──────────────────────────────────────────────

def average_duplicate_probes(expr: pd.DataFrame) -> pd.DataFrame:
    """Average expression across probes that map to the same gene symbol.

    On the Affymetrix U133 Plus 2.0 array, many genes are interrogated by
    several probe sets.  Averaging gives a single representative value per
    gene and prevents the same gene from being counted multiple times during
    pair-based classification.

    Returns
    -------
    expr : pd.DataFrame
        One column per unique gene symbol, values averaged over all probes.
    """
    n_before = expr.shape[1]
    expr = expr.T.groupby(level=0).mean().T    # group by gene name, average
    n_after  = expr.shape[1]

    print(f"  Probe averaging: {n_before} → {n_after} unique genes")
    return expr


# ── 7. MAD filtering ─────────────────────────────────────────────────────────

def compute_mad(expr: pd.DataFrame, scaling_factor: float = 1.4826) -> pd.Series:
    """Compute the scaled MAD for every gene.

    The MAD (Median Absolute Deviation) is a robust measure of spread:

        MAD_gene = median( |X_gene - median(X_gene)| )

    Multiplying by 1.4826 makes the MAD consistent with the standard deviation
    under normality (the consistency constant for a normal distribution).

    Only training-set samples should be passed here to avoid data leakage.

    Parameters
    ----------
    expr : pd.DataFrame
        Expression matrix (samples × genes).
    scaling_factor : float
        Consistency constant; default 1.4826.

    Returns
    -------
    mad : pd.Series
        Scaled MAD per gene, sorted descending.
    """
    mad = expr.apply(lambda col: median_abs_deviation(col, scale="normal"), axis=0)
    # Alternatively, to match the notebook implementation exactly:
    #   gene_medians = expr.median(axis=0)
    #   mad = expr.sub(gene_medians, axis=1).abs().median(axis=0) * scaling_factor
    mad = mad.sort_values(ascending=False)
    return mad


def mad_filter(expr: pd.DataFrame, mad_threshold: float = 0.4,
               plot: bool = True, plot_path: str = "mad_distribution.png") -> pd.DataFrame:
    """Keep genes whose scaled MAD exceeds `mad_threshold`.

    A fixed threshold of 0.4 (on the log2-intensity scale) was chosen for the
    GSE53733 analysis; it retains ~1,837 genes out of ~16,344 protein-coding
    genes.  Genes with very low variability carry little discriminative
    information and introduce noise.

    Filtering is performed on all samples (the dataset is small: 70 samples).
    When a train/test split is done first, compute MAD on the training set
    only and apply the same gene list to the test set.

    Parameters
    ----------
    expr : pd.DataFrame
        Expression matrix (all samples OR training samples only).
    mad_threshold : float
        Minimum scaled MAD to keep a gene.  Default: 0.4.
    plot : bool
        Whether to save a histogram of the MAD distribution.
    plot_path : str
        Path for the optional histogram figure.

    Returns
    -------
    filtered_expr : pd.DataFrame
        Expression matrix restricted to high-MAD genes.
    mad_genes : pd.Series (selected gene names) — also returned via 2nd position
    """
    mad = compute_mad(expr)

    if plot:
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.hist(mad.values, bins=60, color="steelblue", edgecolor="white")
        ax.axvline(mad_threshold, color="red", linewidth=1.5, label=f"threshold = {mad_threshold}")
        ax.set_title("Distribution of gene MAD\nGlioblastoma (GSE53733)")
        ax.set_xlabel("Scaled MAD")
        ax.set_ylabel("Number of genes")
        ax.legend()
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"  MAD distribution saved to {plot_path}")

    selected = mad[mad > mad_threshold].index.tolist()
    print(f"  MAD filter (threshold={mad_threshold}): {len(selected)} / {len(mad)} genes retained")

    return expr[selected], selected


# ── 8. Stratified train/test split ───────────────────────────────────────────

def stratified_split(expr: pd.DataFrame, target: pd.Series,
                     test_size: float = 0.20,
                     random_state: int = 42):
    """Split into train and test preserving class proportions.

    With 70 samples and 3 balanced classes (~23 per class), a 20 % test set
    gives approximately 4–5 samples per class in the test set.

    Parameters
    ----------
    expr : pd.DataFrame
        Gene expression matrix aligned with `target`.
    target : pd.Series
        Ordinal labels {0, 1, 2}.
    test_size : float
        Fraction of samples for the test set.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    X_train, X_test : pd.DataFrame
    y_train, y_test : pd.Series
    """
    X_train, X_test, y_train, y_test = train_test_split(
        expr, target,
        test_size=test_size,
        stratify=target,
        random_state=random_state,
    )
    print(f"Train: {X_train.shape[0]} samples  {Counter(y_train)}")
    print(f"Test : {X_test.shape[0]}  samples  {Counter(y_test)}")
    return X_train, X_test, y_train, y_test


# ── 9. Save datasets ──────────────────────────────────────────────────────────

def save_datasets(X_train, X_test, y_train, y_test,
                  out_dir: str = "./Data/",
                  prefix: str = "GSE53733_MAD"):
    """Save train and test DataFrames (genes + target column) as CSV files.

    The last column of every saved file is named 'target'; this is the
    convention expected by all downstream MBMC scripts.

    Output files
    ------------
    <out_dir>/<prefix>_train.csv
    <out_dir>/<prefix>_test.csv
    """
    os.makedirs(out_dir, exist_ok=True)

    df_train = pd.concat([X_train, y_train], axis=1)
    df_test  = pd.concat([X_test,  y_test],  axis=1)

    train_path = os.path.join(out_dir, f"{prefix}_train.csv")
    test_path  = os.path.join(out_dir, f"{prefix}_test.csv")

    df_train.to_csv(train_path)
    df_test.to_csv(test_path)

    print(f"Saved: {train_path}  ({df_train.shape})")
    print(f"Saved: {test_path}   ({df_test.shape})")

    return train_path, test_path


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Prepare GSE53733 glioblastoma data for MBMC analysis."
    )
    parser.add_argument("--destdir",       default="./Data/",
                        help="Directory for raw GEO downloads (default: ./Data/)")
    parser.add_argument("--mart",          default="mart_export.txt",
                        help="Path to the BioMart protein-coding gene CSV "
                             "(default: mart_export.txt)")
    parser.add_argument("--mad-threshold", type=float, default=0.4,
                        help="Scaled MAD threshold for gene filtering (default: 0.4)")
    parser.add_argument("--test-size",     type=float, default=0.20,
                        help="Fraction of samples for the test set (default: 0.20)")
    parser.add_argument("--seed",          type=int,   default=42,
                        help="Random seed for the train/test split (default: 42)")
    parser.add_argument("--out-dir",       default="./Data/",
                        help="Output directory for processed CSVs (default: ./Data/)")
    args = parser.parse_args()

    # ── Load GEO data ──────────────────────────────────────────────────────
    gse    = load_gse(destdir=args.destdir)
    target = extract_targets(gse)

    # ── Build expression matrix ────────────────────────────────────────────
    expr = build_expression_matrix(gse)

    # ── Map probes to gene symbols ─────────────────────────────────────────
    expr = map_probes_to_genes(expr, gse)

    # ── Restrict to protein-coding genes ──────────────────────────────────
    expr = filter_protein_coding(expr, mart_path=args.mart)

    # ── Average duplicate probes ───────────────────────────────────────────
    expr = average_duplicate_probes(expr)

    # ── Align samples (expression matrix and targets must share the same index) ──
    common = expr.index.intersection(target.index)
    expr   = expr.loc[common]
    target = target.loc[common]

    # ── Save full processed dataset (before split) ─────────────────────────
    full_df = expr.copy()
    full_df["target"] = target
    coding_path = os.path.join(args.out_dir, "GSE53733_coding_genes.csv")
    os.makedirs(args.out_dir, exist_ok=True)
    full_df.to_csv(coding_path)
    print(f"Full dataset saved to {coding_path}")

    # ── MAD filtering (on all samples — dataset is small: 70 samples) ──────
    #
    # NOTE: For larger datasets, compute MAD on the training set only to avoid
    # data leakage.  Here the same gene set is applied to both splits, which
    # is acceptable given the small sample size and that MAD does not use the
    # target labels.
    expr_filtered, selected_genes = mad_filter(
        expr,
        mad_threshold=args.mad_threshold,
        plot=True,
        plot_path=os.path.join(args.out_dir, "mad_distribution_GSE53733.png"),
    )

    # ── Save MAD-filtered full dataset ─────────────────────────────────────
    mad_full_path = os.path.join(args.out_dir, "GSE53733_MAD.csv")
    mad_df = expr_filtered.copy()
    mad_df["target"] = target
    mad_df.to_csv(mad_full_path)
    print(f"MAD-filtered full dataset saved to {mad_full_path}")

    # ── Stratified 80/20 split ─────────────────────────────────────────────
    X_train, X_test, y_train, y_test = stratified_split(
        expr_filtered, target,
        test_size=args.test_size,
        random_state=args.seed,
    )

    # ── Save train / test CSVs ─────────────────────────────────────────────
    train_path, test_path = save_datasets(
        X_train, X_test, y_train, y_test,
        out_dir=args.out_dir,
        prefix="GSE53733_MAD",
    )

    print("\nData preparation complete.")
    print(f"  Training set: {train_path}")
    print(f"  Test set    : {test_path}")
    print(f"  Features    : {X_train.shape[1]} genes (MAD > {args.mad_threshold})")
    print("\nNext step: run step2_run_mbmc.py with these files.")


if __name__ == "__main__":
    main()
