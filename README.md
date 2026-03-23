# Multi-class Bivariate Monotonic Classifiers (MBMC)

## Introduction

This repository provides a Python implementation of **Multi-class Bivariate Monotonic Classifiers (MBMC)** for classifying and predicting ordinal disease outcomes from transcriptomic data. MBMC extends Bivariate Monotonic Classifiers (BMC), which operate on binary outcomes, to the ordinal multi-class setting.

Each classifier uses only two genes with monotonic decision boundaries, making it visually inspectable and biologically interpretable. The method is described in:

> Fourquet O., Afenteva D., Zhang K., Hautaniemi S., Krejca M. S., Doerr C., Schwikowski B.
> **Identification of Monotonically Classifying Pairs of Genes for Ordinal Disease Outcomes.**
> *Submitted*, 2025.

The preselection algorithm used internally is described in:

> Fourquet O., Krejca M. S., Doerr C., et al.
> **Towards the genome-scale discovery of bivariate monotonic classifiers.**
> *BMC Bioinformatics*, 26:228, 2025.

---

## Possible Use Cases

- Identifying pairs of genes with monotonic relationships with an ordinal disease outcome
- Discovering a gene-pair signature strongly associated with a disease outcome
- Selecting top-performing gene pairs for functional enrichment analysis
- Constructing a gene network from top-performing pairs to gain biological insight

---

## Repository Contents

```
Module/
    mbmc.py                    Core MBMC classifier (Data, MultiClassMonotonicClassifier,
                               error computation, cross-validation)
    selection_top_mbmc.py      Preselection algorithm to identify the top-performing pairs
    visualization_MBMC.py      Visualization of MBMC decision boundaries
    monotonic_classifier.py    Low-level monotonic regression algorithm (Stout 2013)
    dynamic_preselection.py    Disjoint-pair selection utilities
    mappings.py                Key-to-configuration mappings
    metrics.py                 AUC, ROC, confusion matrix, and ensemble diversity metrics
    prediction_functions.py    Point prediction and majority-vote ensemble functions

Example/
    Breast_Cancer_METABRIC/
        step1_prepare_data.py  Data preprocessing and train/test split
        step2_run_mbmc.py      MBMC pipeline (preselection → ensemble → prediction)
        step3_evaluate.py      Performance evaluation and benchmarking
    Glioblastoma_GSE53733/
        step1_prepare_data.py  Data preprocessing and train/test split
        step2_run_mbmc.py      MBMC pipeline (preselection → ensemble → prediction)
        step3_evaluate.py      Performance evaluation and benchmarking
    identification_top_pairs.py  Low-level example: preselection + visualization only
```

### Module descriptions

| File | Role |
|---|---|
| `mbmc.py` | Implements the `Data` class (data container) and `MultiClassMonotonicClassifier` (divide-and-conquer classifier). Also provides the functions for computing MAE, cross-validated MAE, and the full error matrix over a set of candidate pairs. |
| `selection_top_mbmc.py` | Implements the preselection heuristic described in the paper. Generates all gene-pair configurations, sorts them by full-data MAE (lower bound on CV-MAE), and identifies the minimal set of pairs that contains at least `k_max` disjoint pairs. |
| `visualization_MBMC.py` | Plots the 2-D decision regions of an MBMC alongside the scatter plot of the samples. Supports all four monotonicity configurations (keys 1–4). |
| `monotonic_classifier.py` | Low-level implementation of the isotonic regression algorithm of Stout (2013) adapted to the bivariate setting. Not intended to be called directly by end users. |
| `dynamic_preselection.py` | Utility functions for maintaining and pruning the priority queue of candidate pairs during preselection (`update_dict`, `check_disjoint_pairs_naive`, `supp_H_above_a`, `supp_H_below_a`). |
| `mappings.py` | Dictionaries mapping integer keys (1–4) to the `(rev, up)` boolean pair that encodes the four monotonicity directions. |
| `metrics.py` | AUC, ROC curve, confidence interval, confusion matrix, accuracy, and ensemble diversity measures (disagreement, double-fault, entropy). |
| `prediction_functions.py` | Point prediction functions for binary classifiers (with uncertainty and class-favouring variants) and majority-vote ensemble aggregation. |

---

## Requirements

Python 3.8 or later. Install the dependencies with:

```bash
pip install -r requirements.txt
```

The dependencies are:

| Package | Purpose |
|---|---|
| `numpy` | Numerical arrays |
| `pandas` | Data loading and manipulation |
| `matplotlib` | Visualization |

`multiprocessing`, `itertools`, `collections`, `copy`, `os`, `sys`, `math`, `heapq`, `random`, and `time` are all part of the Python standard library and do not need to be installed.

---

## Example Usage

Two end-to-end examples are provided under `Example/`.  Both follow the same
three-step structure: data preparation → MBMC pipeline → evaluation.

---

### Example 1 — Breast Cancer METABRIC

**Dataset:** METABRIC cohort (~630 patients with relapse, 1 708 genes after MAD
filtering).  The `target` column encodes relapse-free survival: 0 = short,
1 = intermediate, 2 = long.

**Download:** [cBioPortal — METABRIC](https://www.cbioportal.org/study/summary?id=brca_metabric)
(`data_mrna_illumina_microarray.txt` + `data_clinical_patient.txt`)

```bash
cd Example/Breast_Cancer_METABRIC/

# Step 1 — preprocessing (produces breast_cancer_train.csv / test.csv)
python step1_prepare_data.py

# Step 2 — run the MBMC pipeline
python step2_run_mbmc.py --nbcpus 4 --max-k 10 --kfold 5

# Step 3 — evaluate results
python step3_evaluate.py
```

**Key parameters for step 2:**

| Flag | Default | Description |
|---|---|---|
| `--nbcpus` | 4 | Number of parallel CPU cores |
| `--max-k` | 10 | Maximum ensemble size to evaluate |
| `--kfold` | 5 | Number of cross-validation folds |
| `--train` | `breast_cancer_train.csv` | Training CSV from step 1 |
| `--test` | `breast_cancer_test.csv` | Test CSV from step 1 |

**Output:** `ensemble_predictions.csv` with columns `true` and `pred` for each
test sample.

---

### Example 2 — Glioblastoma GSE53733

**Dataset:** GSE53733 (70 primary glioblastoma samples, ~1 837 genes after MAD
filtering).  The `target` column encodes overall survival: 0 = short-term
(<12 months), 1 = intermediate, 2 = long-term (>36 months).

**Download:** [GEO — GSE53733](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE53733)

```bash
cd Example/Glioblastoma_GSE53733/

# Step 1 — preprocessing (produces DATA/GSE53733_MAD_train.csv / test.csv)
python step1_prepare_data.py

# Step 2 — run the MBMC pipeline
python step2_run_mbmc.py --nbcpus 4 --max-k 5 --kfold 5

# Step 3 — evaluate results
python step3_evaluate.py
```

**Key parameters for step 2:**

| Flag | Default | Description |
|---|---|---|
| `--nbcpus` | 4 | Number of parallel CPU cores |
| `--max-k` | 5 | Maximum ensemble size to evaluate |
| `--kfold` | 5 | Number of cross-validation folds |
| `--train` | `DATA/GSE53733_MAD_train.csv` | Training CSV from step 1 |
| `--test` | `DATA/GSE53733_MAD_test.csv` | Test CSV from step 1 |

**Output:** `ensemble_predictions.csv` with columns `true` and `pred` for each
test sample.

---

### Selecting the Top-Performing Gene Pairs (low-level API)

If you only need the preselection step (e.g. for enrichment analysis or network
construction), you can call the Module directly:

```python
import sys
sys.path.append("../")

import pandas as pd
from Module import selection_top_mbmc as stm

nbcpus = 2   # number of CPUs for multiprocessing
kfold  = 5   # number of folds for cross-validation
k_max  = 10  # minimum number of disjoint pairs to identify

data = pd.read_csv('your_data.csv', index_col=0)

# Generate all gene-pair configurations (4 monotonicity directions per pair)
cls = stm.all_configurations(data)

# Run the preselection algorithm
pairs = stm.preselection_multiclass(cls, data, k_max, nbcpus, kfold)

print(pairs)
```

The `k_max` parameter controls the scope of the analysis:

| Goal | Suggested `k_max` |
|---|---|
| Gene-pair signature | 5–10 |
| Functional enrichment analysis | 25–100 (50–200 genes) |
| Gene network construction | >50 |

For a full end-to-end run including error matrix computation and visualization, see [Example/identification_top_pairs.py](Example/identification_top_pairs.py).

---

## Citation

If you use this code, please cite:

```bibtex
@article{fourquet2025mbmc,
  title   = {Identification of Monotonically Classifying Pairs of Genes for Ordinal Disease Outcomes},
  author  = {Fourquet, Oc\'{e}ane and Afenteva, Daria and Zhang, Kaiyang and
             Hautaniemi, Sampsa and Krejca, Martin S. and Doerr, Carola and
             Schwikowski, Benno},
  journal = {Submitted},
  year    = {2025}
}
```

For the preselection algorithm specifically, please also cite:

```bibtex
@article{fourquet2025genomescale,
  title   = {Towards the genome-scale discovery of bivariate monotonic classifiers},
  author  = {Fourquet, Oc\'{e}ane and Krejca, Martin S. and Doerr, Carola and others},
  journal = {BMC Bioinformatics},
  volume  = {26},
  pages   = {228},
  year    = {2025}
}
```
