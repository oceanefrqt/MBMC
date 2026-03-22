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
    mbmc.py                 Core MBMC classifier (Data, MultiClassMonotonicClassifier,
                            error computation, cross-validation)
    selection_top_mbmc.py   Preselection algorithm to identify the top-performing pairs
    visualization_MBMC.py   Visualization of MBMC decision boundaries
    monotonic_classifier.py Low-level monotonic regression algorithm (Stout 2013)
    dynamic_preselection.py Disjoint-pair selection utilities
    mappings.py             Key-to-configuration mappings

Example/
    identification_top_pairs.py  End-to-end example on the GSE53733 glioblastoma dataset
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

## Example Usage: GSE53733 Dataset

### Retrieving and Preprocessing the Data

The example uses the GSE53733 dataset (primary glioblastoma, 70 samples: 23 long-term survivors with >36 months OS, 16 short-term survivors with <12 months OS, 31 intermediate survivors).

Download the raw data from the [GEO database](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE53733) and preprocess as follows:

- Select protein-coding genes only.
- Normalize with TMM (Trimmed Mean of M-values) and apply a log1p transformation.
- Filter with the Median Absolute Deviation (MAD) to remove low-variability genes.

The resulting CSV has samples in rows and genes (plus a `target` column encoding the ordinal class) in columns:

```python
import pandas as pd

data = pd.read_csv('GSE53733_data.csv', index_col=0)
```

### Selecting the Top-Performing Gene Pairs

```python
import sys
sys.path.append("../")

import pandas as pd
from Module import selection_top_mbmc as stm

nbcpus = 2   # number of CPUs for multiprocessing
kfold  = 5   # number of folds for cross-validation
k_max  = 10  # minimum number of disjoint pairs to identify

data = pd.read_csv('GSE53733_data.csv', index_col=0)

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
  journal = {},
  year    = {}
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
