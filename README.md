# Multi-class Bivariate Monotonic Classifiers (MBMC)
==============================================

## Introduction
The aim of this repository is to provide a Python implementation of Multi-class Bivariate Monotonic Classifiers (MBMC) for classifying and predicting ordinal disease outcomes based on transcriptomic data. This is an extension of the Bivariate Monotonic Classifiers (BMC) that are working on binary outcomes.

## Possible Use Cases
MBMC can be used for various purposes, including:

* Identifying pairs of genes with monotonic and ordinal relationships with a disease outcome
* Discovering a signature of gene pairs that are strongly associated with a disease outcome
* Selecting top-performing gene pairs for functional enrichment analysis
* Constructing a gene network based on the top-performing pairs to gain insights into the underlying biological mechanisms

## Repository Contents
This repository contains the implementation of MBMC, including:

* The construction of MBMC using transcriptomic data
* A selection algorithm to identify the top-performing MBMC models

## Requirements
To run the code in this repository, you will need to install the following Python libraries:

* `numpy`
* `pandas`
* `matplotlib`
* `multiprocessing`

You can install these libraries using pip:
```bash
pip install numpy pandas matplotlib multiprocessing
```


## Example Usage: GSE53733 Dataset

### Retrieving and Preprocessing the Data
To demonstrate the usage of MBMC, we will use the GSE53733 dataset. This dataset contains expression data from primary Glioblastoma in adults. It contains 70 samples, including 23 longterm survivors with >36 months overall survival (OS), 16 short-term survivors with <12 months OS, and 31 patients with intermediate OS. The goal is to predict the overall survival (OS) range. 

The data set for this example has been downloaded from the [GEO database](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE53733) and preprocessed as follows:
- Selection of only the protein-coding genes to focus on genes with potential functional relevance for the disease outcome.
- Normalization using TMM (Trimmed Mean of M-values) and log1p transformation to stabilize the variance and make the data more suitable for analysis.
- Filtering using the Median Absolute Deviation (MAD) to remove genes with low variability and improve the performance of the MBMC implementation.

```python
import pandas as pd


data = pd.read_csv('GSE53733_data.csv', index_col=0)
```

The data is in the form of a csv file, with the samples in rows, the genes and the class ('target') in columns.

### Selection of the top-performing pairs of genes

The selection algorithm contains a parameter k_max, i.e. the minimum number of disjoint pairs expected. Depending on the analyses you wish to perform later, this parameter may vary. In fact, to determine a signature of gene pairs, you can expect between 5 and 10 pairs. But for functional analyses, it seems more relevant to identify more than fifty pairs. 

Moreover, two other parameters are the number of CPU for the computation (this code use the library multiprocessing) and the number of fold for the cross-validation. By default they are respectively at 5 and 2.

```python
import sys
import pandas as pd
from Module import selection_top_mbmc as stm

sys.path.append("../")


nbcpus=2
kfold = 5

k_max = 10


data = pd.read_csv('GSE53733_data.csv', index_col=0)
    
    
# List of all the pairs and their configurations
cls = stm.all_configurations(data)

# Identification of the best pairs
pairs = stm.preselection_multiclass(cls, data, m, nbcpus, kfold)
    
print(pairs)

```

For more, refers to identification_top_pairs.py.
