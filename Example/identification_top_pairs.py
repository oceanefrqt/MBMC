import sys
import os

sys.path.append("../")
import pandas as pd

from Module import selection_top_mbmc as stm
from Module import visualization as viz

def main():

    nbcpus=2
    kfold = 5
    k_max = 10
    
    path = 'Output/'
    
    os.mkdir(path)


    data = pd.read_csv('GSE53733_data.csv', index_col=0)
    
    
    # List of all the pairs and their configurations
    cls = stm.all_configurations(data)

    # Identification of the best pairs
    pairs = stm.preselection_multiclass(cls, data, m, nbcpus, kfold)
    
    # Computation of their performance (KFOLD CV and MAE)
    em = stm.error_matrix_multiclass(data, pairs, nbcpus, kfold)
    em.to_csv(path + 'performance_top_pairs.csv')
    
    
    # Visualization of the top-pairs
    
    step = 0.01 
    viz.visualization(data, pairs, em, step, path)
    

    
if __name__ == "__main__":
    main()