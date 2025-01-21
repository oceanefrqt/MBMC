from monotonic_classifier import compute_recursion

import numpy as np
import multiprocessing as mp
import copy
import pandas as pd
from itertools import combinations
from copy import deepcopy




# CREATION OF THE MULTICLASS MONOTONIC MODEL
class Data:
    def __init__(self, data):
        self.data = data
        self.coord, self.weights, self.labels = zip(*self.data)
        self.classes = sorted(list(set(self.labels)))
        self.concordance_labels = {}
        self.concordance_coordinates = {}
        self.concordance_weights = {}
        
    def concordance(self):
        self.concordance_labels = defaultdict(set)
        self.concordance_coordinates = defaultdict(set)
        self.concordance_weights = defaultdict(set)
        
        for c, i in zip(self.labels, self.coord):
            self.concordance_labels[c].add(i)
            
        for c, i in zip(self.coord, self.labels):
            self.concordance_coordinates[c] = i
            
        for c, i in zip(self.coord, self.weights):
            self.concordance_weights[c] = i
            
            
class MultiClassMonotonicClassifier:
    def __init__(self, D):
        self.data = D
        self.seps = {c : {'a':[], 'pt':[]} for c in self.data.classes}
            
            
            
    def DAC(self, new_data, case, classes, k):

        sub1 = classes[:k+1]
        sub2 = classes[k+1:]


        if len(sub1) > 0 and len(sub2) > 0:

            z = [classes[k], classes[k+1]]

            if len(new_data)>0:

                m = compute_recursion(new_data, z, case)
                re, bpr, bpb, r_p, b_p = m[case[2]]




                if len(sub1) == 1:
                    self.seps[classes[k]]['pt'] = b_p
                    self.seps[classes[k]]['a'] = bpb
                if len(sub2) == 1:
                    self.seps[classes[k+1]]['pt'] = r_p
                    self.seps[classes[k+1]]['a'] = bpr


                if len(sub1)%2 == 1:
                    mid1 = len(sub1)//2
                else:
                    mid1 = len(sub1)//2 - 1

                if len(sub2)%2 == 1:
                    mid2 = len(sub2)//2
                else:
                    mid2 = len(sub2)//2 - 1

                data1 = [((p),self.data.concordance_weights[p],self.data.concordance_coordinates[p]) for p in b_p]
                data2 = [((p),self.data.concordance_weights[p],self.data.concordance_coordinates[p]) for p in r_p]

                if len(sub1) > 1:
                    self.DAC(data1, case, sorted(sub1), mid1)

                if len(sub2) > 1:
                    self.DAC(data2, case, sorted(sub2), mid2)



    def clean_seps(self):
        for c in self.seps.keys():
            self.seps[c]['a'] = list(set(self.seps[c]['a']))
            self.seps[c]['pt'] = list(set(self.seps[c]['pt']))
            
            
#### COMPUTATION OF THE ERROR MATRIX
                
def vals_mp(pairs, df_2, out):
    vals = list()
    for p in pairs:
        vals.append((p, df_2, out))
    return vals



def monotonic_model_CE_multiclass(p, df):
    """
    Parameters:
    - p: pair of features.
    - df: dataframe with data.
    
    Returns:
    Tuple with the pair and the classification error (not l1-norm).
    """
    p1, p2, key = p.split('/')
    key = int(key)
    rev, up = equiv_to_key[key]
    
    diag = df['target'].values.tolist()
    tr1, tr2 = df[p1].values.tolist(), df[p2].values.tolist()
    data = [((tr1[n], tr2[n] ), 1, diag[n]) for n in range(len(diag))]
    
    D = Data(data)
    D.concordance()

    mcmc = MultiClassMonotonicClassifier(D)
    if len(set(diag))%2 == 1:
        mid = len(set(diag))//2
    else:
        mid = len(set(diag))//2 - 1       


    mcmc.DAC(D.data, (rev, up, key), D.classes, mid)

    mcmc.clean_seps()
    
    
    CE = 0
    for c1 in D.classes:
        real_c = D.concordance_labels[c1]
        pred_c = mcmc.seps[c1]['pt']
        CE += len(set(real_c) - set(pred_c))
        
    CE /= len(df)
    
    return (p, CE)



def monotonic_model_MAE_multiclass(p, df):
    """
    Parameters:
    - p: pair of features.
    - df: dataframe with data.
    
    Returns:
    Tuple with the pair and the l1 error.
    """
    p1, p2, key = p.split('/')
    key = int(key)
    rev, up = equiv_to_key[key]
    
    diag = df['target'].values.tolist()
    tr1, tr2 = df[p1].values.tolist(), df[p2].values.tolist()
    data = [((tr1[n], tr2[n] ), 1, diag[n]) for n in range(len(diag))]

    D = Data(data)
    D.concordance()

    mcmc = MultiClassMonotonicClassifier(D)
    if len(set(diag))%2 == 1:
        mid = len(set(diag))//2
    else:
        mid = len(set(diag))//2 - 1       


    mcmc.DAC(D.data, (rev, up, key), D.classes, mid)

    mcmc.clean_seps()
    

    l1E = 0
    for c1 in D.classes:
        real_c = D.concordance_labels[c1]
        for c2 in D.classes:
            pred_c = mcmc.seps[c2]['pt']
            nb = len(set(real_c).intersection(set(pred_c)))
            l1E += nb*abs(c1-c2)

    l1E /= len(data)
    return (p, l1E)


def pred_multiclass(out, seps, key):
    """
    Parameters:
    - out: left-out sample.
    - seps: dictionary containing the separation points for each class.
    - key: key of the configuration.
    
    Returns:
    Predicted class.
    """
    
    classes = sorted(list(seps.keys()), reverse=True)
    
    for c in classes:
        flag = False
        
        for pt_front in seps[c]['a']:
            if key == 2:
                if pt_front[0] >= out[0] and pt_front[1] <= out[1]:
                    flag = True
            elif key == 1:
                if pt_front[0] <= out[0] and pt_front[1] <= out[1]:
                    flag = True
            elif key == 3:
                if pt_front[0] >= out[0] and pt_front[1] >= out[1]:
                    flag = True
            elif key == 4:
                if pt_front[0] <= out[0] and pt_front[1] >= out[1]:
                    flag = True
        if flag == True:
            return c
    
    return classes[-1]

def monotonic_model_LOOCVE_multiclass(p, df, kfold):
    
    """
    Parameters:
    - p: pair of features.
    - df: dataframe with data.
    - out: left-out sample

    
    Returns:
    Tuple with the pair and the LOOCVE.
    """
    
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    fold_size = len(df_shuffled) // kfold
    
    folds = []
    for i in range(kfold):
        start = i * fold_size
        if i == k - 1:
            end = len(df_shuffled)
        else:
            end = (i + 1) * fold_size
        folds.append(df_shuffled.iloc[start:end])

    
    err = 0
    for i in range(kfold):
        df_test = folds[i]
        df_train = pd.concat([fold for j, fold in enumerate(folds) if j != i], ignore_index=True)
        
        res = LOOCVE_multiclass(p, df_train, df_test)
        
        err += res[1]
    err /= len(df)
    
    return (p, err)





def LOOCVE_multiclass(p, df_train, df_test):
    
    """
    Parameters:
    - p: pair of features.
    - df: dataframe with data.
    - out: left-out sample

    
    Returns:
    Tuple with the pair and the error of the prediction.
    """

    p1, p2, key = p.split('/')
    key = int(key)
    rev, up = equiv_to_key[key]

    diag = df_train['target'].values.tolist()

    tr1, tr2 = df_train[p1].values.tolist(), df_train[p2].values.tolist()

    data = [((tr1[n], tr2[n] ), 1, diag[n]) for n in range(len(diag))]


    D = Data(data)
    D.concordance()

    mcmc = MultiClassMonotonicClassifier(D)
    if len(set(diag))%2 == 1:
        mid = len(set(diag))//2
    else:
        mid = len(set(diag))//2 - 1       


    mcmc.DAC(D.data, (rev, up, key), D.classes, mid)

    mcmc.clean_seps()
    
    error = 0
    for idx in df_test.index:
        out = df_test.loc[idx]
        out_p = (out[p1].values[0], out[p2].values[0])
        pred = pred_multiclass(out_p, mcmc.seps, key)
        
        if abs(out['target'].values[0]-pred) == 0:
            error += 0
        else:
            error += 1    
    return (p, error)
            
    
def monotonic_model_MAE_CVE_multiclass(p, df, kfold):
    
    """
    Parameters:
    - p: pair of features.
    - df: dataframe with data.
    - out: left-out sample

    
    Returns:
    Tuple with the pair and the mean absolute error.
    """

    
    
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    fold_size = len(df_shuffled) // kfold
    
    folds = []
    for i in range(kfold):
        start = i * fold_size
        if i == kfold - 1:
            end = len(df_shuffled)
        else:
            end = (i + 1) * fold_size
        folds.append(df_shuffled.iloc[start:end])

    
    err = 0
    for i in range(kfold):
        df_test = folds[i]
        df_train = pd.concat([fold for j, fold in enumerate(folds) if j != i], ignore_index=True)
        
        res = l1_error_multiclass(p, df_train, df_test)
        
        err += sum(res[1])
    err /= len(df)
    
    return (p, err)





       
        
    



def l1_error_multiclass(p, df_train, df_test):
    
    """
    Parameters:
    - p: pair of features.
    - df: dataframe with data.
    - out: left-out sample

    
    Returns:
    Tuple with the pair and the l1-norm.
    """

    p1, p2, key = p.split('/')
    key = int(key)
    rev, up = equiv_to_key[key]

    diag = df_train['target'].values.tolist()

    tr1, tr2 = df_train[p1].values.tolist(), df_train[p2].values.tolist()

    data = [((tr1[n], tr2[n] ), 1, diag[n]) for n in range(len(diag))]
 
    D = Data(data)
    D.concordance()

    mcmc = MultiClassMonotonicClassifier(D)
    if len(set(diag))%2 == 1:
        mid = len(set(diag))//2
    else:
        mid = len(set(diag))//2 - 1       


    mcmc.DAC(D.data, (rev, up, key), D.classes, mid)

    mcmc.clean_seps()
    
    errors = list()
    for idx in df_test.index:
        out = df_test.loc[idx]
        out_p = (out[p1], out[p2])
        pred = pred_multiclass(out_p, mcmc.seps, key)
        errors.append(abs(out['target']-pred))
        
    return (p, errors)


def error_matrix_multiclass(df, pairs, nbcpus, kfold):
    """
    Parameters:
    - df: dataframe with data.
    - pairs: pairs to evaluate
    - nbcpus: nb of cpus to use for the multiprocessing

    Returns:
    - Error matrix DataFrame.
    """

    pool = mp.Pool(nbcpus)




    
    df_shuffled = df.sample(frac=1, random_state=42)#.reset_index(drop=True)
    fold_size = len(df_shuffled) // kfold
    
    folds = []
    for i in range(kfold):
        start = i * fold_size
        if i == kfold - 1:
            end = len(df_shuffled)
        else:
            end = (i + 1) * fold_size
        folds.append(df_shuffled.iloc[start:end])
    
    dic_pairs = {k: list() for k in pairs}
    targets = list()
    
    index = list()
    
    for i in range(kfold):
        df_test = folds[i]
        df_train = pd.concat([fold for j, fold in enumerate(folds) if j != i], ignore_index=True)
        
        vals = vals_mp(pairs, df_train, df_test)
        data_dict = dict(pool.starmap(l1_error_multiclass, vals, max(1, len(vals) // nbcpus)))
        
        dic_pairs = {key: dic_pairs[key] + data_dict[key] for key in pairs}
        index += list(df_test.index)
        targets += df_test['target'].values.tolist()
        
    dic_pairs['target'] = targets
    mat_err = pd.DataFrame(dic_pairs, index=index)

    vals_re = [(c, df) for c in pairs]
    
    l1d = dict(pool.starmap(monotonic_model_MAE_multiclass, vals_re, max(1,len(vals)//nbcpus)))
    l1d['target'] = np.nan
    l1_s = pd.Series(l1d)
    l1_s.name = 'MAE'

    mat_err_re = pd.concat((mat_err,l1_s.to_frame().T), axis=0)

    err = {col: np.mean(mat_err_re[col][:-1]) for col in pairs}
    err['target'] = np.nan
    err_s = pd.Series(err)
    err_s.name = 'MAE-CVE'
    
    mat_err_final = pd.concat((mat_err_re,err_s.to_frame().T), axis=0)


    mat_err_final.sort_values(axis = 1, by=['MAE-CVE' ,'MAE'], inplace=True)
    
    prefixes = [col.rsplit('/', 1)[0] for col in mat_err_final.columns]

    first_occurrence_columns = {}
    for i, prefix in enumerate(prefixes):
        if prefix not in first_occurrence_columns:
            first_occurrence_columns[prefix] = mat_err_final.columns[i]

    mat_err_final = mat_err_final[first_occurrence_columns.values()]


    return mat_err_final
