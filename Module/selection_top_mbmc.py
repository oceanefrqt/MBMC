from mbmcimport *
from collections import defaultdict

import numpy as np
import multiprocessing as mp
import copy
import pandas as pd
from itertools import combinations
from copy import deepcopy

### PRESELECTION

def H_df(df, cls, nbcpus):
    pool = mp.Pool(nbcpus)
    vals = [(c, df) for c in cls]
    res = pool.starmap(monotonic_model_MAE_multiclass, vals, max(1, len(vals) // nbcpus))
    pool.close()
    return sorted(res)

def H_dict(H):
    Hd = dict()
    for h in H:
        kh = h[1]
        if kh not in Hd.keys():
            Hd[kh] = list()
        Hd[kh].append(h[0])
    return Hd


def Q_df(df, cls, nbcpus, kfold):
    pool = mp.Pool(nbcpus)
    vals = [(c, df, kfold) for c in cls]
    S = sorted(pool.starmap(monotonic_model_MAE_CVE_multiclass, vals, max(1, len(vals) // nbcpus)))
    pool.close()
    return S


def Q_dict(Q):
    Qd = dict()
    G = dict()
    for q in Q:
        kq = q[1]
        g1, g2, g3 = q[0].split('/')
        if kq not in Qd.keys():
            Qd[kq] = set()
        if kq not in G.keys():
            G[kq] = set()
        Qd[kq].add(q[0])
        G[kq].add(g1)
        G[kq].add(g2)


    return Qd, G


def preselection_multiclass(cls, df, m, nbcpus, kfold):

    H = H_df(df, cls, nbcpus)

    Hd = H_dict(H)

    Q = dict() #For each strat of LOOCV, we have a list of pairs

    count = 0

    for h_key in sorted(Hd.keys()):

        pairs = Hd[h_key]
        Q_ = Q_df(df, pairs, nbcpus, kfold) #Compute the LOOCVE of the pairs with RE = h_key
        Qd, Gd = Q_dict(Q_)

        Q = dp.update_dict(Q, Qd)# Update Q to get the pairs grouped according to their LOOCVE


        if dp.check_disjoint_pairs_naive(Q,m):
            break


    a = max(Q.keys()) #Highest value of LOOCV in Q
    Hd = dp.supp_H_above_a(Hd, a)
    Hd = dp.supp_H_below_a(Hd, h_key)
    


    for h_key in sorted(Hd.keys()):

        
        a = max(Q.keys())
        if h_key <= a:

            pairs = Hd[h_key]
            Q_ = Q_df(df, pairs, nbcpus, kfold)
            Qd, Gd = Q_dict(Q_)

            Qd = dp.supp_H_above_a(Qd, a)

            Q = dp.update_dict(Q, Qd)

            Q_ = deepcopy(Q)
            del Q_[a]

            while dp.check_disjoint_pairs_naive(Q_, m):
                

                Q = Q_
                a = max(Q.keys())

                Q_ = deepcopy(Q)
                del Q_[a]
                
    pairs = list()
    for key in Q.keys():
        pairs += Q[key]

    return pairs


def all_configurations(df):
    transcripts = list(df.columns)
    transcripts.remove('target')

    configurations = list()
    for i in range(len(transcripts)):
        for j in range(i+1, len(transcripts)):
            for key in range(1,5):
                configurations.append('/'.join([transcripts[i], transcripts[j], str(key)]))
    return configurations