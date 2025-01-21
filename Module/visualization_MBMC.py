from MBMC import *

import numpy as np
import multiprocessing as mp
import copy
import pandas as pd
from itertools import combinations
from copy import deepcopy

from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt


def visualization(df, pairs, cm, step, path):
    
    for p in pairs:
        print(p)
    
        p1, p2, key = p.split('/')

        key = int(key)
        rev, up = equiv_to_key[key]
        tr1 = df[p1].values.tolist()
        tr2 = df[p2].values.tolist()
        diag = df['target'].values.tolist()
        data = [((tr1[n], tr2[n] ), 1, diag[n]) for n in range(len(diag))]

        classes = sorted(list(set(diag)))

        colormap = plt.cm.viridis
        num_colors = len(classes) 
        discrete_cmap = ListedColormap(colormap(np.linspace(0, 1, num_colors)))
        colors = discrete_cmap.colors
        
        
        D = Data(data)
        D.concordance()

        mcmc = MultiClassMonotonicClassifier(D)
        if len(set(diag))%2 == 1:
            mid = len(set(diag))//2
        else:
            mid = len(set(diag))//2 - 1       


        mcmc.DAC(D.data, (rev, up, key), D.classes, mid)

        mcmc.clean_seps()
        
        seps = mcmc.seps

        X,Y = list(), list()
        for c in seps.keys():
            for pt in seps[c]['pt']:
                X.append(pt[0])
                Y.append(pt[1])
                
        minX = min(X)
        minY = min(Y)
        maxX = max(X)
        maxY = max(Y)
        
        
        X_ = np.arange(minX - 1, maxX + 1, step)
        Y_ = np.arange(minY - 1, maxY + 1, step)


        Xv, Yv = np.meshgrid(sorted(X_), sorted(Y_, reverse=True))
        Z = np.zeros((len(Y_), len(X_)))
        



        for c in sorted(seps.keys(), reverse=True):
            for (x,y) in seps[c]['pt']:
                x = round(x,2)
                y = round(y,2)
                


                if key == 1:

                    for i in range(len(X_)):
                        for j in range(len(Y_)):
                            if x== round(Xv[j,i],2) and y == round(Yv[j,i],2):

                                Z[j,i] = c

                                for l in range(0,j+1):
                                    for m in range(i, len(X_)):
                                        if Z[l,m] == 0:
                                            Z[l,m] =c
                elif key == 3:

                    for i in range(len(X_)):
                        for j in range(len(Y_)):
                            if x== round(Xv[j,i],2) and y == round(Yv[j,i],2):
                                Z[j,i] = c

                                for l in range(j, len(Y_)):
                                    for m in range(0, i+1):
                                        if Z[l,m] == 0:
                                            Z[l,m] =c

                elif key == 4:

                    for i in range(len(X_)):
                        for j in range(len(Y_)):
                            if x== round(Xv[j,i],2) and y == round(Yv[j,i],2):
                                Z[j,i] = c

                                for l in range(j, len(Y_)):
                                    for m in range(i, len(X_)):
                                        if Z[l,m] == 0:
                                            Z[l,m] =c

                elif key == 2:

                    for i in range(len(X_)):
                        for j in range(len(Y_)):
                            if x== round(Xv[j,i],2) and y == round(Yv[j,i],2):
                                Z[j,i] = c

                                for l in range(0,j+1):
                                    for m in range(0, i+1):
                                        if Z[l,m] == 0:
                                            Z[l,m] =c

        plt.figure(figsize=(5,5))

        plt.xlabel(p1)
        plt.ylabel(p2)



        

        try:
            l1CVE = cm.at['MAE-CVE', p]

            plt.title('MBMC\n MAE-CVE = {}'.format(round(l1CVE,2)))
        except:
            pass

        plt.contourf(sorted(X_),sorted(Y_),np.flipud(Z), alpha=0.8)

        my_labels = {c: f"{int(c)}" for c in classes}
        
        if key == 1 or key == 4:

            for d in data:
                plt.scatter(d[0][0], d[0][1], c=colors[int(d[2])], edgecolors='k', label=my_labels[int(d[2])])
                my_labels[int(d[2])] = "_nolegend_"
                
        else:
            for d in data:
                scatter = plt.scatter(d[0][0], d[0][1], c=colors[int(d[2])], edgecolors='k', label=my_labels[int(d[2])])
                my_labels[int(d[2])] = "_nolegend_"
            

        plt.legend(title="Classes")
        plt.savefig(path + f"{p1}_{p2}_MBMC.png", bbox_inches='tight')
        plt.show()