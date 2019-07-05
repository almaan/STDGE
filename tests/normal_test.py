#!/usr/bin/env python3

# Small test to see wheter mean sums genes are normally distributed

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

#%%
def SampleVals(x,
              n_samples,
              n_genes):

    if len(x.shape) < 2:
        reshape = True
        shape = x.shape
        x = x.reshape(1,-1)
    else:
        reshape = False

    vals = np.zeros((x.shape[0],n_samples))
    for samp in range(n_samples):
        genes = np.floor(np.random.random(n_genes) * x.shape[1]).astype(int)
        vals[:,samp] = x[:,genes].mean(axis = 1)

    if reshape:
        x = x.reshape(shape)
        vals = vals.reshape(-1,)

    return vals

pth = "/home/alma/ST-2018/CNNp/DGE/data/mapped_count_data/her2nonlum/count_data-23567_D2.map.underspot.tsv"

cnt = pd.read_csv(pth, sep = '\t', header = 0, index_col = 0)
cnt = cnt.values
data = cnt
#%%
n_genes = 500
n_samples = 10000
vals = SampleVals(data,n_samples,n_genes)
nrm = np.random.normal(0,1, vals.shape[1])
nrm.sort()

pos = 100
qvals = vals[pos,:]
svals = np.sort(qvals)
#svals = (svals - svals.mean()) / svals.std()

stats.probplot(svals, dist="norm", plot=plt)
plt.show()

