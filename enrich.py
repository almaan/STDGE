#!/usr/bin/env python3

import pandas as pd
import numpy as np
from scipy.special import loggamma


def log_binom(n,k):
    """Numerical stable binomial coeffient"""
    n1 = loggamma(n+1)
    d1 = loggamma(k+1)
    d2 = loggamma(n-k + 1)
    return n1 - d1 -d2

def fex(target_set : list,
        query_set : list,
        full_set : list,
        alpha = 0.05,
        ):
    """Fischer Exact test"""
    
    ts = set(target_set)
    qs = set(query_set)
    fs = set(full_set)
    qs_and_ts = qs.intersection(ts)
    qs_not_ts = qs.difference(ts)
    ts_not_qs = fs.difference(qs).intersection(ts)
    not_ts_not_qs = fs.difference(qs).difference(ts)
    
    x = np.zeros((2,2))
    x[0,0] = len(qs_and_ts)
    x[0,1] = len(qs_not_ts)
    x[1,0] = len(ts_not_qs)
    x[1,1] = len(not_ts_not_qs)
    
    p1 = log_binom(x[0,:].sum(), x[0,0])
    p2 = log_binom(x[1,:].sum(),x[1,0])
    p3 = log_binom(x.sum(), x[:,0].sum())
    
    p = np.exp( p1 + p2 - p3)
    
    return p

def select_set(datum,
               names,
               mass_proportion = 0.95):
    
    """select top genes in set
    
    Selects the top G genes which constitutes
    thrs fraction of the totatl observed counts
    
    """
    
    if len(datum.shape) != 2:
        reshape = True
        datum = datum.reshape(1,-1)
    else:
        reshape = False
        
    sidx = np.fliplr(np.argsort(datum,axis = 1)).astype(int)
    cumsum = np.cumsum(np.take_along_axis(datum,sidx,axis=1),
                       axis = 1)
    
    
    lim = np.max(cumsum,axis = 1) * mass_proportion
    lim = lim.reshape(-1,1)
    
    q = np.argmin(cumsum <= lim,axis = 1)
    set_list = [names[sidx[x,0:q[x]]].tolist() for x \
                in range(datum.shape[0])]
    
    if reshape:
        datum = datum.reshape(-1,)
    
    return set_list


def enrichment_score(cnt : pd.DataFrame,
                     target_set : list,
                     mass_proportion : float,
                     ):
    
    """Compute enrichment score
    
    computes the enrichment score for all
    samples (rows) based on a target_set.
    
    """
    
    pvals = []
    query_all = cnt.columns.values
    query_top_list = select_set(cnt.values,
                                query_all,
                                mass_proportion = mass_proportion,
                                )
    
    for ii in range(len(query_top_list)):
        full_set =  query_all.tolist() + target_set
        p = fex(target_set,query_top_list[ii], full_set)
        pvals.append(p)
    
    pvals = np.array(pvals)
    
    return -np.log(pvals)




