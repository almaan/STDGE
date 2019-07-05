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


def select_set_full(datum,
                    names,
                    target_set
                    ):
    """select smallest subset with full coverage
    
    Selects the smallest ranked
    subset of genes within a
    spot which contains all of the genes of 
    interest
    
    
    """
    
    sidx = np.fliplr(np.argsort(datum,axis = 1)).astype(int)
    set_list = list()
    for spot in range(sidx.shape[0]):
        snames = [names[x] for x in sidx[spot,:]]
        fidx = [k for k,x in enumerate(snames) if x in target_set]
        fidx = np.max(fidx)
        set_list.append(snames[0:(fidx +1)])
    
    return set_list



def select_set_cumsum(datum,
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

def sampleValues(x,
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
        vals[:,samp] = x[:,genes].sum(axis = 1)

    if reshape:
        x = x.reshape(shape)
        vals = vals.reshape(-1,)

    return vals


def enrichment_score_sampling(cnt : pd.DataFrame,
                             target_set : list,
                             mass_proportion,
                             ):
    """
    Bases enrichment score on comparision to 
    multiple sampled set of genes. Attractive but
    unfeasible
    
    """
    
    inter = cnt.columns.intersection(pd.Index(target_set))
    n_targets = len(target_set)
    nsamples = 100000
    
    # compute sum of target set for all spots
    selscore = cnt.loc[:,inter].values.sum(axis=1).reshape(-1,1)
    
    svals = sampleValues(x = cnt.values,
                         n_genes = n_targets,
                         n_samples = nsamples,
                         )
    
    
    pvals = (selscore < svals).sum(axis=1) /nsamples
    pvals[pvals == 0] = pvals[pvals != 0].min()
    
    return -np.log(pvals).reshape(-1,)
    
    
def enrichment_score_fischer(cnt : pd.DataFrame,
                         target_set : list,
                         mass_proportion : float,
                         ):
    
    """Compute enrichment score
    
    computes the enrichment score for all
    samples (rows) based on a target_set.
    
    """
    
    pvals = []
    query_all = cnt.columns.values
    query_top_list = select_set_cumsum(cnt.values,
                                       query_all,
                                       mass_proportion = mass_proportion,
                                       )
    
    n_in_set = len(set(target_set).intersection(set(query_all)))
    print(f'{n_in_set} / {len(target_set)} of target genes present in data')
    
    full_set =  query_all.tolist() + target_set
    print(f'full-set cardinality {len(set(full_set))}')
    print(f'target-set cardinality {len(set(target_set))}')
    
    for ii in range(len(query_top_list)):
        p = fex(target_set,query_top_list[ii], full_set)
        pvals.append(p)
    
    pvals = np.array(pvals)
    
    return -np.log(pvals)
