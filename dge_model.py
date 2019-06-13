#!/usr/bin/env python3


import sys
import os
import datetime
import re

import pandas as pd
import numpy as np
import argparse as arp

sys.path.append(os.path.abspath('../'))

from dataloading.dataloader import STsection, joint_matrix
from utils import min_dist

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests as mpt

import mygene 

# Functions ---------------------------------------------

def gene_dge(data,
             formula,
             hypothesis):
    
        
        family = sm.families.NegativeBinomial()
        mod1 = smf.glm(formula = formula,
                       data = data,
                       family = family)
        
        mod1 = mod1.fit()
        wtres = mod1.wald_test(hypothesis)
        
        result = dict(pval = wtres.pvalue.item(),
                   beta = mod1.params['min_dist'],
                   ci_low = mod1.conf_int().loc['min_dist'][0],
                   ci_high = mod1.conf_int().loc['min_dist'][1],
                   )
        return result
    
def distance_dependency(cnt,
                        meta,
                        feature = None,
                        value = None,
                        n_genes = 500,
                        formula = 'y ~ section + size + min_dist',
                        hypothesis = '(min_dist = 0)',
                        ):
    
        
    if all([feature,value]):
        sel = meta.index[(meta[feature] ==  value)]
    else:
        sel = meta.index
    
    if n_genes > cnt.shape[1] or n_genes < 1 or not n_genes:
        n_genes = cnt.shape[1]
    
    srt = np.argsort(np.sum(cnt.values,axis = 0))[::-1]
    srt = srt[0:n_genes]
    genelist = cnt.columns[srt].values.tolist()
    
    metacols = formula.split('~')[1].replace(' ','').split('+')
    varnames = ['y'] + metacols
    data_base = meta[metacols].loc[sel]
    
    data_base['min_dist'] = data_base['min_dist'].abs()
    data_base['min_dist'] -= data_base['min_dist'].min()
    
    colnames = pd.Index(['pval',
                         'adjpval',
                         'beta',
                         'ci_low',
                         'ci_high'])
    
    results = pd.DataFrame(index = pd.Index(genelist),
                           columns = colnames)
    
    for gene in genelist:
        try:
            data = pd.concat([cnt.loc[sel][gene],data_base],axis = 1)
            data.columns = varnames
            results.loc[gene] = gene_dge(data,
                                         formula,
                                         hypothesis)
        except ZeroDivisionError:
            results = results.drop(gene, axis = 0)
        
    adjpval = mpt(results['pval'].values,
                  alpha = 0.01,
                  method = 'fdr_bh')
    
    adjpval = adjpval[1].astype(np.float)
    results['adjpval'] = adjpval
    
    results = results.sort_values('adjpval')
    
    return results

def ensmbl2hgnc(ensmbl):
    noambig = [x if 'ambig' not in x else None \
               for x in ensmbl ]
    
    mg = mygene.MyGeneInfo()
    query = mg.getgenes(noambig, fields = 'symbol')
    symbols = [x['symbol'] if 'symbol' in x else \
               'None' for x in query]
    return symbols


def main(cnt_list,
         meta_list,
         tag,
         outdir,
         n_genes = 500,
         feature = None,
         value = None,
         dlim = 0):
    
    eps = 10e-3
    sections = []
    for cp,mp in zip(cnt_list,meta_list):
        section = STsection(cp,mp)
        dichotomous = all([x in section.meta['tumor'].values\
                       for x in ['tumor','non']])
        
        if dichotomous: 
            min_dist(section,'tumor','non','tumor')
            sup_lim = np.where(np.abs(section.meta['min_dist'].values) > dlim + eps)[0]
            section.subset(sup_lim,ax = 0)

            dichotomous = all([x in section.meta['tumor'].values\
                           for x in ['tumor','non']])
            
                
        
            if dichotomous and section.S > 1:
                rep = section.meta['replicate'].values.astype(str)
                pat = section.meta['patient'].values.astype(str)
                ui = np.array(['_'.join([p,r]) for p,r in zip(pat,rep)])
                section.update_meta(ui,'section')
                
                sf = np.log(section.cnt.values.sum(axis = 1))
                section.update_meta(sf,'size')
                
                sections.append(section)
    
    joint = joint_matrix(sections)
    
    res = distance_dependency(joint['count_matrix'],
                              joint['meta_data'],
                              feature = feature,
                              value = value,
                              n_genes = n_genes,
                              )
    
    res['symbol'] = ensmbl2hgnc(res.index.tolist())    
    oname = os.path.join(outdir,'-'.join([timestamp,'DGE','dist.tsv']))
    res.to_csv(oname,
               sep = '\t',
               header = True,
               index = True,
               )
    

# Parser -----------------------------

if __name__ == '__main__':
    
    prs = arp.ArgumentParser()
    
    prs.add_argument('-c','--count_matrix',
                     required = True,
                     nargs = '+',
                     help = '',
                     )


    prs.add_argument('-m','--meta_data',
                     required = True,
                     nargs = '+',
                     help = '',
                     )
    
    
    prs.add_argument('-o','--outdir',
                     required = False,
                     default = None,
                     nargs = 1,
                     help = '',
                     )
    
    prs.add_argument('-n','--n_genes',
                     required = False,
                     default = None,
                     type = int,
                     help = '',
                     )
    
    prs.add_argument('-f','--feature',
                     type = str,
                     default  = None,
                     help = '',
                     )
    
    prs.add_argument('-v','--value',
                     type = str,
                     default  = None,
                     help = '',
                     )
    
    
    prs.add_argument('-dl','--dlim',
                     type = float,
                     default  = 0,
                     help = '',
                     )
    
    
    args = prs.parse_args()
    timestamp = str(datetime.datetime.today())
    timestamp = re.sub(':| |-|\.','',timestamp)
    
    
    if not isinstance(args.count_matrix,list):
        clist = [args.count_matrix]
    else:
        clist = args.count_matrix
        
    if not isinstance(args.meta_data,list):
        mlist = [args.meta_data]
    else:
        mlist = args.meta_data
    
    if not args.outdir:
        outdir = os.path.abspath(os.path.curdir)
    else:
        outdir = outdir
    
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    
    input_args = dict(cnt_list = clist,
                      meta_list = mlist,
                      tag = timestamp,
                      outdir = outdir,
                      n_genes = args.n_genes,
                      feature = args.feature,
                      value = args.value,
                      dlim = args.dlim,
                      )
    
    print(input_args)
    
    main(**input_args)