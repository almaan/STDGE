#!/usr/bin/env python3


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse as arp

import os

from QuickST.data import STsection

from enrich import enrichment_score
from utils import  comp_list


def main(clist : list,
         mlist : list,
         outname : list,
         genes : pd.DataFrame,
         mass_proportion,
         alpha : float = 0.01,
         n_cols : float = 3,
         increasing = False,
         ):

    
    genes = genes.iloc[genes['adjpval'].values < alpha,:]
    print(f'using threshold {alpha} for adjusted pvalue')
    if increasing:
        genes = genes.iloc[genes['beta'].values > 0,:]
        sign = 'positive'
    else:
        genes = genes.iloc[genes['beta'].values < 0,:]
        sign = 'negative'
        
    print(' '.join([f'using genes with',
                        f'{sign} distance coef']))
    
    genes = genes.index.tolist()
    
    n_samples = len(clist)
    n_cols = int(np.min((n_cols,n_samples)))
    n_rows = int(np.ceil(n_samples / n_cols))
    
    figsize = (6 * n_cols, 6 * n_rows)
    fig, ax = plt.subplots(n_rows,
                           n_cols,
                           sharex = True,
                           sharey = True,
                           figsize = figsize,
                           )
    ax = ax.flatten()
    max_e = -np.inf
    min_e = np.inf

    section_list = list()
    enrichment_list = list()

    
    for num in range(n_samples):
            print(f'Analyzing sample {num+1}/{n_samples}')
            print(f'count file >> {clist[num]}')
            print(f'meta file >> {mlist[num]}')
            section = STsection.STsection(clist[num],mlist[num])
            enrichment = enrichment_score(section.cnt,
                                  genes,
                                  mass_proportion,
                                 )
            
            section_list.append(section)
            enrichment_list.append(enrichment)
            
            if np.max(enrichment) > max_e:
                max_e = enrichment.max()
            if np.min(enrichment) < min_e:
                min_e = enrichment.min()
            
    for num in range(n_samples):
            section_list[num].plot_custom(enrichment_list[num],
                                mark_feature='tumor',
                                mark_val='tumor',
                                marker_size = 80,
                                eax = ax[num],
                                vmin = min_e,
                                vmax = max_e,
                                )
            
            ax[num].set_title('_'.join([section_list[num].patient,
                                        section_list[num].replicate]))
            ax[num].set_aspect('equal')
            ax[num].set_xticks([])
            ax[num].set_yticks([])
            
            for spine in ax[num].spines.values():
                spine.set_visible(False)
        
            section_list[num] = None
            
    for bnum in range(n_samples,n_cols*n_rows):
        fig.delaxes(ax[bnum])
    
    fig.savefig(outname)
            

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
    
    prs.add_argument('-g','--genelist',
                     required = True,
                     help = '',
                     )
    prs.add_argument('-i','--increasing',
                     default = False,
                     action = 'store_true',
                     help = '',
                     )
    
    prs.add_argument('-o','--outdir',
                     type = str,
                     help = '',
                     )
    
    
    prs.add_argument('-t','--tag',
                     type = str,
                     default = None,
                     help = '',
                     )
    
    prs.add_argument('-a','--alpha',
                     type = float,
                     default = 0.01,
                     help = '',
                     )
    
    prs.add_argument('-p','--mass_proportion',
                     type = float,
                     default = 0.95,
                     help = '',
                     )
    
    
    
    args = prs.parse_args()
    
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
        outdir = args.outdir
    
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    
    if args.tag:
        idf = args.tag
    else:
        idf = str(int(np.random.random()*10000))
    
    outname = os.path.basename(args.genelist).replace('.tsv',f'.{idf}-plot.png')
    outname = os.path.join(args.outdir,outname)
    
    clist.sort()
    mlist.sort()
    
    
    input_match = comp_list(clist,mlist)
    
    if input_match:
        print('Input lists are likely equally sorted')
    else:
        print('Input lists are likely not equally sorted')

    genes = pd.read_csv(args.genelist,
                        sep = '\t',
                        header = 0,
                        index_col = 0)
    
    input_args = dict(clist = clist,
                      mlist = mlist,
                      genes = genes,
                      outname = outname,
                      mass_proportion = args.mass_proportion,
                      alpha = args.alpha,
                      )
    
    main(**input_args)