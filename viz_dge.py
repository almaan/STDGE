#!/usr/bin/env python3


import numpy as np
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath('../'))

from dataloading.dataloader import STsection, joint_matrix
from utils import min_dist

import umap

#%%
cntdir = '/home/alma/ST-2018/CNNp/DGE/data/count_data/her2nonlum'
metadir = '/home/alma/ST-2018/CNNp/DGE/data/curated_feature_files/her2nonlum'
cnt_list = [os.path.join(cntdir,x) for x in os.listdir(cntdir)]
meta_list = [os.path.join(metadir,x) for x in os.listdir(metadir)]
#%%
gl = "/home/alma/ST-2018/BC_Stanford/scripts/dge/20190612223115355577-DGE-dist.tsv"
gl = pd.read_csv(gl, sep = '\t', header = 0, index_col = 0)
gl = gl.iloc[gl['beta'].values < 0,:]
gl = gl.head(20).index.tolist()
um = umap.UMAP(n_components=3) 

for cp,mp in zip(cnt_list[0:5],meta_list[0:5]):
        section = STsection(cp,mp)
        
#        cvals = um.fit_transform(section.cnt[gl].values)
#        mx = cvals.max(axis=0).reshape(1,-1)
#        mn = cvals.min(axis=0).reshape(1,-1)
#        cvals = (cvals - mn) / (mx - mn)
#        rgba = np.ones((cvals.shape[0],4))
#        rgba[:,0:3] = cvals
#        f,a = section.plot_custom(cvals)        
        section.plot_meta('tumor',var_type = 'cat')
        section.plot_gene(gl)

#%%
