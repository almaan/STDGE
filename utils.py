#!/usr/bin/env python3


import numpy as np
from scipy.spatial import distance_matrix

def min_dist(section,
             feature,
             endo_tag,
             exog_tag,
             translate = False,
             ):
    
    
    idx_endo = section.idx_of(feature, endo_tag)
    idx_exog = section.idx_of(feature, exog_tag)
    
    dmat = np.hstack((section.x.reshape(-1,1),section.y.reshape(-1,1)))
    dmat = distance_matrix(dmat,dmat)
    
    min_endo = np.min(dmat[idx_endo,:][:,idx_exog], axis = 1)
    min_exog = np.min(dmat[idx_exog,:][:,idx_endo], axis = 1)
    
    if translate:
        min_endo = min_endo - min_endo.min()
        min_exog = min_exog - min_exog.min()
    
    dvec = np.zeros((section.S,))
    dvec[idx_endo] = min_endo
    dvec[idx_exog] = -1.0*min_exog
    section.update_meta(dvec,'min_dist')
