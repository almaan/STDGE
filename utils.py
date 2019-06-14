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



def comp_list(list1,list2):
    """
    Control if the sorting of two lists
    are optimal in terms of similarity
    
    
    """
    
    def maxham(s1,s2):
        abscore = 0.0
        if len(s1) > len(s2):
            major,minor = s1,s2
        else:
            major, minor = s2,s1
        
        w = len(minor)
        for pos in range(len(major) - w +1):
            tscore = hamming(major[pos:pos + w],minor)
            if tscore > abscore:
                abscore = tscore
                
        return abscore
    
    def hamming(x,y):
        hd = 0.0
        for (xs,ys) in zip(x,y):
            if xs == ys:
                hd += 1
                
        return hd

    if not len(list1) == len(list2):
        return False
    else:
        smat = np.zeros((len(list1),len(list2)))
        for l1 in range(len(list1)):
            for l2 in range(len(list2)):
                smat[l1,l2] = maxham(list1[l1],list2[l2])
        
        diag = np.diag(smat)
        mx = np.max(smat, axis = 1)
        
        return (mx == diag).all()
    
