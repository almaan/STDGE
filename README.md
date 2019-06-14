# DGE analysis for ST-data

Contains tools for analysis and visualization of Spatial-DGE analysis

## Differential Expression

Similar to what DESeq and edgeR does but better adapted for ST-data. Shrinkage and MAP estimates are not made, since large support for parameter inference exists within ST-data (each spot is treated
as a sample)

The model is simple and based on a GLM, it uses the NB2 - distribution with ```var = mu + mu*alpha^2```. The mean is taken as a
function of different covariates according to:

```
log(mu_i) = beta_[i] * section + gamma_1 * np.log(libsize) + gamma_2 * min_dist

```

where libsize is the total amount of observed counts within a spot, and min_dist is the distance to the nearest spot of a different group than the spot itself (eg nearest distance to a tumor spot for
a non_tumor spot)

A wald test with ```gamma_2 = 0 ``` is used to test whether including the covariate min_dist significantly improves the model

## Enrichement

Once a set of genes has been retreived from the DGE analysis, the enrichment within each spot of these genes can calculated. The enrichment score is obatained
by taking the negative log of the pvalue of the fischer exact test
