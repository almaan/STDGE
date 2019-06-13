#DGE analysis for ST-data

Similar to what DESeq and edgeR does but better adapted for ST-data. Shrinkage and MAP estimates are not made, since large support for parameter inference exists within ST-data (each spot is treated
as a sample)

The model is simple and based on a GLM, it uses the NB2 - distribution with ```var = mu + mu*alpha^2```. The mean is taken as a
function of different covariates according to:

```
log(mu_i) = beta_[i] * section + gamma_1 * np.log(libsize) + gamma_2 * min_dist

```

where libsize is the total amount of observed counts within a spot, and min_dist is the distance to the nearest spot of a different group than the spot itself (eg nearest distance to a tumor spot for
a non_tumor spot)

A wald test with ```gamma_2 = 0 ``` is used to test the significance of the min_dist parameter
