library(ade4)
library(data.table)

sp_mean_df = fread('./TPD/sp_no_neg_union_feat_mean.csv', sep='\t')
sp_mean = sp_mean_df[,3:1026]

dim(sp_mean)

sp_pca = dudi.pca(sp_mean, nf=3, scannf=F)

fam_bca = bca(sp_pca, fac=as.factor(sp_mean_df$family), scannf = F, nf=3)

randtest_fam_bca = randtest(fam_bca, nrep=999)

# comparing our groups to random groups
randtest_fam_bca$pvalue

# cross validating if our groups are real groups
# WARNING: TAKE SUPER LONG TIME (it failed when paralleling, and took more than 5 days to complete with single process)
xvbca = loocv(fam_bca, progress=T)
xvbca
