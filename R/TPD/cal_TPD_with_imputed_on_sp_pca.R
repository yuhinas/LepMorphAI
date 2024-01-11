library(data.table)
library(TPD)
# https://cran.r-project.org/web/packages/TPD/TPD.pdf

# override the original TPDs function
source('TPDs.R')

fam_of_novel_group = c('Bombycidae', 'Brahmaeidae', 'Cossidae', 'Drepanidae', 'Endromidae', 
                       'Erebidae', 'Eupterotidae', 'Euteliidae', 'Geometridae', 'Hepialidae', 
                       'Lasiocampidae', 'Limacodidae', 'Lycaenidae', 'Megalopygidae', 'Mimallonidae', 
                       'Noctuidae', 'Nolidae', 'Notodontidae', 'Nymphalidae', 'Papilionidae', 'Pieridae', 
                       'Riodinidae', 'Saturniidae', 'Sematuridae', 'Sphingidae', 'Thyrididae', 'Uraniidae')

##### Data preparation

X = fread('sp_subfam_picked_top2_no_neg_union_pca.csv', header = T)
X= X[,c('fam', 'pc1', 'pc2', 'pc3')]
names(X) = c('family', 'pc1', 'pc2', 'pc3')

X_novel = X[X$family %in% fam_of_novel_group]
X_non_novel = X[!(X$family %in% fam_of_novel_group)]

X_novel[, groups:='novel']
X_non_novel[, groups:='non-novel']

X = rbind(X_novel, X_non_novel)

# read family sp numbers from Goldstein2017
meta = fread('../../../DFC_VSC/save/vsc_epoch_40000/pred_fam/no_neg/top2/pca_highlights/fam_mean/moth_geometry_with_meta.csv')
lit_fam_sp_counts = meta[, c('family', 'Goldstein2017')]

# calc means and sds
X_stats_ = X[, .(pc1_mean=mean(pc1), pc2_mean=mean(pc2), pc3_mean=mean(pc3), pc1_std=sd(pc1), pc2_std=sd(pc2), pc3_std=sd(pc3)), by=family]
X_stats_ = merge(x = X_stats_, y=lit_fam_sp_counts, by = 'family', sort = F)
X_stats = merge(x = X_stats_, y=unique(X[,c('family', 'groups')]), by = 'family', sort = F)

fam_sp_counts = X[, .N, by='family']
fam_sp_enough = fam_sp_counts[fam_sp_counts$N > 3]$family
X_refined = X[family %in% fam_sp_enough]

##### Option 1. TPD of missing data deletion

#TPDs_ = TPDs(species = X_refined$family, traits = X_refined[,c('pc1', 'pc2', 'pc3')])
set.seed(42)
#TPDs_del = TPDs(species = X_refined$family, traits = X_refined[,c('pc1', 'pc2', 'pc3')], trait_ranges = list(c(-3,3),c(-3,3),c(-3,3)))
TPDs_del = TPDs(species = X_refined$family, traits = X_refined[,c('pc1', 'pc2', 'pc3')])

X_stats_refined = X_stats[family %in% fam_sp_enough]
dum = dcast(X_stats_refined, groups ~ family, value.var = 'Goldstein2017')
dum_mat = as.matrix(dum[, 2:dim(dum)[2]])
rownames(dum_mat) = dum$groups
dim(dum_mat)

TPDc_del <- TPDc(TPDs = TPDs_del, sampUnit = dum_mat)

set.seed(42)
RicEveDiv_del <- REND (TPDc = TPDc_del)
set.seed(42)
redundancy_del = redundancy(TPDc = TPDc_del)

RicEveDiv_del
redundancy_del

# Primitive Recent
# comm$Richness 
# 84.84 96.80
# comm$Evenness
# 0.426 0.443
# comm$Divergence
# 0.492 0.558
# 
# Fam
# richness
# 21 26
# redundancy
# 8.683 11.770
# redRel
# 0.434 0.471

##### Option 2. TPD of missing data imputation

X_fam_sp_not_enough = X[family %in% fam_sp_counts[fam_sp_counts$N <= 3]$family]

#X_fam_sp_not_enough_mean = X_fam_sp_not_enough[, .(pc1=mean(pc1), pc2=mean(pc2), pc3=mean(pc3), groups=head(groups,1)), by=c('family')]
#X_imputed = rbind(X_refined, X_fam_sp_not_enough_mean, X_fam_sp_not_enough)

X_fam_sp_not_enough_median = X_fam_sp_not_enough[, .(pc1=median(pc1), pc2=median(pc2), pc3=median(pc3), groups=head(groups,1)), by=c('family')]
X_imputed_ = rbind(X_refined, X_fam_sp_not_enough_median, X_fam_sp_not_enough)

X_imputed_ = X_imputed_[order(family)][order(groups, decreasing = T)]
#X_imputed[family %in% fam_sp_counts[fam_sp_counts$N <= 3]$family]
X_imputed = copy(X_imputed_)

#TPDs_imp = TPDs(species = X_imputed$family, traits = X_imputed[,c('pc1', 'pc2', 'pc3')], trait_ranges = list(c(-4,4),c(-4,4),c(-4,4)))
X_imputed[duplicated(X_imputed_), 'pc1'] = X_imputed_[duplicated(X_imputed_), 'pc1'] + 1e-6
X_imputed[duplicated(X_imputed_), 'pc2'] = X_imputed_[duplicated(X_imputed_), 'pc2'] + 1e-6
X_imputed[duplicated(X_imputed_), 'pc3'] = X_imputed_[duplicated(X_imputed_), 'pc3'] + 1e-6

X_imputed[duplicated(X_imputed_)]
X_imputed_[duplicated(X_imputed_)]

set.seed(42)
TPDs_imp = TPDs(species = X_imputed$family, traits = X_imputed[,c('pc1', 'pc2', 'pc3')])

dum = dcast(X_stats, groups ~ family, value.var = 'Goldstein2017')
dum_mat = as.matrix(dum[, 2:dim(dum)[2]])
rownames(dum_mat) = dum$groups
dim(dum_mat)

set.seed(42)
TPDc_imp <- TPDc(TPDs = TPDs_imp, sampUnit = dum_mat)

set.seed(42)
RicEveDiv_imp <- REND (TPDc = TPDc_imp)
set.seed(42)
redundancy_imp = redundancy(TPDc = TPDc_imp)

RicEveDiv_imp
redundancy_imp

# Primitive Recent
# comm$Richness 
# 96.14 96.86
# comm$Evenness
# 0.393 0.443
# comm$Divergence
# 0.466 0.558
# 
# Fam
# richness
# 25 27
# redundancy
# 9.014 11.788
# redRel
# 0.376 0.453


#############################

RicEveDiv_del
RicEveDiv_imp

redundancy_del
redundancy_imp


names(TPDc_imp$TPDc)
names(TPDc_imp$TPDc$TPDc)

#plotTPD(TPDc_del)
source('plotTPD.R')
pdf(file = "revised2_TPDc_100MA_spPC12.pdf",   # The directory you want to save the file in
    width = 4, # The width of the plot in inches
    height = 8) # The height of the plot in inches
png(file = "revised2_TPDc_100MA_spPC12.png",   # The directory you want to save the file in
    width = 400,
    height = 800)
plotTPD(TPDc_imp, probColors = c('dodgerblue4', 'firebrick3'), leg.text = c('Primitive', 'Recent'))
dev.off()

pdf(file = "revised2_TPDc_100MA_spPC13.pdf",   # The directory you want to save the file in
    width = 4, # The width of the plot in inches
    height = 8) # The height of the plot in inches
png(file = "revised2_TPDc_100MA_spPC13.png",   # The directory you want to save the file in
    width = 400,
    height = 800)
plotTPD(TPDc_imp, plotAxisX = 1, plotAxisY = 3, probColors = c('dodgerblue4', 'firebrick3'), leg.text = c('Primitive', 'Recent'))
dev.off()

#plotTPD(TPDc_del, plotAxisX = 2, plotAxisY = 3)
pdf(file = "revised2_TPDc_100MA_spPC23.pdf",   # The directory you want to save the file in
    width = 4, # The width of the plot in inches
    height = 8) # The height of the plot in inches
png(file = "revised2_TPDc_100MA_spPC23.png",   # The directory you want to save the file in
    width = 400,
    height = 800)
plotTPD(TPDc_imp, plotAxisX = 2, plotAxisY = 3, probColors = c('dodgerblue4', 'firebrick3'), leg.text = c('Primitive', 'Recent'))
dev.off()

###########################
source('plotTPD.R')
pdf(file = "revised2_TPDc_100MA_spPC12.pdf",   # The directory you want to save the file in
    width = 4, # The width of the plot in inches
    height = 8) # The height of the plot in inches
png(file = "revised2_TPDc_100MA_spPC12.png",   # The directory you want to save the file in
    width = 400,
    height = 800)
plotTPD(TPDc_del, probColors = c('dodgerblue4', 'firebrick3'), leg.text = c('Primitive', 'Recent'))
dev.off()

TPDc_del$data

pdf(file = "revised2_TPDc_100MA_spPC13.pdf",   # The directory you want to save the file in
    width = 4, # The width of the plot in inches
    height = 8) # The height of the plot in inches
png(file = "revised2_TPDc_100MA_spPC13.png",   # The directory you want to save the file in
    width = 400,
    height = 800)
plotTPD(TPDc_imp, plotAxisX = 1, plotAxisY = 3, probColors = c('dodgerblue4', 'firebrick3'), leg.text = c('Primitive', 'Recent'))
dev.off()

#plotTPD(TPDc_del, plotAxisX = 2, plotAxisY = 3)
pdf(file = "revised2_TPDc_100MA_spPC23.pdf",   # The directory you want to save the file in
    width = 4, # The width of the plot in inches
    height = 8) # The height of the plot in inches
png(file = "revised2_TPDc_100MA_spPC23.png",   # The directory you want to save the file in
    width = 400,
    height = 800)
plotTPD(TPDc_imp, plotAxisX = 2, plotAxisY = 3, probColors = c('dodgerblue4', 'firebrick3'), leg.text = c('Primitive', 'Recent'))
dev.off()

###########################

plotTPD(TPDc_del, plotAxisX = 1, plotAxisY = 3)
plotTPD(TPDc_imp, plotAxisX = 1, plotAxisY = 2)
plotTPD(TPDs_imp)


plotAxisX = 1
plotAxisY = 3
TPD_grids = TPDc_imp$data$evaluation_grid
TPD_probs = TPDc_imp$TPDc$TPDc

group_name = 'non-novel'
mat_ <- cbind(TPD_grids[,c(plotAxisX,plotAxisY)], TPD_probs[group_name])
names(mat_)<-c("T1", "T2", "prob")

# aggregate probabilities to axes being displayed
mat_non_novel = data.table(mat_)[, .(prob=sum(prob)), by=c('T1', 'T2')]


group_name = 'novel'
mat_ <- cbind(TPD_grids[,c(plotAxisX,plotAxisY)], TPD_probs[group_name])
names(mat_)<-c("T1", "T2", "prob")

# aggregate probabilities to axes being displayed
mat_novel = data.table(mat_)[, .(prob=sum(prob)), by=c('T1', 'T2')]

mat_novel

r = 1
T1_tick_size = mean(diff(sort(unique(mat_novel$T1))))
T2_tick_size = mean(diff(sort(unique(mat_novel$T2))))
max_prob = max(mat_novel$prob)
for (r in 1:nrow(mat_novel)) {
  if (mat_novel[r]$prob > 0) {
    random_sample_n = mat_novel[r]$prob * 10 / max_prob
    break
  }
}


# calc grid size
T1_grid_size = mean(diff(sort(unique(mat_novel$T1))))
T2_grid_size = mean(diff(sort(unique(mat_novel$T2))))
# get num of points per grid center by their probabilities
random_sampled = sample(1:nrow(mat_novel),nrow(mat_novel),replace=TRUE,prob=mat_novel$prob)
# convert to data.table to calc how many random points we need in each grid
random_sampled_dt = data.table(grid_row=random_sampled)
random_grid_points_N = random_sampled_dt[, .N, by='grid_row']
# container for random points
random_points_dt = NULL
for (r in 1:nrow(random_grid_points_N)) {
  # get grid center
  grid_center = mat_novel[random_grid_points_N[r]$grid_row]
  # get random offset
  t1_rand = runif(random_grid_points_N[r]$N, grid_center$T1-T1_grid_size / 2, grid_center$T1+T1_grid_size / 2)
  t2_rand = runif(random_grid_points_N[r]$N, grid_center$T2-T2_grid_size / 2, grid_center$T2+T2_grid_size / 2)
  # concat random points
  random_points_dt = rbind(random_points_dt, data.table(t1_rand, t2_rand))
}

# sort xy for contour function
mat_novel_xy_sorted = mat_novel[order(T1, T2)]
# ignore zeros for finding better contour levels
mat_novel_xy_sorted_no_zero = mat_novel_xy_sorted[mat_novel_xy_sorted$prob!=0]

# plot random points and contours
plot(random_points_dt$t1_rand, random_points_dt$t2_rand, cex=1, col=alpha('skyblue', .8), pch=16)
contour(unique(mat_novel_xy_sorted$T1), unique(mat_novel_xy_sorted$T2), matrix(mat_novel_xy_sorted$prob, ncol = 50, byrow=T), add = T, levels = quantile(mat_novel_xy_sorted_no_zero$prob, prob=c(.05, .5, .95), na.rm = T))

?contour

# plot1 = with(mat_non_novel, ggplot2::ggplot(mat_non_novel, ggplot2::aes(T1, T2,  size = prob),
#                           interpolate=T)) +
#   ggplot2::geom_point(color='skyblue', alpha=.8) +
#   ggplot2::scale_fill_gradient(na.value = "grey80")
# 
# plot2 = with(mat_novel, ggplot2::ggplot(mat_novel, ggplot2::aes(T1, T2,  size = prob),
#                                     interpolate=T)) +
#   ggplot2::geom_point(color='orange', alpha=.8) +
#   ggplot2::scale_fill_gradient(na.value = "grey80")
# 
# plot2
# mat_non_novel_xy_sorted = mat_non_novel[order(T1, T2)]
# 
# psize = -log(mat_non_novel_xy_sorted$prob)/10
# psize = max(psize, na.rm = T) - psize
# 
# plot(mat_non_novel_xy_sorted$T1, mat_non_novel_xy_sorted$T2, cex=psize, col=alpha('skyblue', .8), pch=16)


plotTPD(TPDs_imp, plotAxisX = 2, plotAxisY = 3)

contour(unique(mat_non_novel_xy_sorted$T1), unique(mat_non_novel_xy_sorted$T2), matrix(mat_non_novel_xy_sorted$prob, ncol = 50, byrow=T), add = T, levels = quantile(mat_non_novel_xy_sorted$prob, na.rm = T))
?contour

library(latticeExtra)
library(viridisLite)
?levelplot

levelplot(prob ~ T1 * T2, mat_non_novel, contour=T, at=c(-Inf, quantile(mat_non_novel_xy_sorted$prob, na.rm = T, probs = c(.05, .25, .5, .75, .95)), 1))

levelplot(prob ~ T1 * T2, mat_non_novel, pretty = T)



levelplot(prob ~ T1 * T2, mat_novel, pretty = T, col.regions = viridis(50), contour=T, cuts=10)
levelplot(prob ~ T1 * T2, mat_non_novel, pretty = T, col.regions = viridis(50), contour=T, cuts=10)
