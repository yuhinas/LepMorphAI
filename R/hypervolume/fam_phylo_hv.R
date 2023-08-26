# Calcutate and plot the hypervolumes of the recent and primative groups

library(hypervolume)
library(data.table)

fam_before_sp_burst = c('Tineidae', 'Psychidae', 'Prodoxidae', 'Adelidae', 'Nepticulidae', 'Hepialidae', 'Eriocraniidae')
fam_with_phylodata = c('Hydroptilidae', 'Phryganeidae', 'Helicopsychidae', 'Leptoceridae',
                       'Hydropsychidae', 'Stenopsychidae', 'Philopotamidae',
                       'Polycentropodidae', 'Psychomyiidae', 'Micropterigidae',
                       'Agathiphagidae', 'Heterobathmiidae', 'Eriocraniidae',
                       'Acanthopteroctetidae', 'Neopseustidae', 'Lophocoronidae',
                       'Hepialidae', 'Opostegidae', 'Nepticulidae', 'Andesianidae',
                       'Adelidae', 'Prodoxidae', 'Palaephatidae', 'Tischeriidae',
                       'Meessiidae', 'Psychidae', 'Dryadaulidae', 'Tineidae',
                       'Roeslerstammiidae', 'Plutellidae', 'Yponomeutidae',
                       'Bucculatricidae', 'Lyonetiidae', 'Gracillariidae', 'Urodidae',
                       'Choreutidae', 'Immidae', 'Tortricidae', 'Sesiidae', 'Lacturidae',
                       'Zygaenidae', 'Dalceridae', 'Limacodidae', 'Megalopygidae',
                       'Cossidae', 'Castniidae', 'Papilionidae', 'Hedylidae',
                       'Hesperiidae', 'Pieridae', 'Riodinidae', 'Lycaenidae',
                       'Nymphalidae', 'Depressariidae', 'Autostichidae',
                       'Cosmopterigidae', 'Gelechiidae', 'Alucitidae', 'Pterophoridae',
                       'Callidulidae', 'Thyrididae', 'Pyralidae', 'Crambidae',
                       'Mimallonidae', 'Cimeliidae', 'Doidae', 'Drepanidae',
                       'Notodontidae', 'Erebidae', 'Nolidae', 'Euteliidae', 'Noctuidae',
                       'Epicopeiidae', 'Sematuridae', 'Uraniidae', 'Geometridae',
                       'Lasiocampidae', 'Bombycidae', 'Brahmaeidae', 'Eupterotidae',
                       'Endromidae', 'Saturniidae', 'Sphingidae')


# fam_of_novel_group = c('Sphingidae', 'Saturniidae', 'Endromidae', 'Eupterotidae', 
#                        'Brahmaeidae', 'Bombycidae', 'Lasiocampidae', 'Geometridae', 
#                        'Uraniidae', 'Sematuridae', 'Noctuidae', 'Euteliidae', 
#                        'Nolidae', 'Erebidae', 'Notodontidae', 'Drepanidae', 'Mimallonidae',
#                        'Nymphalidae', 'Lycaenidae', 'Riodinidae', 'Pieridae' ,'Hesperiidae', 'Papilionidae')

fam_of_novel_group = c('Bombycidae', 'Brahmaeidae', 'Cossidae', 'Drepanidae', 'Endromidae', 
                       'Erebidae', 'Eupterotidae', 'Euteliidae', 'Geometridae', 'Hepialidae', 
                       'Lasiocampidae', 'Limacodidae', 'Lycaenidae', 'Megalopygidae', 'Mimallonidae', 
                       'Noctuidae', 'Nolidae', 'Notodontidae', 'Nymphalidae', 'Papilionidae', 'Pieridae', 
                       'Riodinidae', 'Saturniidae', 'Sematuridae', 'Sphingidae', 'Thyrididae', 'Uraniidae')
length(fam_of_novel_group)
moth_geometry_with_meta = fread('../../save/vsc_epoch_40000/pred_fam/no_neg/top2/pca_highlights/fam_mean/moth_geometry_with_meta.csv')


##### Family-mean 
X_dr_for_1d = fread('../../save/vsc_epoch_40000/pred_fam/no_neg/top2/pca_highlights/fam_mean/X_dr_for_1d.csv', header = T)
names(X_dr_for_1d) = c('family', 'pc1', 'pc2', 'pc3')
X_dr_for_1d

X_dr_for_1d_with_phylodata = X_dr_for_1d[X_dr_for_1d$family %in% fam_with_phylodata]


##### species-mean on fam-mean pca
sp_on_fam_pca = fread('../save/vsc_epoch_40000/pred_fam/no_neg/top2/pca_highlights/fam_mean/sp_mean_on_fam_mean_pca.csv', header = T)
names(sp_on_fam_pca) = c('family', 'pc1', 'pc2', 'pc3')
sp_on_fam_pca_with_phylodata = sp_on_fam_pca[sp_on_fam_pca$family %in% fam_with_phylodata]
# 

# Novel morpho group (Recent groupds)
sp_on_fam_pca_with_phylodata_of_novel_group = sp_on_fam_pca_with_phylodata[sp_on_fam_pca_with_phylodata$family %in% fam_of_novel_group]
# fwrite(sp_on_fam_pca_with_phylodata_of_novel_group, './fam_mean/sp_on_fam_pca_with_phylodata_of_novel_group.csv')
sp_on_fam_pca_with_phylodata_of_novel_group_bw = estimate_bandwidth(data = sp_on_fam_pca_with_phylodata_of_novel_group[,2:4], method = 'cross-validation')
set.seed(42)
hv_sp_on_fam_pca_with_phylodata_of_novel_group = hypervolume(sp_on_fam_pca_with_phylodata_of_novel_group[,2:4], 
                                                             name='Hypervolume After 100Ma', 
                                                             kde.bandwidth=sp_on_fam_pca_with_phylodata_of_novel_group_bw,
                                                             verbose=T)


# Non novel morpho group(Primative groups)
sp_on_fam_pca_with_phylodata_of_non_novel_group = sp_on_fam_pca_with_phylodata[!sp_on_fam_pca_with_phylodata$family %in% fam_of_novel_group]
#fwrite(sp_on_fam_pca_with_phylodata_of_non_novel_group, './fam_mean/sp_on_fam_pca_with_phylodata_of_non_novel_group.csv')
sp_on_fam_pca_with_phylodata_of_non_novel_group_bw = estimate_bandwidth(data = sp_on_fam_pca_with_phylodata_of_non_novel_group[,2:4], method = 'cross-validation')
set.seed(42)
hv_sp_on_fam_pca_with_phylodata_of_non_novel_group = hypervolume(sp_on_fam_pca_with_phylodata_of_non_novel_group[,2:4], 
                                                                 name='Hypervolume Before 100Ma', 
                                                                 kde.bandwidth=sp_on_fam_pca_with_phylodata_of_non_novel_group_bw,
                                                                 verbose=T)

hv_novel_or_else = hypervolume_join(hv_sp_on_fam_pca_with_phylodata_of_non_novel_group, hv_sp_on_fam_pca_with_phylodata_of_novel_group)

#pdf('hv_100MYa_PC12.pdf', width = 10, height = 10)
pdf('revised_hv_100MYa_PC23_with_data.pdf', width = 10, height = 10)
#png('hv_100MYa_PC23.png', width = 2400, height = 2400, res=300)
# all axes: 1,2,3
# display axes
yx_axis = c(3, 2)
# the remain axis
dum_axis = 1
rdim = 2

# Change the contents of list items to plot the selected axes only
list_item1 = copy(hv_novel_or_else[[1]])

list_item1_for_inc = copy(hv_novel_or_else[[1]])
list_item1_for_inc@Data[, dum_axis] = 0
pts_in_non_novel = hypervolume_inclusion_test(hv_sp_on_fam_pca_with_phylodata_of_non_novel_group, list_item1_for_inc@Data, fast.or.accurate = 'accurate')
pts_not_in_non_novel_data = list_item1_for_inc@Data[!pts_in_non_novel,]

#list_item1@Data = pts_not_in_non_novel_data[,yx_axis] # hv_novel_or_else[[1]]@Data[,yx_axis]
list_item1@Data = hv_novel_or_else[[1]]@Data[,yx_axis]
list_item1@RandomPoints = hv_novel_or_else[[1]]@RandomPoints[,yx_axis]
list_item1@Dimensionality = rdim
list_item1@Name = 'Before 100MYa'
list_item2 = copy(hv_novel_or_else[[2]])

list_item2_for_inc = copy(hv_novel_or_else[[2]])
list_item2_for_inc@Data[, dum_axis] = 0
pts_in_novel = hypervolume_inclusion_test(hv_sp_on_fam_pca_with_phylodata_of_novel_group, list_item2_for_inc@Data, fast.or.accurate = 'accurate')
pts_not_in_novel_data = list_item2_for_inc@Data[!pts_in_novel,]

#list_item2@Data = pts_not_in_novel_data[,yx_axis] #hv_novel_or_else[[2]]@Data[,yx_axis]
list_item2@Data = hv_novel_or_else[[2]]@Data[,yx_axis]
list_item2@RandomPoints = hv_novel_or_else[[2]]@RandomPoints[,yx_axis]
list_item2@Dimensionality = rdim
list_item2@Name = 'After 100MYa'
tempered_hv_novel_or_else = hypervolume_join(list_item1, list_item2)
set.seed(42)

# dum_xy_label = function (j, i) {
#   print('Shua shua tsun zai gan')
#   title(xlab="PC1", ylab="PC2", cex.lab=1)
# }

plot(tempered_hv_novel_or_else, show.random=F, show.data=T,
     colors = c('#7ABDC7', '#FD8889'), 
     point.dark.factor = 0.3, 
     point.alpha.min = 1,
     limits=list(c(-4, 4), c(-4, 4)),
     show.legend=T,
     contour.type = 'alphahull',
     contour.alphahull.alpha = .25
     )

# ?plot.HypervolumeList

dev.off()


# The statistics of set operation (intersect, union, etc.)
set.seed(42)
hv_novel_or_else_set = hypervolume_set(hv_sp_on_fam_pca_with_phylodata_of_non_novel_group, hv_sp_on_fam_pca_with_phylodata_of_novel_group, check.memory = F)
hypervolume_overlap_statistics(hv_novel_or_else_set)

hv_novel_or_else_set[[2]]
