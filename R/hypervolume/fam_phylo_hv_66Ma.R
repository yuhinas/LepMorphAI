# Calcutate and plot the hypervolumes of the families in the recent group delineated with 66Ma

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

fam_of_novel_group = c('Bombycidae', 'Brahmaeidae', 'Cossidae', 'Drepanidae', 'Endromidae', 
                       'Erebidae', 'Eupterotidae', 'Euteliidae', 'Geometridae', 'Hepialidae', 
                       'Lasiocampidae', 'Limacodidae', 'Lycaenidae', 'Megalopygidae', 'Mimallonidae', 
                       'Noctuidae', 'Nolidae', 'Notodontidae', 'Nymphalidae', 'Papilionidae', 'Pieridae', 
                       'Riodinidae', 'Saturniidae', 'Sematuridae', 'Sphingidae', 'Thyrididae', 'Uraniidae')

fam_after_660Ma_group = c('Brahmaeidae', 'Eupterotidae', 'Noctuidae', 'Euteliidae', 'Nolidae', 'Lycaenidae', 'Riodinidae')

moth_geometry_with_meta = fread('../../save/vsc_epoch_40000/pred_fam/no_neg/top2/pca_highlights/fam_mean/moth_geometry_with_meta.csv')
fam_of_recent_group_dt = moth_geometry_with_meta[family %in% fam_of_novel_group]
fam_of_primitive_group_dt = moth_geometry_with_meta[!(family %in% fam_of_novel_group)]

sum(fam_of_primitive_group_dt$family_sp_count_with_photo)
sum(fam_of_recent_group_dt$family_sp_count_with_photo)
sum(fam_of_recent_group_dt[family %in% fam_after_660Ma_group]$family_sp_count_with_photo)

length(fam_of_novel_group)
length(fam_after_660Ma_group)

names(moth_geometry_with_meta)

# num of species
a = sum(moth_geometry_with_meta[family %in% fam_after_660Ma_group]$Goldstein2017)
b = sum(moth_geometry_with_meta[!(family %in% fam_after_660Ma_group)]$Goldstein2017)
# species ratio
a / (a + b)

sum(moth_geometry_with_meta[family %in% fam_after_660Ma_group]$family_sp_count_with_photo)

a = sum(moth_geometry_with_meta[family %in% fam_after_660Ma_group]$family_sp_count_with_photo)
b = sum(moth_geometry_with_meta[!(family %in% fam_after_660Ma_group)]$family_sp_count_with_photo)
# species with photo ratio
a / (a + b)


#moth_geometry_with_meta = fread('./fam_mean/moth_geometry_with_meta.csv')


##### Family-mean 
X_dr_for_1d = fread('../../save/vsc_epoch_40000/pred_fam/no_neg/top2/pca_highlights/fam_mean/X_dr_for_1d.csv', header = T)
names(X_dr_for_1d) = c('family', 'pc1', 'pc2', 'pc3')
X_dr_for_1d

X_dr_for_1d_with_phylodata = X_dr_for_1d[X_dr_for_1d$family %in% fam_with_phylodata]


##### species-mean on fam-mean pca
sp_on_fam_pca = fread('./fam_mean/sp_mean_on_fam_mean_pca.csv', header = T)
names(sp_on_fam_pca) = c('family', 'pc1', 'pc2', 'pc3')
sp_on_fam_pca_with_phylodata = sp_on_fam_pca[sp_on_fam_pca$family %in% fam_with_phylodata]

# Novel morpho group
sp_on_fam_pca_with_phylodata_of_novel_group = sp_on_fam_pca_with_phylodata[sp_on_fam_pca_with_phylodata$family %in% fam_of_novel_group]

# Non-novel morpho group
sp_on_fam_pca_with_phylodata_of_non_novel_group = sp_on_fam_pca_with_phylodata[!sp_on_fam_pca_with_phylodata$family %in% fam_of_novel_group]

#fwrite(sp_on_fam_pca_with_phylodata_of_non_novel_group, './fam_mean/sp_on_fam_pca_with_phylodata_of_non_novel_group.csv')
sp_on_fam_pca_with_phylodata_of_non_novel_group_bw = estimate_bandwidth(data = sp_on_fam_pca_with_phylodata_of_non_novel_group[,2:4], method = 'cross-validation')
set.seed(42)
hv_sp_on_fam_pca_with_phylodata_of_non_novel_group = hypervolume(sp_on_fam_pca_with_phylodata_of_non_novel_group[,2:4], 
                                                                 name='Hypervolume Before 100Ma', 
                                                                 kde.bandwidth=sp_on_fam_pca_with_phylodata_of_non_novel_group_bw,
                                                                 verbose=T)


sp_on_fam_pca_with_phylodata_of_novel_group_bw = estimate_bandwidth(data = sp_on_fam_pca_with_phylodata_of_novel_group[,2:4], method = 'cross-validation')



sp_on_fam_pca_with_phylodata_before66Ma = sp_on_fam_pca_with_phylodata_of_novel_group[!(sp_on_fam_pca_with_phylodata_of_novel_group$family %in% fam_after_660Ma_group)]
# fwrite(sp_on_fam_pca_with_phylodata_of_novel_group, './fam_mean/sp_on_fam_pca_with_phylodata_of_novel_group.csv')
sp_on_fam_pca_with_phylodata_before66Ma_bw = estimate_bandwidth(data = sp_on_fam_pca_with_phylodata_before66Ma[,2:4], method = 'cross-validation')
set.seed(42)
hv_sp_on_fam_pca_with_phylodata_before66Ma = hypervolume(sp_on_fam_pca_with_phylodata_before66Ma[,2:4], 
                                                             name='Hypervolume Before 66Ma', 
                                                             kde.bandwidth=sp_on_fam_pca_with_phylodata_before66Ma_bw,
                                                             verbose=T)

sp_on_fam_pca_with_phylodata_after66Ma = sp_on_fam_pca_with_phylodata_of_novel_group[sp_on_fam_pca_with_phylodata_of_novel_group$family %in% fam_after_660Ma_group]
# fwrite(sp_on_fam_pca_with_phylodata_of_novel_group, './fam_mean/sp_on_fam_pca_with_phylodata_of_novel_group.csv')
sp_on_fam_pca_with_phylodata_after66Ma_bw = estimate_bandwidth(data = sp_on_fam_pca_with_phylodata_after66Ma[,2:4], method = 'cross-validation')
set.seed(42)
hv_sp_on_fam_pca_with_phylodata_after66Ma = hypervolume(sp_on_fam_pca_with_phylodata_after66Ma[,2:4], 
                                                         name='Hypervolume After 66Ma', 
                                                         kde.bandwidth=sp_on_fam_pca_with_phylodata_after66Ma_bw,
                                                         verbose=T)


hv_before_or_after66Ma = hypervolume_join(hv_sp_on_fam_pca_with_phylodata_before66Ma, hv_sp_on_fam_pca_with_phylodata_after66Ma)

# pdf('hv_novel_or_else.pdf')
# png('hv_novel_or_else.png', width = 2400, height = 2400, res=300)
# set.seed(42)
# plot(hv_novel_or_else, show.random=F, show.data=F, colors = c('#7ABDC7', '#FD8889'), point.dark.factor = 0.3, point.alpha.min = 1)
# dev.off()


#pts_in_novel = hypervolume_inclusion_test(hv_sp_on_fam_pca_with_phylodata_of_novel_group, hv_novel_or_else[[2]]@Data, fast.or.accurate = 'accurate')
#pts_not_in_novel_data = hv_novel_or_else[[2]]@Data[!pts_in_novel,]

#pdf('hv_100MYa_PC12.pdf', width = 10, height = 10)
pdf('revised_hv_66Ma_PC23_with_data.pdf', width = 10, height = 10)
#png('hv_100MYa_PC23.png', width = 2400, height = 2400, res=300)
yx_axis = c(3, 2)
dum_axis = 1
rdim = 2

list_item1 = copy(hv_before_or_after66Ma[[1]])

# list_item1_for_inc = copy(hv_before_or_after66Ma[[1]])
# list_item1_for_inc@Data[, dum_axis] = 0
# pts_before66Ma = hypervolume_inclusion_test(hv_sp_on_fam_pca_with_phylodata_before66Ma, list_item1_for_inc@Data, fast.or.accurate = 'accurate')
# pts_after66Ma = list_item1_for_inc@Data[!pts_before66Ma,]

#list_item1@Data = pts_not_in_non_novel_data[,yx_axis] # hv_novel_or_else[[1]]@Data[,yx_axis]
list_item1@Data = hv_before_or_after66Ma[[1]]@Data[,yx_axis]
list_item1@RandomPoints = hv_before_or_after66Ma[[1]]@RandomPoints[,yx_axis]
list_item1@Dimensionality = rdim
list_item1@Name = 'More Recent Group Before 66Ma'


list_item2 = copy(hv_before_or_after66Ma[[2]])

# list_item2_for_inc = copy(hv_before_or_after66Ma[[2]])
# list_item2_for_inc@Data[, dum_axis] = 0
# pts_before66Ma = hypervolume_inclusion_test(hv_sp_on_fam_pca_with_phylodata_before66Ma, list_item2_for_inc@Data, fast.or.accurate = 'accurate')
# pts_after66Ma = list_item2_for_inc@Data[!pts_before66Ma,]

#list_item2@Data = pts_not_in_novel_data[,yx_axis] #hv_novel_or_else[[2]]@Data[,yx_axis]
list_item2@Data = hv_before_or_after66Ma[[2]]@Data[,yx_axis]
list_item2@RandomPoints = hv_before_or_after66Ma[[2]]@RandomPoints[,yx_axis]
list_item2@Dimensionality = rdim
list_item2@Name = 'More Recent Group After 66Ma'


list_item0 = copy(hv_sp_on_fam_pca_with_phylodata_of_non_novel_group)
list_item0@Data = hv_sp_on_fam_pca_with_phylodata_of_non_novel_group@Data[,yx_axis]
list_item0@RandomPoints = hv_sp_on_fam_pca_with_phylodata_of_non_novel_group@RandomPoints[,yx_axis]
list_item0@Dimensionality = rdim
list_item0@Name = 'Primitive Group'


tempered_hv_non_novel_with_before_or_after66Ma = hypervolume_join(list_item0, list_item1, list_item2)
#tempered_hv_before_or_after66Ma = hypervolume_join(list_item1, list_item2)

set.seed(1)
plot(tempered_hv_non_novel_with_before_or_after66Ma, show.random=F, show.data=T,
     colors = c('#7ABDC7', '#FD8889', 'green'), 
     point.dark.factor = 0.3, 
     point.alpha.min = 1,
     limits=list(c(-4, 4), c(-4, 4), c(-4, 4)),
     show.legend=T,
     contour.type = 'alphahull',
     contour.alphahull.alpha = .25
     )

# ?plot.HypervolumeList

dev.off()


getwd()

set.seed(42)
hv_before_or_after66Ma_set = hypervolume_set(hv_sp_on_fam_pca_with_phylodata_before66Ma, hv_sp_on_fam_pca_with_phylodata_after66Ma, check.memory = F)
hypervolume_overlap_statistics(hv_before_or_after66Ma_set)

hv_before_or_after66Ma_set[[2]] # HV after 66 Ma
hv_before_or_after66Ma_set[[4]] # HV of Union of (before and after 66Ma)

18.76 / 31.82

#############################################################
list_item0 = copy(hv_sp_on_fam_pca_with_phylodata_of_non_novel_group)
list_item0@Data = hv_sp_on_fam_pca_with_phylodata_of_non_novel_group@Data[,yx_axis]
list_item0@RandomPoints = hv_sp_on_fam_pca_with_phylodata_of_non_novel_group@RandomPoints[,yx_axis]
list_item0@Dimensionality = rdim
list_item0@Name = 'Primitive'

hv_non_novel_with_before_or_after66Ma = hypervolume_join(list_item0, list_item1, list_item2)

set.seed(42)
plot(hv_non_novel_with_before_or_after66Ma, show.random=F, show.data=T,
     colors = c('#7ABDC7', '#FD8889', 'green'), 
     point.dark.factor = 0.3, 
     point.alpha.min = 1,
     limits=list(c(-4, 4), c(-4, 4), c(-4, 4)),
     show.legend=T,
     contour.type = 'alphahull',
     contour.alphahull.alpha = .25
)

