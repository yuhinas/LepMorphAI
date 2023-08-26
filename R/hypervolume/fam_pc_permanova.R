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

fam_of_novel_group = c('Bombycidae', 'Brahmaeidae', 'Cossidae', 'Drepanidae', 'Endromidae', 'Erebidae', 'Eupterotidae', 'Euteliidae', 'Geometridae', 'Hepialidae', 'Lasiocampidae', 'Limacodidae', 'Lycaenidae', 'Megalopygidae', 'Mimallonidae', 'Noctuidae', 'Nolidae', 'Notodontidae', 'Nymphalidae', 'Papilionidae', 'Pieridae', 'Riodinidae', 'Saturniidae', 'Sematuridae', 'Sphingidae', 'Thyrididae', 'Uraniidae')

length(fam_of_novel_group)

moth_geometry_with_meta = fread('../../save/vsc_epoch_40000/pred_fam/no_neg/top2/pca_highlights/fam_mean/moth_geometry_with_meta.csv')


##### Family-mean 
X_dr_for_1d = fread('../../save/vsc_epoch_40000/pred_fam/no_neg/top2/pca_highlights/fam_mean/X_dr_for_1d.csv', header = T)
names(X_dr_for_1d) = c('family', 'pc1', 'pc2', 'pc3')
X_dr_for_1d

X_dr_for_1d_with_phylodata = X_dr_for_1d[X_dr_for_1d$family %in% fam_with_phylodata]


more_recent = X_dr_for_1d_with_phylodata[X_dr_for_1d_with_phylodata$family %in% fam_of_novel_group]
more_recent[,group:='more_recent']
primative = X_dr_for_1d_with_phylodata[!(X_dr_for_1d_with_phylodata$family %in% fam_of_novel_group)]
primative[,group:='primative']

dt = rbind(more_recent, primative)

sort(dt$family)

pcs = dt[,2:4]


library(vegan)

adonis(pcs ~ group, data=dt, method = 'euclidean', permutations = 1000000)
# should use adonis2 in new version
# adonis2(pcs ~ group, data=dt, method = 'euclidean', permutations = 1000)

