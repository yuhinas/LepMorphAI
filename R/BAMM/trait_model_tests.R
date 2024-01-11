#install.packages(c('geiger', 'OUwie'))
#to test the model fit to continuous traits PC1~3
library(phytools);
library(geiger);
library(OUwie);
#read the tree
comp.tree <- read.tree('moth.tre');
#edit the tree to show only those have trait data
pc.table <- read.csv("PC123.csv");#this is the PC1~3 table
tip2remove <- which(comp.tree$tip.label%in%pc.table[,1]==FALSE);
tips.remove <- comp.tree$tip.label[tip2remove];
newtree <- drop.tip(comp.tree, tip = tips.remove);
#make trait file from PC1~3
pc1trait <- pc.table[,2];
names(pc1trait) <- pc.table[,1];
pc2trait <- pc.table[,3];
names(pc2trait) <- pc.table[,1];
pc3trait <- pc.table[,4];
names(pc3trait) <- pc.table[,1];
group.def <- read.csv(' MorphoGroup.csv');
#run continuous trait analyses using phytools and geiger
#start with PC1
fitContinuous(newtree,pc1trait,'BM');#Brownian motion
fitContinuous(newtree,pc1trait,model = 'rate_trend');#trend model
fitContinuous(newtree,pc1trait,model = 'EB');#early burst
fitDiversityModel(newtree,pc1trait,showTree=FALSE);#diversity dependent model
ecomorph <- as.factor(group.def[,2]);
names(ecomorph) <- group.def[,1];
data.re <- data.frame(pc.table[,1],ecomorph,pc1trait);
ecomorphTree<-make.simmap(newtree,ecomorph,model='ER');
OUwie(ecomorphTree,data.re,model='OU1',simmap.tree=TRUE);#OUmodel with 1 regime
OUwie(ecomorphTree,data.re,model='OUM',simmap.tree=TRUE);#OUmodel with multiple regime

#PC2
fitContinuous(newtree,pc2trait,'BM');#Brownian motion
fitContinuous(newtree,pc2trait,model = 'rate_trend');#trend model
fitContinuous(newtree,pc2trait,model = 'EB');#early burst
fitDiversityModel(newtree,pc2trait,showTree=FALSE);#diversity dependent model
data.re <- data.frame(pc.table[,1],ecomorph,pc2trait);
ecomorphTree<-make.simmap(newtree,ecomorph,model='ER');
OUwie(ecomorphTree,data.re,model='OU1',simmap.tree=TRUE);#OUmodel with 1 regime
OUwie(ecomorphTree,data.re,model='OUM',simmap.tree=TRUE);#OUmodel with multiple regime

#PC3
fitContinuous(newtree,pc3trait,'BM');#Brownian motion
fitContinuous(newtree,pc3trait,model = 'rate_trend');#trend model
fitContinuous(newtree,pc3trait,model = 'EB');#early burst
fitDiversityModel(newtree,pc3trait,showTree=FALSE);#diversity dependent model
data.re <- data.frame(pc.table[,1],ecomorph,pc3trait);
ecomorphTree<-make.simmap(newtree,ecomorph,model='ER');
OUwie(ecomorphTree,data.re,model='OU1',simmap.tree=TRUE);#OUmodel with 1 regime
OUwie(ecomorphTree,data.re,model='OUM',simmap.tree=TRUE);#OUmodel with multiple regime
