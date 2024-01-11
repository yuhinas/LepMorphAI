#mvMorph
library(mvMORPH);
#first read in the tree (original tree 83 tips)
comp.tree <- read.tree('moth.tre');
#forewing data set to be used for analuses
forewing <- read.csv('moth_geometry_with_meta.csv',sep='\t');
forewing.data <- forewing[,c(8,15,17)];
rownames(forewing.data)<-forewing[,2];
#identify those that are not represented on the tree
missing.taxa <- which(comp.tree$tip.label%in%forewing[,2]==FALSE);
missing.taxa.list <- comp.tree$tip.label[missing.taxa];
#make a new tree with 52 tips that have trait data
my.tree <- drop.tip(comp.tree, tip = missing.taxa.list);
#check the tree, it has now indeed 52 tips

#test phylo signals for the two traits separately
f1 <- forewing.data[,2];
names(f1) <- forewing[,2];
phylosig(my.tree, f1, method = 'K', test = TRUE);
phylosig(my.tree, f1, method = 'lambda', test = TRUE);
#this is significant
f2 <- forewing.data[,3];
names(f2) <- forewing[,2];
phylosig(my.tree, f2, method = 'K', test = TRUE);
phylosig(my.tree, f2, method = 'lambda', test = TRUE);
#only significant based on K, but not lambda

#before analyses, check if all names are the same between tips of the tree and the trait data
#all(rownames(forewing.data)==my.tree$tip.label);
#it is TURE in this case

#single opt BM model
fit_bm1 <- mvBM(my.tree, forewing.data[,2:3], model = 'BM1', param=list(root=F));
#single OU model
fit_OU1 <- mvOU(my.tree, forewing.data[,2:3], model = 'OU1', param=list(root=F));
#early burst
fit_EB <- mvEB(my.tree, forewing.data[,2:3], param=list(root=F));
#multiple opt OU
#make a simmap according to the 6 morphogroups
groups <- forewing.data[,1];
names(groups) <- forewing[,2];
tree_OUM <- make.simmap(my.tree, groups, model = 'ARD');
fit_OUM <- mvOU(tree_OUM, forewing.data[,2:3], model = 'OUM', param=list(root=F));
#multiple opt BM
fit_bmm <- mvBM(tree_OUM, forewing.data[,2:3], model = 'BMM', param=list(root=F));

#compare model fit
results <- list(fit_bm1, fit_bmm, fit_EB, fit_OU1, fit_OUM);
results <- aicw(results, aicc = TRUE);
results;
#OUM multiple optimal regimes is the best based on aic weight

#ancestral state
oum.asr.estimates <- estim(tree_OUM, forewing.data[,2:3], fit_OUM, asr= TRUE);
oum.asr.estimates$estimates;#this will show the estimated ancesrtal states for each node
#for node number
plot(my.tree);
nodelabels(bg = 'white');

#then try hindwing color
hindwing <- read.csv('fam_hw_chars_20230518.csv',sep='\t');
hindwing.data <- hindwing[,c(2,4)];
rownames(hindwing.data)<-hindwing[,1];

#test phylo signals for the two traits separately
t1 <- hindwing.data[,1];
names(t1) <- hindwing[,1];
phylosig(my.tree, t1, method = 'K', test = TRUE);
phylosig(my.tree, t1, method = 'lambda', test = TRUE);
#not significant
t2 <- hindwing.data[,2];
names(t2) <- hindwing[,1];
phylosig(my.tree, t2, method = 'K', test = TRUE);
phylosig(my.tree, t2, method = 'lambda', test = TRUE);
#also not significant at all

#single opt BM model
fitHW_bm1 <- mvBM(my.tree, hindwing.data[,1:2], model = 'BM1', param=list(root=F));
#single OU model
fit_OU1HW <- mvOU(my.tree, hindwing.data[,1:2], model = 'OU1', param=list(root=F));
#early burst
fit_EBHW <- mvEB(my.tree, hindwing.data[,1:2], param=list(root=F));

#multiple opt OU
fit_OUMHW <- mvOU(tree_OUM, hindwing.data[,1:2], model = 'OUM', param=list(root=F));
#multiple opt BM
fit_bmmHW <- mvBM(tree_OUM, hindwing.data[,1:2], model = 'BMM', param=list(root=F));

#compare model fit
resultsHW <- list(fitHW_bm1, fit_EBHW, fit_OU1HW, fit_OUMHW);
resultsHW <- aicw(resultsHW, aicc = TRUE);
resultsHW;
#OUM multiple optimal regimes is the best based on aic weight

#estimate ancestral trait values
oumHW.asr.estimates <- estim(tree_OUM, hindwing.data[,1:2], fit_OUMHW, asr= TRUE);
oumHW.asr.estimates$estimates;#this will show the estimated ancesrtal states for each node
