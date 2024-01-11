#after running everything this section is how we produced the figure
par(mfrow=c(1,2));
q1 <- plot.bammdata(STRAPP.tree, tau = 0.001,method= 'polar', breaksmethod = 'jenks', pal='temperature',, lwd = 1, spex = 'netdiv', labels=TRUE, cex = 0.6, part=0.88);
abline(v = 1.9, lty = 2);#55MYA bat
abline(v = 1.79, lty = 2);#65MYA
abline(v = 1.2, lty = 2);#125MYA
abline(v = 1, lty = 2);#140MYA Angiosperm
addBAMMlegend(q1, direction = 'vertical', location = 'left', cex.axis = 1);
axisPhylo();
title(main = 'Net-Diversification Rate', cex.main = 1.2);
nodelabels(pie=pd$ace,piecol=my.col, cex=0.5);
temp.mat <- to.matrix(state.6, sort(unique(state.6)));
tt<-temp.mat[match(newtree$tip.label,rownames(temp.mat)),]
tiplabels(pie=tt,piecol = my.col, cex=0.3)
vioplot(g0,g1,g2,g3,g4,g5, names = c('0','1','2','3','4','5'),xlab="Morpho-group", ylab='Net-Diversification rate', main='6 groups',col=my.col);

#the followings are individual analysis
library(BAMMtools);
comp.tree <- read.tree('moth.tre');
comp.edata <- getEventData(comp.tree, eventdata = 'event_data.txt', burnin = 0.5);
#plot BAMM results
par(mfrow = c(1,2));
q1 <- plot.bammdata(comp.edata, tau = 0.001, method= 'polar', breaksmethod = 'jenks', pal='temperature', lwd = 1, spex = 'netdiv', labels=TRUE, cex = 0.6);
abline(v = 2.45, lty = 2);#65MYA
abline(v = 1.85, lty = 2);#125MYA Angiosperm
addBAMMlegend(q1, direction = 'vertical', location = 'left', cex.axis = 1);
axisPhylo();
title(main = 'Net-Diversification Rate', cex.main = 1.2);

#div rate shifts plots
credshift <- credibleShiftSet(comp.edata,expectedNumberOfShifts=1,thresholds=10)
plot(credshift)
bestshift<-getBestShiftConfiguration(comp.edata, expectedNumberOfShifts=1,threshold=10)
q1<-plot.bammdata(comp.edata,lwd=1.25, labels=TRUE,cex=0.4)
addBAMMlegend(q1, direction = 'vertical', location = 'left', cex.axis = 1);
axisPhylo();
addBAMMshifts(bestshift,cex=2)
#rate through time plot for all lepidopteran
mrca(comp.tree)['Micropterigidae','Sphingidae'];
#node 93 all moths and butterflies
netdivplot = plotRateThroughTime(STRAPP.tree, avgCol = 'black', xlim = c(2.5,0), ylim = c(0,10), cex.axis = 1, cex.lab = 1, intervalCol = 'grey50', intervals = c(0.05, 0.95), opacity = 1, ratetype = 'netdiv');
abline(v = 1.4, lty = 2);#angiosperm

#plot cophylogeny if you want to see 
#morph.tre <- read.tree('ww_feat_preview_tree_c_20220503.nwk');
library(phytools)
#obj <- cophylo(comp.tree, morph.tre, rotate = FALSE);
#plot(obj,fsize=0.8,link.type='curved',link.lty='solid', link.col=make.transparent('red',0.25))
#did not show this in the paper

#The follwoing only use 52 tips from the original tree because we only have 52 data points representing the morphological states (discrete or binary) from 52 tips
pc.table <- read.csv("PC123.csv");
tip2remove <- which(comp.tree$tip.label%in%pc.table[,1]==FALSE);
tips.remove <- comp.tree$tip.label[tip2remove];
newtree <- drop.tip(comp.tree, tip = tips.remove);
#newtree is the tree used in all the following analyses

#retrieve the states
#use get tip rates before. 
#the tip rates have been saved in a file called TipNetDivRate.csv already
div.rate <- read.csv('TipNetDivRate.csv');
group.def <- read.csv(' MorphoGroup.csv');
total <- merge(div.rate, group.def, by = 'Family');
#manage the state data
state.6 <- total[,3];
names(state.6) <- total[,1];
keep.tips <- which(comp.edata$tip.label%in%total[,1]==TRUE);
STRAPP.tree <- subtreeBAMM(comp.edata, tips = keep.tips);
traitDependentBAMM(STRAPP.tree, (state.6), reps = 1000);
traitDependentBAMM(STRAPP.tree, (state.6), reps = 1000, method='pearson');
traitDependentBAMM(STRAPP.tree, (state.6), reps = 1000, method='kruskal');

#set colors for different morpho groups
my.col <- c('#E1DFCD','#7ABDC7','#987EA7','#849e73','#E7C734','#FD8889');

#ancestral state reconstruction
library(phytools);
ERchar <- ace(state.6, newtree, type = 'discrete', model = 'ER');
SYMchar <- ace(state.6, newtree, type = 'discrete', model = 'SYM');
ARDchar <- ace(state.6, newtree, type = 'discrete', model = 'ARD');
1-pchisq(2*abs(ERchar$loglik - SYMchar$loglik),1);
1-pchisq(2*abs(ERchar$loglik - ARDchar$loglik),1);
1-pchisq(2*abs(ARDchar$loglik - SYMchar$loglik),1);
#ARD has the best loglik
plotTree(newtree, fsize=0.8,ftype='i');
axisPhylo();
nodelabels(node=1:newtree$Nnode+Ntip(newtree), pie=SYMchar$lik.anc, piecol=my.col,cex=0.5)
temp.mat <- to.matrix(state.6, sort(unique(state.6)));
tt<-temp.mat[match(newtree$tip.label,rownames(temp.mat)),]
tiplabels(pie=tt,piecol = my.col, cex=0.3)
#could not plot ARD...use sim instead
par(mfrow=c(2,2))
mttrees <- make.simmap(newtree,state.6,model='ARD',nsim=1000)
pd <- describe.simmap(mttrees,plot=FALSE);
plotTree(newtree, fsize=0.6,ftype='i');
nodelabels(pie=pd$ace,piecol=my.col, cex=0.5);
temp.mat <- to.matrix(state.6, sort(unique(state.6)));
tt<-temp.mat[match(newtree$tip.label,rownames(temp.mat)),]
tiplabels(pie=tt,piecol = my.col, cex=0.3)

#treated morpho as continuous data by extract PC1~3
#first to test if there is disparity of trait value evolution through time
pc1trait <- pc.table[,2];
names(pc1trait) <- pc.table[,1];
pc2trait <- pc.table[,3];
names(pc2trait) <- pc.table[,1];
pc3trait <- pc.table[,4];
names(pc3trait) <- pc.table[,1];
library(phytools);
library(geiger);
par(mfrow=c(2,3));
phenogram(newtree,pc1trait, ylab = "PC1",fsize=.5);
phenogram(newtree,pc2trait, ylab = 'PC2',fsize=.5);
phenogram(newtree,pc3trait, ylab = 'PC3',fsize=.5);
dtt_pc1 = dtt(newtree,pc1trait,nsim=1000);
dtt_pc2 = dtt(newtree,pc2trait,nsim=1000);
dtt_pc3 = dtt(newtree,pc3trait,nsim=1000);

dtt_pc2$times
dtt_pc2$dtt

plot(x=dtt_pc1$times, y=dtt_pc1$dtt)
Ma1 =  (1-dtt_pc1$times) * (max(newtree$edge.length) - min(newtree$edge.length))
# Ma2 =  - (1-dtt_pc2$times) * (max(newtree$edge.length) - min(newtree$edge.length))
# Ma3 =  - (1-dtt_pc3$times) * (max(newtree$edge.length) - min(newtree$edge.length))
Ma = Ma1
ma_dtt_df = data.frame(Ma=Ma, DTT_PC1=dtt_pc1$dtt, DTT_PC2=dtt_pc2$dtt, DTT_PC3=dtt_pc3$dtt)
write.csv(ma_dtt_df, 'ma_dtt_pcs.csv')

#geiger disparity through time PC1-3
#BM expectation in multivariate space. test state = deviation from BM


#PhylogeneticEN analysis morpho rate shift
library(PhylogeneticEM);
newtree1 <- force.ultrametric(newtree, method='extend');
aa<-matrix(pc3trait,nrow=1);
colnames(aa)<-pc.table[,1];
res1 <- PhyloEM(phylo=newtree1,Y_data=aa,process="BM");

#phylogenetic signal tests using lambda and K
pc1test <- phylosig(newtree, pc1trait, test = TRUE);
pc1testL <- phylosig(newtree, pc1trait, method = 'lambda', test = TRUE);

pc2test <- phylosig(newtree, pc2trait, test = TRUE);
pc2testL <- phylosig(newtree, pc2trait, method = 'lambda', test = TRUE);

pc3test <- phylosig(newtree, pc3trait, test = TRUE);
pc3testL <- phylosig(newtree, pc3trait, method = 'lambda', test = TRUE);



#test grouping and div rates
par(mfrow = c(1,3))

div.rate <- read.csv('TipNetDivRate.csv');
group.def <- read.csv(' MorphoGroup.csv');
total <- merge(div.rate, group.def, by = 'Family');
library(vioplot);
g0 <- total$Rate[total$group==0];
g1 <- total$Rate[total$group==1];
g2 <- total$Rate[total$group==2];
g3 <- total$Rate[total$group==3];
g4 <- total$Rate[total$group==4];
g5 <- total$Rate[total$group==5];
vioplot(g0,g1,g2,g3,g4,g5, names = c('0','1','2','3','4','5'),xlab="Morpho-group", ylab='Net-Diversification rate', main='6 groups',col=my.col)

#10groups
group.tab <- read.csv('sim_group.csv');
ten.group <- group.tab[,-(2:8)];
total <- merge(div.rate, ten.group, by = 'Family');
g0 <- total$Rate[total$X10groups==0];
g1 <- total$Rate[total$X10groups==1];
g2 <- total$Rate[total$X10groups==2];
g3 <- total$Rate[total$X10groups==3];
g4 <- total$Rate[total$X10groups==4];
g5 <- total$Rate[total$X10groups==5];
g6 <- total$Rate[total$X10groups==6];
g7 <- total$Rate[total$X10groups==7];
g8 <- total$Rate[total$X10groups==8];
g9 <- total$Rate[total$X10groups==9];
vioplot(g0,g1,g2,g3,g4,g5,g6,g7,g8,g9, names = c('0','1','2','3','4','5','6','7','8','9'), main='10 groups', xlab='Group', ylab='Ne-Diversification Rate')
fit10 <- lm(total$X10groups~total$Rate);
anova(fit10);

#3groups
group.tab <- read.csv('sim_group.csv');
three.group <- group.tab[,-(3:9)];
total <- merge(div.rate, three.group, by = 'Family');
g0 <- total$Rate[total$X3groups==0];
g1 <- total$Rate[total$X3groups==1];
g2 <- total$Rate[total$X3groups==2];
vioplot(g0,g1,g2, names = c('0','1','2'), main='3 groups', xlab='Group', ylab='Ne-Diversification Rate')
fit3 <- lm(total$X3groups~total$Rate);
anova(fit3);

