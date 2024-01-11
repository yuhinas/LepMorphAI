library(ape)
library(caper)
library(data.table)
library(phytools)

#mvMorph
library(mvMORPH);
#first read in the tree (original tree 83 tips)
comp.tree <- read.tree('moth.tre');
#forewing data set to be used for analuses
forewing <- fread('moth_geometry_with_meta.csv', sep='\t');
forewing.data <- forewing[,c(2, 8,15,17)];
rownames(forewing.data)<-forewing[,2]$family;
#identify those that are not represented on the tree
missing.taxa <- which(comp.tree$tip.label%in%forewing[,2]$family==FALSE);
missing.taxa.list <- comp.tree$tip.label[missing.taxa];
#make a new tree with 52 tips that have trait data
my.tree <- drop.tip(comp.tree, tip = missing.taxa.list);
#check the tree, it has now indeed 52 tips
my.tree
plot(my.tree)

# Create a data frame with your trait data
# Make sure the row names of your data frame match the tip labels in your phylogenetic tree
fw_trait_data <- fread("forewing_asr_mvMorph_OUM.csv")
hw_trait_data <- fread("hindwing_asr_mvMorph_OUM.csv")

##########################################
ar.lm = lm(aspect_ratio ~ node_age, trait_data)
summary(ar.lm)

x2ndMoA.lm = lm(nd2ndMoA_root_on_right ~ node_age, trait_data)
summary(x2ndMoA.lm)

#########################################
sat.lm = lm(color_saturation ~ node_age, trait_data)
rich.lm = lm(color_richness ~ node_age, trait_data)

summary(sat.lm)
summary(rich.lm)

##########################################

#############################
# FW aspect ratio
fw_tip_traits = matrix(forewing.data$aspect_ratio, nrow = 1)
names(fw_tip_traits) = forewing.data$family

fw_node_traits = matrix(fw_trait_data$aspect_ratio, nrow = 1)
names(fw_node_traits) = fw_trait_data$node

contMap(my.tree, fw_tip_traits, method='user', anc.states=fw_node_traits)
title('Aspect Ratio')
abline(v=1.41438, lty=2)

# FW 2nd MoA
fw_tip_traits = matrix(forewing.data$nd2ndMoA_root_on_right, nrow = 1)
names(fw_tip_traits) = forewing.data$family

fw_node_traits = matrix(fw_trait_data$nd2ndMoA_root_on_right, nrow = 1)
names(fw_node_traits) = fw_trait_data$node


contMap(my.tree, fw_tip_traits, method='user', anc.states=fw_node_traits)
title('Second Moment of Area')
abline(v=1.41438, lty=2)


####
tip_two_traits = cbind(forewing.data$nd2ndMoA_root_on_right, forewing.data$aspect_ratio)
row.names(tip_two_traits) = forewing.data$family
node_two_traits = cbind(fw_trait_data$nd2ndMoA_root_on_right, fw_trait_data$aspect_ratio)
row.names(node_two_traits) = fw_trait_data$node

# x=0, edge.length=2.414
# edge.length=1.1, x=1.414
# x=2.414, edge.length=
min(my.tree$edge.length)
plotTree(my.tree,ftype="i")
nodelabels(frame="circ",bg="white",cex=0.8)
abline(v=1.41438, lty=2)
abline(v=2.41438, lty=2)
abline(v=0, lty=2)

node_depths = node.depth.edgelength(my.tree)
parent_nodes = c()
parent_depths = c()
for (ni in 1:103) {
  edge_row = which(my.tree$edge[, 2] == ni)
  if(length(edge_row) == 0) {
    print('root')
    parent_nodes = c(parent_nodes, 0)
    parent_depths = c(parent_depths, 999)
  } else {
    parent_node = my.tree$edge[edge_row, 1]
    parent_nodes = c(parent_nodes, parent_node)
    parent_depths = c(parent_depths, node_depths[parent_node])
  }
}

parent_ages = 2.41438 - parent_depths
nodes11 = which(parent_ages <= 1.1)
nodes_not11 = which(parent_ages > 1.1)
# 88, 87, 84, 81, 74, 'Cossidae', 70, Sessidae,
# painted.tree = paintSubTree(my.tree ,88 ,"more recent")
# painted.tree = paintSubTree(painted.tree ,87 ,"more recent")
# painted.tree = paintSubTree(painted.tree ,84 ,"more recent")
# painted.tree = paintSubTree(painted.tree ,81 ,"more recent")
# painted.tree = paintSubTree(painted.tree ,97 ,"more recent")
cols11 = setNames(rep('red', length(nodes11)), nodes11)
cols_not11 = setNames(rep('blue', length(nodes_not11)), nodes_not11)
phylomorphospace(my.tree, tip_two_traits, node_two_traits, label='horizontal', control=list(col.node=c(cols11, cols_not11)))


#############################
# HW Saturation
hindwing <- fread('fam_hw_chars_20230518.csv', sep='\t');
hindwing.data <- hindwing[,c('family', 'color_richness', 'color_saturation')];

hw_tip_traits = matrix(hindwing.data$color_saturation, nrow = 1)
names(hw_tip_traits) = hindwing.data$family

hw_node_traits = matrix(hw_trait_data$color_saturation, nrow = 1)
names(hw_node_traits) = hw_trait_data$node

contMap(my.tree, hw_tip_traits, method='user', anc.states=hw_node_traits)
title('Saturation')
abline(v=1.41438, lty=2)


# HW Richness
hw_tip_traits = matrix(hindwing.data$color_richness, nrow = 1)
names(hw_tip_traits) = hindwing.data$family

hw_node_traits = matrix(hw_trait_data$color_richness, nrow = 1)
names(hw_node_traits) = hw_trait_data$node

contMap(my.tree, hw_tip_traits, method='user', anc.states=hw_node_traits)
title('Richness')
abline(v=1.41438, lty=2)

my.tree
