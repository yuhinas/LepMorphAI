library(ape)
library(caper)
library(data.table)
library(phytools)

#mvMorph
library(mvMORPH);
#first read in the tree (original tree 83 tips)
comp.tree <- read.tree('moth.tre');

#remove outgroups
og_names = comp.tree$tip.label[1:9]

lep.tree = drop.tip(comp.tree, og_names)
lep.tree$edge.length = lep.tree$edge.length * .1

#forewing data set to be used for analuses
forewing <- fread('moth_geometry_with_meta.csv', sep='\t');

fam_with_phylo = forewing$family

tip_colors = setNames(rep('black', length(lep.tree$tip.label)), lep.tree$tip.label)

for (fam in fam_with_phylo) {
  tip_colors[fam] = 'red'
}

plot(lep.tree, tip.color = tip_colors, cex = .6)
