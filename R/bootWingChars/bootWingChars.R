# install.packages(c('data.table', 'vegan'))
library(data.table)
library(vegan)

result_dt = NULL
wing = 'hw'
#method_str = 'mean'
method_str = 'var'
# target_str = 'shapes'
target_str = 'colors'


wchars = fread(paste0('../../wing_characters/fam_', wing, '_chars_20230518.csv'))
names(wchars)

wchars$group = 'primative'
wchars[wchars$`6groups` %in% c(2,4,5), group:='more_recent']

dim(wchars)

# colors = c('color_saturation', 'color_brightness', 'color_richness', 'color_evenness', 'rgb_red', 'rgb_green', 'rgb_blue', 'distinctive_from_wing_base')
colors = c('color_saturation', 'color_brightness', 'color_richness', 'color_evenness', 'distinctive_from_wing_base')
shapes = c('aspect_ratio', 'secondMoA')
#wchars_colors = wchars[, c('color_saturation', 'color_brightness', 'color_richness', 'color_evenness', 'rgb_red', 'rgb_green', 'rgb_blue', 'distinctive_from_wing_base')]


target = NULL
if (target_str == 'shapes') {
  target = shapes
} else if (target_str == 'colors') {
  target = colors
}

library(boot)
getvar = function(col, data, indices, method) {
  return (method(as.matrix(data[indices[1:25], ..col])))
}

if (method_str=='var') {
  method = var  
} else if (method_str=='mean') {
  method = mean
}

R = 10000
for (col in target) {
  print(col)
  result.primative = boot(data=wchars[group=='primative'], getvar, col=col, R=R, method=method)
  result.more_recent = boot(data=wchars[group=='more_recent'], getvar, col=col, R=R, method=method)
  # boot.ci(result.primative, type='bca')
  # boot.ci(result.more_recent, type='bca')
  result_dt = rbind(result_dt, data.table(wing=wing, char=col, method=method_str, rgtp=sum(result.primative$t < result.more_recent$t), rltp=sum(result.primative$t > result.more_recent$t), R=R))
}

#result_dt
#fwrite(result_dt, 'wing_chars_bootstrap_20230518.csv', sep='\t')
