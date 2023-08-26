## The deep learning model training and data analyses can be done by running following scripts.

### main.py
Train DFC-VSC.

### encoding_vsc_repeatN.py 
Extract 512D features from images.

### grid_explore_vsc_dimensions.py 
Visualize 512D features.

### predict_sufam_with_reparams_no_neg_codes.py 
Train simple subfamily classfier for finding key features.

### do_paper_protocals.py
Run line by line on IDE such as VSCode for the main analysis.

### ./moth_geometry/AR_and_2ndMoA.ipynb
Calculate aspect ratio and second moment of area of lep images generated from the 3*3 points sampled on each two of pc axes

### ./moth_geometry/AR_and_2ndMoA_base.ipynb 
Calculate aspect ratio and second moment of area of the typical form of each family

### ./moth_geometry/WingCharacters.ipynb
Calculate traditional color and shape characters of lep images generated from the 3*3 points sampled on each two of pc axes

### ./wing_chars_fam.py
Calculate traditional color and shape characters of the typical form of each family

### ./wing_chars_plots.py
Boxplot and scatter plot for wing characters of the primitive and recent groups

### ./R/hypervolume/fam_phylo_hv.R
Calculate the hypervolumes of the primitive and recent groups

### ./R/hypervolume/fam_phylo_hv_66Ma.R
Calculate the hypervolumes of sub groups in the recent group delineated by 66Ma

### ./R/hypervolume/fam_pc_permanova.R
PERMANOVA on the primitive and recent groups on the PC axes

### ./R/TPD/cal_TPD_with_imputed.R
The TPD analysis on the primitive and recent groups on the PC axes

### ./R/TPD/plotTPD.R and ./R/TPD/TPDs.R
We edit these two file from the TPD package of better customized plotting

### ./R/bootWingChars/bootWingChars.R
Use bootstrapping to estimate and compare means and variance in wing color patterning
