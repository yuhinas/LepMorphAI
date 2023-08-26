library(ks)

TPDs <- function(species, traits, samples = NULL, weight = NULL, alpha = 0.95,
                 trait_ranges = NULL, n_divisions = NULL, tolerance = 0.05){
  
  # INITIAL CHECKS:
  # 	1. Compute the number of dimensions (traits):
  traits <- as.matrix(traits)
  dimensions <- ncol(traits)
  if (dimensions > 4) {
    stop("No more than 4 dimensions are supported at this time; reduce the
      number of dimensions")
  }
  #	2. NA's not allowed in traits & species:
  if (any(is.na(traits)) | any(is.na(species))) {
    stop("NA values are not allowed in 'traits' or 'species'")
  }
  #	3. Compute the species or populations upon which calculations will be done:
  if (is.null(samples)) {
    species_base <- species
    if (length(unique(species_base)) == 1){
      type <- "One population_One species"
    } else{
      type <- "One population_Multiple species"
    }
    if (min(table(species_base)) <= dimensions ){
      non_good <- which(table(species_base) <= dimensions)
      stop("You must have more observations (individuals) than traits for all
        species. Consider removing species with too few observations.\n
        Check these species:",
           paste(names(non_good), collapse=" / "))
    }
  } else {
    species_base <- paste(species, samples, sep = ".")
    if (length(unique(species)) == 1){
      type <- "Multiple populations_One species"
    } else{
      type <- "Multiple populations_Multiple species"
    }
    if (min(table(species_base)) <= dimensions ){
      non_good <- which(table(species_base) <= dimensions)
      stop("You must have more observations (individuals) than traits for all
        populations. \n Consider removing populations with too few observations
        or pooling populations of the same species together.\n
        Otherwise, consider using the TPDs2 function.
        Check these populations:",
           paste(names(non_good), collapse=" / "))
    }
  }
  #	4. Define trait ranges:
  if (is.null(trait_ranges)) {
    trait_ranges <- rep (15, dimensions)
  }
  if (class(trait_ranges) != "list") {
    trait_ranges_aux <- trait_ranges
    trait_ranges <- list()
    for (dimens in 1:dimensions) {
      obs_range <- as.numeric(stats::dist(range(traits[, dimens])))
      min_aux <- min(traits[, dimens]) -
        (obs_range * trait_ranges_aux[dimens] / 100)
      max_aux <- max(traits[, dimens]) +
        (obs_range * trait_ranges_aux[dimens] / 100)
      trait_ranges[[dimens]] <- c(min_aux, max_aux)
    }
  }
  #	5. Create the grid of cells in which the density function is evaluated:
  if (is.null(n_divisions)) {
    n_divisions_choose<- c(1000, 200, 50, 25)
    n_divisions<- n_divisions_choose[dimensions]
  }
  grid_evaluate<-list()
  edge_length <- list()
  cell_volume<-1
  for (dimens in 1:dimensions){
    grid_evaluate[[dimens]] <- seq(from = trait_ranges[[dimens]][1],
                                   to = trait_ranges[[dimens]][2],
                                   length=n_divisions)
    edge_length[[dimens]] <- grid_evaluate[[dimens]][2] -
      grid_evaluate[[dimens]][1]
    cell_volume <- cell_volume * edge_length[[dimens]]
  }
  evaluation_grid <- expand.grid(grid_evaluate)
  if (is.null(colnames(traits))){
    names(evaluation_grid) <- paste0("Trait.",1:dimensions)
  } else {
    names(evaluation_grid) <- colnames(traits)
  }
  if (dimensions==1){
    evaluation_grid <- as.matrix(evaluation_grid)
  }
  # Creation of lists to store results:
  results <- list()
  # DATA: To store data and common features
  results$data <- list()
  results$data$evaluation_grid <- evaluation_grid
  results$data$cell_volume <- cell_volume
  results$data$edge_length <- edge_length
  results$data$traits <- traits
  results$data$dimensions <- dimensions
  results$data$species <- species
  if (is.null(samples)){
    results$data$populations <-  NA
  } else{
    results$data$populations <-  species_base
  }
  if (is.null(weight)){
    results$data$weight <-  weight
  } else{
    results$data$weight <-  NA
  }
  results$data$alpha <- alpha
  results$data$pop_traits <- list()
  results$data$pop_weight <- list()
  results$data$type <- type
  results$data$method <- "kernel"
  
  # TPDs: To store TPDs features of each species/population
  results$TPDs<-list()
  
  ########KDE CALCULATIONS
  for (spi in 1:length(unique(species_base))) {
    # Some information messages
    if (spi == 1) { message(paste0("-------Calculating densities for ", type, "-----------\n")) }
    #Data selection
    selected_rows <- which(species_base == unique(species_base)[spi])
    results$data$pop_traits[[spi]] <- traits[selected_rows, ]
    results$data$pop_weight[[spi]] <- length(selected_rows) *
      weight[selected_rows] / sum(weight[selected_rows])
    names(results$data$pop_traits)[spi] <- unique(species_base)[spi]
  }
  
  ######################### change bandwidth seletor ###########################
  
  if (is.null(weight)){
    
    # results$TPDs <- lapply(results$data$pop_traits, ks::kde, eval.points=evaluation_grid)
    
    for (spi in 1:length(unique(species_base))) {
      results$TPDs[[spi]] <- ks::kde(results$data$pop_traits[[spi]], H = Hscv(results$data$pop_traits[[spi]]),
                                     eval.points=evaluation_grid)
    }
    
  } else{
    for (spi in 1:length(unique(species_base))) {
      results$TPDs[[spi]] <- ks::kde(results$data$pop_traits[[spi]], H = Hscv(results$data$pop_traits[[spi]]),
                                     eval.points=evaluation_grid, w = results$data$pop_weight[[spi]])
    }
  }
  
  ######################### change bandwidth seletor ###########################
  
  check_volume <- function(x) sum(x$estimate * cell_volume)
  volumes_checked <- sapply(results$TPDs, check_volume )
  if (any(abs(volumes_checked - 1) > tolerance)) {
    names_fail <- unique(species_base)[ which(abs(volumes_checked - 1) > tolerance)]
    message("Be careful, the integral of the pdf of some cases differ from 1.
      They have been reescaled, but you should consider increasing
      'trait_ranges' \n", paste(names_fail,collapse=" / "))
  }
  rescale_estimate <- function(x){
    x$estimate<- x$estimate / sum(x$estimate)
  }
  results$TPDs<- lapply(results$TPDs, rescale_estimate )
  # Now, we extract the selected fraction of volume (alpha), if necessary
  extract_alpha <- function(x){
    # 1. Order the 'cells' according to their probabilities:
    alphaSpace_aux <- x[order(x, decreasing=T)]
    # 2. Find where does the accumulated sum of ordered probabilities becomes
    #   greater than the threshold (alpha):
    greater_prob <- alphaSpace_aux[which(cumsum(alphaSpace_aux ) > alpha) [1]]
    # 3. Substitute smaller probabilities with 0:
    x[x < greater_prob] <- 0
    # 5. Finally, reescale, so that the total density is finally 1:
    x <- x / sum(x)
    return(x)
  }
  if (alpha < 1) {
    results$TPDs<- lapply(results$TPDs, extract_alpha )
  }
  names(results$TPDs) <- unique(species_base)
  class(results) <- "TPDsp"
  return(results)
}