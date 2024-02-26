library(Seurat)
library(ggplot2)
library(irlba)
library(Matrix)
library(stats)


SetIfNull <- function(x, y) {
  if(is.null(x = x)){
    return(y)
  } else {
    return(x)
  }
}

RunTFIDF.default <- function(
  object,
  assay = NULL,
  method = 1,
  scale.factor = 1e4,
  idf = NULL,
  verbose = TRUE,
  ...
) {
  if (inherits(x = object, what = "data.frame")) {
    object <- as.matrix(x = object)
  }
  if (!inherits(x = object, what = "CsparseMatrix")) {
    object <- as(object = object, Class = "CsparseMatrix")
  }
  if (verbose) {
    message("Performing TF-IDF normalization")
  }
  npeaks <- colSums(x = object)
  if (any(npeaks == 0)) {
    warning("Some cells contain 0 total counts")
  }
  if (method == 4) {
    tf <- object
  } else {
    tf <- tcrossprod(x = object, y = Diagonal(x = 1 / npeaks))
  }
  if (!is.null(x = idf)) {
    precomputed_idf <- TRUE
    if (!inherits(x = idf, what = "numeric")) {
      stop("idf parameter must be a numeric vector")
    }
    if (length(x = idf) != nrow(x = object)) {
      stop("Length of supplied IDF vector does not match",
           " number of rows in input matrix")
    }
    if (any(idf == 0)) {
      stop("Supplied IDF values cannot be zero")
    }
    if (verbose) {
      message("Using precomputed IDF vector")
    }
  } else {
    precomputed_idf <- FALSE
    rsums <- rowSums(x = object)
    if (any(rsums == 0)) {
      warning("Some features contain 0 total counts")
    }
    idf <- ncol(x = object) / rsums
  }

  if (method == 2) {
    if (!precomputed_idf) {
      idf <- log(1 + idf)
    }
  } else if (method == 3) {
    slot(object = tf, name = "x") <- log1p(
      x = slot(object = tf, name = "x") * scale.factor
    )
    if (!precomputed_idf) {
      idf <- log(1 + idf)
    }
  }
  norm.data <- Diagonal(n = length(x = idf), x = idf) %*% tf
  if (method == 1) {
    slot(object = norm.data, name = "x") <- log1p(
      x = slot(object = norm.data, name = "x") * scale.factor
    )
  }
  colnames(x = norm.data) <- colnames(x = object)
  rownames(x = norm.data) <- rownames(x = object)
  # set NA values to 0
  vals <- slot(object = norm.data, name = "x")
  vals[is.na(x = vals)] <- 0
  slot(object = norm.data, name = "x") <- vals
  return(norm.data)
}

#' @rdname RunTFIDF
#' @method RunTFIDF Assay
#' @export
#' @concept preprocessing
#' @examples
#' RunTFIDF(atac_small[['peaks']])
RunTFIDF.Assay <- function(
  object,
  assay = NULL,
  method = 1,
  scale.factor = 1e4,
  idf = NULL,
  verbose = TRUE,
  ...
) {
  new.data <- RunTFIDF.default(
    object = GetAssayData(object = object, slot = "counts"),
    method = method,
    assay = assay,
    scale.factor = scale.factor,
    idf = idf,
    verbose = verbose,
    ...
  )
  new.data <- as(object = new.data, Class = "CsparseMatrix")
  object <- SetAssayData(
    object = object,
    slot = "data",
    new.data = new.data
  )
  return(object)
}

#' @param assay Name of assay to use
#' @rdname RunTFIDF
#' @method RunTFIDF Seurat
#' @export
#' @concept preprocessing
#' @examples
#' RunTFIDF(object = atac_small)
RunTFIDF <- function(
  object,
  assay = NULL,
  method = 1,
  scale.factor = 1e4,
  idf = NULL,
  verbose = TRUE,
  ...
) {
  assay <- SetIfNull(x = assay, y = DefaultAssay(object))
  assay.data <- object[[assay]]
  assay.data <- RunTFIDF.Assay(
    object = assay.data,
    assay = assay,
    method = method,
    scale.factor = scale.factor,
    idf = idf,
    verbose = verbose,
    ...
  )
  object[[assay]] <- assay.data
  return(object)
}

FindTopFeatures.default <- function(
  object,
  assay = NULL,
  min.cutoff = "q5",
  verbose = TRUE,
  ...
) {
  featurecounts <- rowSums(x = object)
  e.dist <- ecdf(x = featurecounts)
  hvf.info <- data.frame(
    row.names = names(x = featurecounts),
    count = featurecounts,
    percentile = e.dist(featurecounts)
  )
  hvf.info <- hvf.info[order(hvf.info$count, decreasing = TRUE), ]
  return(hvf.info)
}

#' @rdname FindTopFeatures
#' @importFrom SeuratObject GetAssayData VariableFeatures
#' @export
#' @method FindTopFeatures Assay
#' @concept preprocessing
#' @examples
#' FindTopFeatures(object = atac_small[['peaks']])
FindTopFeatures.Assay <- function(
  object,
  assay = NULL,
  min.cutoff = "q5",
  verbose = TRUE,
  ...
) {
  data.use <- GetAssayData(object = object, slot = "counts")
  if (IsMatrixEmpty(x = data.use)) {
    if (verbose) {
      message("Count slot empty")
    }
    return(object)
  }
  hvf.info <- FindTopFeatures.default(
    object = data.use,
    assay = assay,
    min.cutoff = min.cutoff,
    verbose = verbose,
    ...
  )
  object[[names(x = hvf.info)]] <- hvf.info
  if (is.null(x = min.cutoff)) {
    VariableFeatures(object = object) <- rownames(x = hvf.info)
  } else if (is.numeric(x = min.cutoff)) {
    VariableFeatures(object = object) <- rownames(
      x = hvf.info[hvf.info$count > min.cutoff, ]
    )
  } else if (is.na(x = min.cutoff)) {
    # don't change the variable features
    return(object)
  } else {
    percentile.use <- as.numeric(
      x = sub(pattern = "q", replacement = "", x = as.character(x = min.cutoff))
    ) / 100
    VariableFeatures(object = object) <- rownames(
      x = hvf.info[hvf.info$percentile > percentile.use, ]
    )
  }
  return(object)
}

#' @rdname FindTopFeatures
#' @importFrom SeuratObject DefaultAssay
#' @export
#' @concept preprocessing
#' @method FindTopFeatures Seurat
#' @examples
#' FindTopFeatures(atac_small)
FindTopFeatures <- function(
  object,
  assay = NULL,
  min.cutoff = "q5",
  verbose = TRUE,
  ...
) {
  assay <- SetIfNull(x = assay, y = DefaultAssay(object))
  assay.data <- object[[assay]]
  assay.data <- FindTopFeatures.Assay(
    object = assay.data,
    assay = assay,
    min.cutoff = min.cutoff,
    verbose = verbose,
    ...
  )
  object[[assay]] <- assay.data
  return(object)
}

RunSVD.default <- function(
  object,
  assay = NULL,
  n = 50,
  scale.embeddings = TRUE,
  reduction.key = "LSI_",
  scale.max = NULL,
  verbose = TRUE,
  irlba.work = n * 3,
  tol = 1e-05,
  ...
) {
  n <- min(n, (ncol(x = object) - 1))
  if (verbose) {
    message("Running SVD")
  }
  components <- irlba(A = t(x = object), nv = n, work = irlba.work, tol = tol)
  feature.loadings <- components$v
  sdev <- components$d / sqrt(x = max(1, nrow(x = object) - 1))
  cell.embeddings <- components$u
  if (scale.embeddings) {
    if (verbose) {
      message("Scaling cell embeddings")
    }
    embed.mean <- apply(X = cell.embeddings, MARGIN = 2, FUN = mean)
    embed.sd <- apply(X = cell.embeddings, MARGIN = 2, FUN = sd)
    norm.embeddings <- t((t(cell.embeddings) - embed.mean) / embed.sd)
    if (!is.null(x = scale.max)) {
      norm.embeddings[norm.embeddings > scale.max] <- scale.max
      norm.embeddings[norm.embeddings < -scale.max] <- -scale.max
    }
  } else {
    norm.embeddings <- cell.embeddings
  }
  rownames(x = feature.loadings) <- rownames(x = object)
  colnames(x = feature.loadings) <- paste0(
    reduction.key, seq_len(length.out = n)
  )
  rownames(x = norm.embeddings) <- colnames(x = object)
  colnames(x = norm.embeddings) <- paste0(
    reduction.key, seq_len(length.out = n)
  )
  reduction.data <- CreateDimReducObject(
    embeddings = norm.embeddings,
    loadings = feature.loadings,
    assay = assay,
    stdev = sdev,
    key = reduction.key,
    misc = components
  )
  return(reduction.data)
}

#' @param features Which features to use. If NULL, use variable features
#'
#' @rdname RunSVD
#' @importFrom SeuratObject VariableFeatures GetAssayData
#' @export
#' @concept dimension_reduction
#' @method RunSVD Assay
#' @examples
#' RunSVD(atac_small[['peaks']])
RunSVD.Assay <- function(
  object,
  assay = NULL,
  features = NULL,
  n = 50,
  reduction.key = "LSI_",
  scale.max = NULL,
  verbose = TRUE,
  ...
) {
  features <- SetIfNull(x = features, y = VariableFeatures(object = object))
  data.use <- GetAssayData(
    object = object,
    slot = "data"
  )[features, ]
  reduction.data <- RunSVD.default(
    object = data.use,
    assay = assay,
    features = features,
    n = n,
    reduction.key = reduction.key,
    scale.max = scale.max,
    verbose = verbose,
    ...
  )
  return(reduction.data)
}

#' @param reduction.name Name for stored dimension reduction object.
#' Default 'svd'
#' @rdname RunSVD
#' @export
#' @concept dimension_reduction
#' @examples
#' RunSVD(atac_small)
#' @method RunSVD Seurat
RunSVD <- function(
  object,
  assay = NULL,
  features = NULL,
  n = 50,
  reduction.key = "LSI_",
  reduction.name = "lsi",
  scale.max = NULL,
  verbose = TRUE,
  ...
) {
  assay <- SetIfNull(x = assay, y = DefaultAssay(object = object))
  assay.data <- object[[assay]]
  reduction.data <- RunSVD.Assay(
    object = assay.data,
    assay = assay,
    features = features,
    n = n,
    reduction.key = reduction.key,
    scale.max = scale.max,
    verbose = verbose,
    ...
  )
  object[[reduction.name]] <- reduction.data
  return(object)
}