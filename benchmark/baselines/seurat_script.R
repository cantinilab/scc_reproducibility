library(Seurat)
library(ggplot2)
library(irlba)
library(Matrix)
library(stats)
source("benchmark/baselines/signac_utils.R")

args = commandArgs(trailingOnly=TRUE)
mod2 <- args[1]
save.path <- args[2]
seed <- args[3]
if (args[4] == "none"){
    imputation.genes <- c()
} else{
    imputation.genes <- strsplit(args[4], "!")
}
latent.dim <- strtoi(args[5])
use.batch <- FALSE
print(latent.dim)
data.1 <- MuDataSeurat::ReadH5AD(paste0(save.path, "/rna.h5ad"))

if (mod2 == "ATAC"){
  data.2 <- MuDataSeurat::ReadH5AD(paste0(save.path, "/atac.h5ad"))
  data.2 <- RenameAssays(data.2, RNA="ATAC")
  mod2.cross <- "ACTIVITY"
  data.2@assays$ACTIVITY <- MuDataSeurat::ReadH5AD(paste0(save.path, "/atac_genes.h5ad"))@assays$RNA
  DefaultAssay(data.2) <- "ACTIVITY"
  data.2 <- NormalizeData(data.2)
  data.2 <- ScaleData(data.2, features = rownames(data.2))
  DefaultAssay(data.2) <- "ATAC"
  data.2 <- RunTFIDF(data.2)
  data.2 <- FindTopFeatures(data.2, min.cutoff = "q0")
  data.2 <- RunSVD(data.2)

  if (nlevels(data.2@meta.data$batch) > 1 && use.batch){
      batch2.list <- SplitObject(data.2, split.by = "batch")
      integration.anchors <- FindIntegrationAnchors(object.list = batch2.list,
                                                    anchor.features = rownames(data.2), reduction = "rlsi",
                                                    dims = 2:latent.dim)
      integrated <- IntegrateEmbeddings(
        anchorset = integration.anchors,
        reductions = data.2[["lsi"]],
        new.reduction.name = "integrated_lsi",
        dims.to.integrate = 1:latent.dim)
      mod2.red <- "integrated_lsi"
      data.2@reductions$integrated_lsi <- integrated@reductions$integrated_lsi
  } else {
      mod2.red <- "lsi"
  }

} else {
  mod2.cross <- mod2
  mod2.red <- "pca"

  if (mod2 == "ADT"){
    data.2 <- MuDataSeurat::ReadH5AD(paste0(save.path, "/adt.h5ad"))
    if (nlevels(data.2@meta.data$batch) > 1 && use.batch){
        batch2.list <- SplitObject(data.2, split.by = "batch")
        batch2.list <- lapply(X = batch2.list, FUN = function(x) {
            x <- NormalizeData(x, normalization.method = "CLR", margin = 2)
        })
        adt.batch.features <- SelectIntegrationFeatures(object.list = batch2.list)
        adt.batch.anchors <- FindIntegrationAnchors(object.list = batch2.list, anchor.features = adt.batch.features)
        data.2 <- IntegrateData(anchorset = adt.batch.anchors)
        mod2.cross <- "integrated"
    } else {
        data.2 <- NormalizeData(data.2, normalization.method = "CLR", margin = 2)
    }
  }
  else { # FISH
    data.2 <- MuDataSeurat::ReadH5AD(paste0(save.path, "/fish.h5ad"))
    data.2 <- NormalizeData(data.2)
  }
  data.2 <- RenameAssays(data.2, RNA=mod2)
  data.2 <- ScaleData(data.2)
  data.2 <- RunPCA(data.2, npcs=latent.dim, features=rownames(data.2))
}

data.2 <- RunUMAP(data.2, reduction = mod2.red, dims = 1:latent.dim, reduction.name = paste0("umap", mod2))
p2 <- DimPlot(data.2, label = FALSE)
ggsave(paste0(save.path, "/", mod2, "_umap.png"), plot=p2)

if (nlevels(data.1@meta.data$batch) > 1 && use.batch){
    batch1.list <- SplitObject(data.1, split.by = "batch")
    batch1.list <- lapply(X = batch1.list, FUN = function(x) {
        x <- NormalizeData(x)
        x <- FindVariableFeatures(x, selection.method = "vst", nfeatures = 2000)
    })
    rna.batch.features <- SelectIntegrationFeatures(object.list = batch1.list)
    rna.batch.anchors <- FindIntegrationAnchors(object.list = batch1.list, anchor.features = rna.batch.features)
    data.1 <- IntegrateData(anchorset = rna.batch.anchors)
    assay.1 <- "integrated"
} else {
    data.1 <- NormalizeData(data.1)
    data.1 <- FindVariableFeatures(data.1, selection.method = "vst", nfeatures = 2000)
    assay.1 <- "RNA"
}
data.1 <- ScaleData(data.1)
data.1 <- RunPCA(data.1)
data.1 <- RunUMAP(data.1, reduction = "pca", dims = 2:latent.dim, reduction.name = "umapGEX")
p1 <- DimPlot(data.1, label = FALSE)
ggsave(paste0(save.path, "/GEX_umap.png"), plot=p1)


if (mod2 == "FISH" || mod2 == "ADT") {
    for (gene in rownames(data.2[[mod2.cross]])) {
        if (!(gene %in% VariableFeatures(data.1))) {
            VariableFeatures(data.1) <- append(VariableFeatures(data.1), gene)
        }
    }
}
transfer.anchors <- FindTransferAnchors(reference = data.1, query = data.2,
                                        features = VariableFeatures(data.1),
                                        reference.assay = assay.1,
                                        query.assay = mod2.cross,
                                        reduction = "cca")

if (mod2 == "FISH" & length(imputation.genes) > 0){
    refdata <- GetAssayData(data.1, assay =assay.1, slot = "data")[unlist(imputation.genes), ]
    imputation <- TransferData(anchorset = transfer.anchors, refdata = refdata,
                               weight.reduction = data.2[[mod2.red]],
                               dims = 2:latent.dim)
    data.2[["RNA"]] <- imputation
    write.csv(data.2[["RNA"]]@data, paste0(save.path, "/imputations_", seed, ".csv"))
}
genes.use <- VariableFeatures(data.1)
refdata <- GetAssayData(data.1, assay = "RNA", slot = "data")[genes.use, ]
imputation <- TransferData(anchorset = transfer.anchors, refdata = refdata, weight.reduction = data.2[[mod2.red]],
    dims = 2:latent.dim, k.weight = 50)

data.2[["RNA"]] <- imputation
coembed <- merge(x = data.1, y = data.2)
coembed <- ScaleData(coembed, features = genes.use, do.scale = FALSE)
coembed <- RunPCA(coembed, features = genes.use, verbose = FALSE, npcs=latent.dim)
write.csv(Embeddings(coembed[["pca"]]), paste0(save.path, "/latents_", seed, ".csv"))
