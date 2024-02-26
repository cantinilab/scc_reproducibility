library(anndata)
library(rliger)

args = commandArgs(trailingOnly=TRUE)
mod2 <- args[1]
save.path <- args[2]
seed <- args[3]
if (args[4] == "none"){
    imputation.genes <- c()
} else{
    imputation.genes <- strsplit(args[3], "!")
}
latent.dim <- strtoi(args[5])

print(latent.dim)
data.1 <- anndata::read_h5ad(paste0(save.path, "/rna.h5ad"))

if (mod2 == "ATAC"){
  data.2 <- anndata::read_h5ad(paste0(save.path, "/atac_genes.h5ad"))
} else {

  if (mod2 == "ADT"){
    data.2 <- anndata::read_h5ad(paste0(save.path, "/adt.h5ad"))
  }
  else { # FISH
    data.2 <- anndata::read_h5ad(paste0(save.path, "/fish.h5ad"))
  }
}
data_list = list()
common_vars <- intersect(data.1$var_names, data.2$var_names)
for (batch_n in levels(data.1$obs$batch)){
    data_list[[batch_n]] <- Matrix::t(data.1[data.1$obs$batch == batch_n, common_vars]$X)
}

for (batch_n in levels(data.2$obs$batch)){
    data_list[[batch_n]] <- Matrix::t(data.2[data.2$obs$batch == batch_n, common_vars]$X)
}

coembed_liger <- createLiger(data_list)
coembed_liger <- normalize(coembed_liger)
if (mod2 == "ATAC"){
    coembed_liger <- selectGenes(coembed_liger, datasets.use = seq(nlevels(data.1$obs$batch)))
} else{
    hvg <- rownames(coembed_liger@raw.data[[levels(data.2$obs$batch)[1]]])
    for (batch_n in c(levels(data.1$obs$batch), levels(data.2$obs$batch))){
        hvg <- intersect(hvg, rownames(coembed_liger@raw.data[[batch_n]]))
    }
    coembed_liger@var.genes <- hvg
}
coembed_liger <- scaleNotCenter(coembed_liger)
coembed_liger <- optimizeALS(coembed_liger, k = latent.dim, rand.seed = strtoi(seed))#, max.iters = 3)
coembed_liger <- quantile_norm(coembed_liger, rand.seed = strtoi(seed))
write.csv(coembed_liger@H.norm, paste0(save.path, "/latents_", seed, ".csv"))
