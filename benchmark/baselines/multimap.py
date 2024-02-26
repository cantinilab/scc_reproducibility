import os

import scanpy as sc
import muon as mu
import pandas as pd
import numpy as np
import anndata
import traceback
from scipy.sparse import issparse
import MultiMAP

from benchmark.preprocessing import normalize_rna, rename_gene_prots
from benchmark.baselines.evaluator import BaselineEvaluator


class MultimapEvaluator(BaselineEvaluator):

    def preprocess_write(self, mdata):
        modalities = list(mdata.mod.keys())
        if "atac" in modalities:
            atac_peaks = mdata.mod["atac"].copy()
            ga_scale_factor = None
            if 'gene_activities_norm' in atac_peaks.obsm.keys():
                atac_genes = anndata.AnnData(atac_peaks.obsm['gene_activities_norm'])
            else:
                atac_genes = anndata.AnnData(atac_peaks.obsm['gene_activities'])
                ga_scale_factor = 1e4
                sc.pp.normalize_total(atac_genes, target_sum=ga_scale_factor)
                sc.pp.log1p(atac_genes)
            atac_genes.var_names = atac_peaks.uns["gene_activities_var_names"]
            atac_genes.obs = atac_peaks.obs
            MultiMAP.TFIDF_LSI(atac_peaks)
            atac_genes.obsm['X_pca'] = atac_peaks.obsm['X_lsi'].copy()

            # load rna
            rna = mdata.mod["rna"].copy()
            # normalize rna just like gene activities
            if ga_scale_factor is None:
                if issparse(atac_genes.X):
                    ga_scale_factor = np.median(np.sum(np.exp(np.array(atac_genes.X.todense())) - 1, axis=1))
                else:
                    ga_scale_factor = np.median(np.sum(np.exp(np.array(atac_genes.X)) - 1, axis=1))
            normalize_rna(rna, target_sum=ga_scale_factor)
            # scale rna  for PCA dim reduction
            rna_pca = rna.copy()
            sc.pp.scale(rna_pca)
            sc.pp.pca(rna_pca)
            rna.obsm['X_pca'] = rna_pca.obsm['X_pca'].copy()

            rna.write(os.path.join(self.save_dir, "adata_1"))
            atac_genes.write(os.path.join(self.save_dir, "adata_2"))
        elif "adt" in modalities:
            adt = mdata.mod["adt"].copy()
            rna = mdata.mod["rna"].copy()
            normalize_rna(rna)
            mu.prot.pp.clr(adt, axis=1)

            # scale rna/adt for PCA dim reduction
            rna_pca = rna.copy()
            sc.pp.scale(rna_pca)
            sc.pp.pca(rna_pca)
            rna.obsm["X_pca"] = rna_pca.obsm["X_pca"].copy()

            adt_pca = adt.copy()
            sc.pp.scale(adt_pca)
            sc.pp.pca(adt_pca)
            adt.obsm["X_pca"] = adt_pca.obsm["X_pca"].copy()

            gene_prot_names = mdata.uns["gene_prots"]
            adt.var_names = rename_gene_prots(gene_prot_names=gene_prot_names, prots=adt.var_names,
                                              data_name=mdata.uns["data_name"])
            rna.write(os.path.join(self.save_dir, "adata_1"))
            adt.write(os.path.join(self.save_dir, "adata_2"))
        elif "fish" in modalities:
            genes = mdata.mod["rna"].var_names
            fish_genes = mdata.mod["fish"].var_names

            common_genes = []
            for gene in fish_genes:
                if gene in genes:
                    common_genes.append(gene)
            rna = mdata.mod["rna"][:, common_genes].copy()
            fish = mdata.mod["fish"].copy()
            # normalize rna
            normalize_rna(rna, target_sum=1000)
            # normalize fish
            normalize_rna(fish, target_sum=1000)

            # scale rna/adt for PCA dim reduction
            rna_pca = rna.copy()
            sc.pp.scale(rna_pca)
            sc.pp.pca(rna_pca)
            rna.obsm["X_pca"] = rna_pca.obsm["X_pca"].copy()

            fish_pca = fish.copy()
            sc.pp.scale(fish_pca)
            sc.pp.pca(fish_pca)
            fish.obsm["X_pca"] = fish_pca.obsm["X_pca"].copy()
            rna.write(os.path.join(self.save_dir, "adata_1"))
            fish.write(os.path.join(self.save_dir, "adata_2"))

    def run(self):
        adata_1 = anndata.read(os.path.join(self.save_dir, "adata_1"))
        adata_2 = anndata.read(os.path.join(self.save_dir, "adata_2"))
        cnct_adata = anndata.AnnData(np.zeros((adata_1.shape[0] + adata_2.shape[0], 1)))
        cnct_adata.obs.index = list(adata_1.obs.index) + list(adata_2.obs.index)
        # handle multiple batches
        list_adata = [adata_1[adata_1.obs["batch"] == batch] for batch in np.unique(adata_1.obs["batch"])] + \
                     [adata_2[adata_2.obs["batch"] == batch] for batch in np.unique(adata_2.obs["batch"])]
        for seed in self.seeds:
            try:
                mmap_adata = MultiMAP.Integration(adatas=list_adata, use_reps=len(list_adata) * ["X_pca"], seed=seed)
                latents = pd.DataFrame(mmap_adata.obsm["X_multimap"], index=mmap_adata.obs.index)
                latents.to_csv(os.path.join(self.save_dir, "latents_{}.csv".format(seed)))
                cnct_adata.obsp["{}_connectivities".format(seed)] = mmap_adata[cnct_adata.obs.index].obsp[
                    "connectivities"]
                del mmap_adata
            except:
                print("Run of MultiMAP on seed {} failed with error message: {}".format(seed, traceback.format_exc()))
        if len(self.imputation_genes) == 0:
            cnct_adata.write(os.path.join(self.save_dir, "cnct_adata"))
        if self.rm_data:
            os.remove(os.path.join(self.save_dir, "adata_1"))
            os.remove(os.path.join(self.save_dir, "adata_2"))
