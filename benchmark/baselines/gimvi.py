import os
import pandas as pd
import numpy as np
import anndata
from scvi.external import GIMVI
import scvi
import traceback

from benchmark.baselines.evaluator import BaselineEvaluator


class GimVIEvaluator(BaselineEvaluator):
    def __init__(self, save_dir, seeds, no_mix=False, rm_data=True, normalized=True, tr_length=200,
                 imputation_genes=[], only_cm=False):
        super().__init__(save_dir=save_dir, seeds=seeds, no_mix=no_mix, rm_data=rm_data, tr_length=tr_length,
                         imputation_genes=imputation_genes)
        self.normalized = normalized
        self.only_cm = only_cm

    def preprocess_write(self, mdata):
        adata_rna = mdata.mod["rna"].copy()
        adata_fish = mdata.mod["fish"].copy()
        all_fish_genes = list(adata_fish.var_names) + self.imputation_genes
        adata_rna = adata_rna[:, all_fish_genes].copy()
        adata_rna.write(os.path.join(self.save_dir, "anndata_rna"))
        adata_fish.write(os.path.join(self.save_dir, "anndata_fish"))

    def run(self):

        for seed in self.seeds:
            scvi.settings.seed = seed
            adata_rna = anndata.read(os.path.join(self.save_dir, "anndata_rna"))
            adata_fish = anndata.read(os.path.join(self.save_dir, "anndata_fish"))
            GIMVI.setup_anndata(adata_rna)
            GIMVI.setup_anndata(adata_fish)
            try:
                model = GIMVI(adata_rna, adata_fish)
                model.train(self.tr_length)
                # get the latent representations for the sequencing and spatial data
                latent_rna, latent_fish = model.get_latent_representation()

                # concatenate to one latent representation
                latents = pd.DataFrame(np.concatenate([latent_rna, latent_fish]),
                                       index=np.concatenate([adata_rna.obs_names, adata_fish.obs_names]))
                latents.to_csv(os.path.join(self.save_dir, "latents_{}.csv".format(seed)))
                _, fish_imputation = model.get_imputed_values(normalized=self.normalized)
                imputation_idxes = [list(adata_rna.var_names).index(gene) for gene in self.imputation_genes]
                imputations = pd.DataFrame(fish_imputation[:, imputation_idxes],
                                           columns=self.imputation_genes, index=adata_fish.obs_names)
                imputations.to_csv(os.path.join(self.save_dir, "imputations_{}.csv".format(seed)))
                with open(os.path.join(self.save_dir, "scaled.txt"), "w") as f:
                    f.write(" ")
            except:
                print("Run of GIMVI on seed {} failed with error message: {}".format(seed, traceback.format_exc()))
        if self.rm_data:
            os.remove(os.path.join(self.save_dir, "anndata_rna"))
            os.remove(os.path.join(self.save_dir, "anndata_fish"))
