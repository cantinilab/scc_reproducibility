import os

import scanpy as sc
import pandas as pd
import numpy as np
import anndata
import uniport as up
import traceback
from scipy.sparse import csr_matrix

from benchmark.baselines.evaluator import BaselineEvaluator
from benchmark.preprocessing import rename_gene_prots


class UniportEvaluator(BaselineEvaluator):

    def preprocess_write(self, mdata):
        adata_rna = mdata.mod["rna"].copy()
        adata_rna.obs['domain_id'] = adata_rna.obs["batch"]
        adata_rna.obs['domain_id'] = adata_rna.obs['domain_id'].astype('category')
        adata_rna.obs['source'] = 'RNA'

        # Load second modality and format them as raw counts
        if "atac" in list(mdata.mod.keys()):
            if "gene_activities" in mdata.mod["atac"].obsm.keys():
                adata_atac_X = mdata.mod["atac"].obsm["gene_activities"]
            else:
                try:
                    adata_atac_X = np.exp(mdata.mod["atac"].obsm["gene_activities_norm"]) - 1
                except:
                    adata_atac_X = np.exp(mdata.mod["atac"].obsm["gene_activities_norm"].todense()) - 1
            adata_2 = anndata.AnnData(adata_atac_X)
            adata_2.var_names = mdata.mod["atac"].uns["gene_activities_var_names"]
            adata_2.obs.index = mdata.mod["atac"].obs.index
            adata_2.obs["celltype"] = mdata.mod["atac"].obs["celltype"].values
            adata_2.obs["batch"] = mdata.mod["atac"].obs["batch"].values
            adata_2.obs['source'] = 'ATAC'
        elif "fish" in list(mdata.mod.keys()):
            adata_2 = mdata.mod["fish"].copy()
            adata_2.obs['source'] = 'FISH'
            adata_2.obs['batch'] = 0
            adata_rna.obs['batch'] = 1
        elif "adt" in list(mdata.mod.keys()):
            adata_2 = mdata.mod["adt"].copy()
            adata_2.obs['source'] = 'ADT'
            adata_2.obs['batch'] = adata_2.obs["batch"].astype('category')
            gene_prot_names = mdata.uns["gene_prots"]
            adata_2.var_names = rename_gene_prots(gene_prot_names=gene_prot_names, prots=adata_2.var_names,
                                                  data_name=mdata.uns["data_name"])

        adata_2.obs['domain_id'] = adata_2.obs["batch"].astype('category')
        adata_2.obs['domain_id'] = adata_2.obs['domain_id'].astype('category')

        # take cells from both modalities with common features only and log normalize before taking hvg
        adata_cm = adata_2.concatenate(adata_rna, join='inner', batch_key='domain_id')
        sc.pp.normalize_total(adata_cm)
        sc.pp.log1p(adata_cm)
        sc.pp.highly_variable_genes(adata_cm, n_top_genes=2000, inplace=False, subset=True)
        up.batch_scale(adata_cm)

        # log normalize rna data and take hvg
        sc.pp.normalize_total(adata_rna)
        sc.pp.log1p(adata_rna)
        if len(self.imputation_genes) > 0:  # Make sure that imputation genes are kept in the rna anndata
            sc.pp.highly_variable_genes(adata_rna, n_top_genes=2000, inplace=True, subset=False)
            for gene in self.imputation_genes:
                adata_rna.var["highly_variable"][gene] = True
            adata_rna = adata_rna[:, adata_rna.var["highly_variable"]].copy()
        else:
            sc.pp.highly_variable_genes(adata_rna, n_top_genes=2000, inplace=False, subset=True)
        up.batch_scale(adata_rna)

        # log normalize second modality and take hvg
        sc.pp.normalize_total(adata_2)
        sc.pp.log1p(adata_2)
        sc.pp.highly_variable_genes(adata_2, n_top_genes=2000, inplace=False, subset=True)
        up.batch_scale(adata_2)

        # convert anndatas to sparse format before saving
        adata_cm.X = csr_matrix(adata_cm.X)
        adata_rna.X = csr_matrix(adata_rna.X)
        adata_2.X = csr_matrix(adata_2.X)
        adata_cm.write(os.path.join(self.save_dir, "adata_cm"))
        adata_rna.write(os.path.join(self.save_dir, "adata_1"))
        adata_2.write(os.path.join(self.save_dir, "adata_2"))

    def run(self):
        adata_cm = anndata.read(os.path.join(self.save_dir, "adata_cm"))
        adata_1 = anndata.read(os.path.join(self.save_dir, "adata_1"))
        adata_2 = anndata.read(os.path.join(self.save_dir, "adata_2"))
        if self.no_mix:
            lambda_ot = 0.
        else:
            lambda_ot = 1.

        for seed in self.seeds:
            try:
                rez_adata = up.Run(adatas=[adata_2, adata_1], adata_cm=adata_cm, lambda_s=1.0, lambda_ot=lambda_ot,
                                   seed=seed, outdir=self.save_dir, iteration=self.tr_length, num_workers=0,
                                   batch_size=min(256, adata_cm.shape[0]))
                if len(self.imputation_genes) > 0:
                    adata_predict = up.Run(adata_cm=adata_2, out='predict', pred_id=1, outdir=self.save_dir)
                    imputation_idxes = [list(adata_1.var_names).index(gene) for gene in self.imputation_genes]
                    imputations = pd.DataFrame(adata_predict.obsm["predict"][:, imputation_idxes],
                                               columns=self.imputation_genes, index=adata_2.obs_names)
                    imputations.to_csv(os.path.join(self.save_dir, "imputations_{}.csv".format(seed)))

                # remove dataset id suffix to obs_names
                latent_index = [name[:-2] for name in rez_adata.obs.index]
                latents = pd.DataFrame(rez_adata.obsm["latent"], index=latent_index)
                latents.to_csv(os.path.join(self.save_dir, "latents_{}.csv".format(seed)))
                del rez_adata
            except:
                print("Run of Uniport on seed {} failed with error message: {}".format(seed, traceback.format_exc()))
        if self.rm_data:
            os.remove(os.path.join(self.save_dir, "adata_cm"))
            os.remove(os.path.join(self.save_dir, "adata_1"))
            os.remove(os.path.join(self.save_dir, "adata_2"))
