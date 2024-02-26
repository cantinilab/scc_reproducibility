import os
import pandas as pd
import anndata
import subprocess

from benchmark.baselines.evaluator import BaselineEvaluator
from benchmark.preprocessing import rename_gene_prots


class RmethodEvaluator(BaselineEvaluator):

    def __init__(self, method_name, save_dir, seeds, no_mix=False, rm_data=True, tr_length=None, imputation_genes=[],
                 n_latent=None):
        super().__init__(save_dir=save_dir, seeds=seeds, no_mix=no_mix, rm_data=rm_data, tr_length=tr_length,
                         imputation_genes=imputation_genes)
        self.n_latent = n_latent
        self.method_name = method_name

    def preprocess_write(self, mdata):
        modalities = list(mdata.mod.keys())
        rna = anndata.AnnData(mdata["rna"].X)
        rna.obs_names = mdata["rna"].obs_names
        rna.var_names = mdata["rna"].var_names
        rna.obs["batch"] = mdata["rna"].obs["batch"]
        rna.write(os.path.join(self.save_dir, "rna.h5ad"))
        if "atac" in modalities:
            self.mod2 = "ATAC"
            if 'gene_activities_norm' in mdata.mod["atac"].obsm.keys():
                atac_genes = anndata.AnnData(mdata.mod["atac"].obsm['gene_activities_norm'])
            else:
                atac_genes = anndata.AnnData(mdata.mod["atac"].obsm['gene_activities'])
                #ga_scale_factor = 1e4
                #sc.pp.normalize_total(atac_genes, target_sum=ga_scale_factor)
                #sc.pp.log1p(atac_genes)
            atac_genes.obs_names = mdata.mod["atac"].obs_names
            atac_genes.var_names = mdata.mod["atac"].uns["gene_activities_var_names"]
            atac_genes.obs["batch"] = mdata["atac"].obs["batch"]
            atac_genes.write(os.path.join(self.save_dir, "atac_genes.h5ad"))
            atac = anndata.AnnData(mdata["atac"].X)
            atac.obs_names = mdata["atac"].obs_names
            atac.var_names = mdata["atac"].var_names
            atac.obs["batch"] = mdata["atac"].obs["batch"]
            atac.write(os.path.join(self.save_dir, "atac.h5ad"))
        elif "adt" in modalities:
            self.mod2 = "ADT"
            adt = anndata.AnnData(mdata["adt"].X)
            adt.obs_names = mdata["adt"].obs_names
            adt.var_names = mdata["adt"].var_names
            gene_prot_names = mdata.uns["gene_prots"]
            adt.var_names = rename_gene_prots(gene_prot_names=gene_prot_names, prots=adt.var_names,
                                              data_name=mdata.uns["data_name"])
            adt.obs["batch"] = mdata["adt"].obs["batch"]
            adt.write(os.path.join(self.save_dir, "adt.h5ad"))
        elif "fish" in modalities:
            self.mod2 = "FISH"
            fish = anndata.AnnData(mdata["fish"].X)
            fish.obs_names = mdata["fish"].obs_names
            fish.var_names = mdata["fish"].var_names
            fish.obs["batch"] = mdata["fish"].obs["batch"]
            fish.write(os.path.join(self.save_dir, "fish.h5ad"))

    def run(self):
        if self.n_latent is None:
            if self.mod2 in ["FISH", "ADT"]:
                self.n_latent = 15
            else:
                self.n_latent = 30
        for seed in self.seeds:
            _ = subprocess.call("Rscript --vanilla ./benchmark/baselines/{}_script.R {} {} {} {} {}"
                                .format(self.method_name, self.mod2, self.save_dir, seed,
                                        "!".join(self.imputation_genes) if len(self.imputation_genes) > 0 else "none",
                                        self.n_latent),
                                shell=True)
            if os.path.exists(os.path.join(self.save_dir, "imputations_{}.csv".format(seed))):
                imputations = pd.read_csv(os.path.join(self.save_dir, "imputations_{}.csv".format(seed)),
                                          index_col=0, header=0).T
                imputations.index = ["{}|FISH".format(name.split(".")[0]) for name in imputations.index]
                imputations.to_csv(os.path.join(self.save_dir, "imputations_{}.csv".format(seed)))
        if self.rm_data:
            for file in os.listdir(self.save_dir):
                if file.endswith(".h5ad"):
                    os.remove(os.path.join(self.save_dir, file))
