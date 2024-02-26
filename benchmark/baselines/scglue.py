import os

import scanpy as sc
import muon as mu
import pandas as pd
import numpy as np
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
import anndata
import traceback
import scglue

from benchmark.baselines.evaluator import BaselineEvaluator


class scGlueEvaluator(BaselineEvaluator):
    def __init__(self, save_dir, seeds, no_mix=False, rm_data=True, cpu_only=True, tr_length=False,
                 imputation_genes=[], latent_dim=50):
        super().__init__(save_dir=save_dir, seeds=seeds, no_mix=no_mix, rm_data=rm_data, tr_length=tr_length,
                         imputation_genes=imputation_genes)
        if cpu_only:
            scglue.config.CPU_ONLY = True
        self.latent_dim = latent_dim

    def preprocess_write(self, mdata):
        rna = mdata.mod["rna"].copy()
        rna.layers["counts"] = rna.X.copy()
        if "atac" in mdata.mod.keys():
            rna = rna[:, rna.var["strand"] != "nan"].copy()

        sc.pp.highly_variable_genes(rna, n_top_genes=2000, flavor="seurat_v3")
        # Make sure that imputation genes are kept in the rna anndata
        if len(self.imputation_genes) > 0:
            sc.pp.highly_variable_genes(rna, n_top_genes=2000, inplace=True, subset=False, flavor="seurat_v3")
            for gene in self.imputation_genes:
                rna.var["highly_variable"][gene] = True
            rna = rna[:, rna.var["highly_variable"]].copy()
        else:
            sc.pp.highly_variable_genes(rna, n_top_genes=2000, inplace=False, subset=True, flavor="seurat_v3")
        sc.pp.normalize_total(rna)
        sc.pp.log1p(rna)
        sc.pp.scale(rna)
        sc.tl.pca(rna, n_comps=100, svd_solver="auto")

        if "atac" in mdata.mod.keys():
            atac = mdata.mod["atac"].copy()
            atac.layers["counts"] = atac.X.copy()
            scglue.data.lsi(atac, n_components=100, n_iter=15)
            # create graph
            split = rna.var["interval"].str.split(r"[:-]")
            rna = rna[:, (split.map(lambda x: len(x)) == 3)].copy()
            split = rna.var["interval"].str.split(r"[:-]")
            rna.var["chrom"] = split.map(lambda x: x[0])
            rna.var["chromStart"] = split.map(lambda x: x[1]).astype(int)
            rna.var["chromEnd"] = split.map(lambda x: x[2]).astype(int)
            split = atac.var_names.str.split(r"[:-]")
            atac.var["chrom"] = split.map(lambda x: x[0])
            atac.var["chromStart"] = split.map(lambda x: x[1]).astype(int)
            atac.var["chromEnd"] = split.map(lambda x: x[2]).astype(int)
            guidance_graph = scglue.genomics.rna_anchored_guidance_graph(rna, atac)
            scglue.graph.check_graph(guidance_graph, [rna, atac])
            atac.write(os.path.join(self.save_dir, "adata_atac"))

        elif "adt" in mdata.mod.keys():
            adt = mdata.mod["adt"].copy()
            adt.layers["counts"] = adt.X.copy()
            mu.prot.pp.clr(adt, axis=1)
            sc.tl.pca(adt, n_comps=min(adt.shape[1] - 1, 100), svd_solver="auto")
            # create graph
            gene_prot_names = mdata.uns["gene_prots"]
            prot_var_names = []
            for prot in adt.var_names:
                if prot in gene_prot_names.keys():
                    if mdata.uns["data_name"] == "bmcite":
                        prot_name = prot.split("_")[1]
                    else:
                        prot_name = prot
                    if type(gene_prot_names[prot_name]) == str:
                        prot_var_names.append(gene_prot_names[prot_name])
                    else:
                        prot_var_names.append(gene_prot_names[prot_name][0])
                else:
                    prot_var_names.append(prot)
            adt.var_names = prot_var_names

            p = np.array(adt.var_names)
            r = np.array(rna.var_names)
            # mask entries are set to 1 where protein name is the same as gene name
            mask = np.repeat(p.reshape(-1, 1), r.shape[0], axis=1) == r
            mask = np.array(mask)
            rna_vars = [v + "_rna" for v in rna.var_names]
            prot_vars = [v + "_prot" for v in adt.var_names]
            rna.var_names = rna_vars
            adt.var_names = prot_vars
            adj = pd.DataFrame(mask, index=prot_vars, columns=rna_vars)
            diag_edges = adj[adj > 0].stack().index.tolist()
            diag_edges = [(n1, n2, {"weight": 1.0, "sign": 1}) for n1, n2 in diag_edges]
            self_loop_rna = [(g, g, {"weight": 1.0, "sign": 1}) for g in rna_vars]
            self_loop_prot = [(g, g, {"weight": 1.0, "sign": 1}) for g in prot_vars]
            guidance_graph = nx.Graph()
            guidance_graph.add_nodes_from(rna_vars)
            guidance_graph.add_nodes_from(prot_vars)
            guidance_graph.add_edges_from(diag_edges)
            guidance_graph.add_edges_from(self_loop_prot)
            guidance_graph.add_edges_from(self_loop_rna)
            scglue.graph.check_graph(guidance_graph, [rna, adt])
            adt.write(os.path.join(self.save_dir, "adata_adt"))
        elif "fish" in mdata.mod.keys():
            fish = mdata.mod["fish"].copy()
            fish.layers["counts"] = fish.X.copy()

            # create graph
            p = np.array(fish.var_names)
            r = np.array(rna.var_names)
            # mask entries are set to 1 where fish gene name is the same as scRNA gene name
            mask = np.repeat(p.reshape(-1, 1), r.shape[0], axis=1) == r
            mask = np.array(mask)
            rna_vars = rna.var_names
            fish_vars = [v + "_fish" for v in fish.var_names]
            fish.var_names = fish_vars
            adj = pd.DataFrame(mask, index=fish_vars, columns=rna_vars)
            diag_edges = adj[adj > 0].stack().index.tolist()
            diag_edges = [(n1, n2, {"weight": 1.0, "sign": 1}) for n1, n2 in diag_edges]
            self_loop_rna = [(g, g, {"weight": 1.0, "sign": 1}) for g in rna_vars]
            self_loop_fish = [(g, g, {"weight": 1.0, "sign": 1}) for g in fish_vars]
            guidance_graph = nx.Graph()
            guidance_graph.add_nodes_from(rna_vars)
            guidance_graph.add_nodes_from(fish_vars)
            guidance_graph.add_edges_from(diag_edges)
            guidance_graph.add_edges_from(self_loop_fish)
            guidance_graph.add_edges_from(self_loop_rna)
            scglue.graph.check_graph(guidance_graph, [rna, fish])
            fish.write(os.path.join(self.save_dir, "adata_fish"))

        rna.write(os.path.join(self.save_dir, "adata_rna"))

        nx.write_graphml(guidance_graph, os.path.join(self.save_dir, "guidance.graphml.gz"))

    def configure_ds(self, adata, mod):
        if mod == "rna":
            scglue.models.configure_dataset(adata, "NB", use_highly_variable=True, use_layer="counts", use_rep="X_pca",
                                            use_batch="batch")
        elif mod == "atac":
            scglue.models.configure_dataset(adata, "NB", use_highly_variable=True, use_rep="X_lsi", use_batch="batch")
        elif mod == "adt":
            scglue.models.configure_dataset(adata, "Normal", use_highly_variable=False, use_rep="X_pca",
                                            use_batch="batch")
        elif mod == "fish":
            scglue.models.configure_dataset(adata, "NB", use_highly_variable=False, use_layer="counts",
                                            use_batch="batch")
        else:
            raise ValueError("Unrecognized modality: {}".format(mod))

    def run(self):
        guidance_graph = nx.read_graphml(os.path.join(self.save_dir, "guidance.graphml.gz"))
        adata_paths = [name for name in os.listdir(self.save_dir) if name.startswith("adata")]
        adata_dic = {name.split("_")[-1]: anndata.read(os.path.join(self.save_dir, name)) for name in adata_paths}
        for mod, adata in adata_dic.items():
            self.configure_ds(adata, mod)
        if self.no_mix:
            lam_align = 0.
        else:
            lam_align = 0.05
        for seed in self.seeds:
            try:
                glue = scglue.models.fit_SCGLUE(adata_dic, guidance_graph, init_kws={"random_seed": seed,
                                                                                     "latent_dim": self.latent_dim},
                                                fit_kws={"directory": self.save_dir, "max_epochs": self.tr_length},
                                                compile_kws={"lam_align": lam_align})
                try:
                    dx = scglue.models.integration_consistency(glue, adata_dic, guidance_graph)
                    ax = sns.lineplot(x="n_meta", y="consistency", data=dx).axhline(y=0.05, c="darkred", ls="--")
                    fig = ax.get_figure()
                    fig.savefig(os.path.join(self.save_dir, "consistency_plot_{}.pdf".format(seed)))
                    plt.close('all')
                except:
                    print("Failed to compute integration consistency for seed {} "
                          "with error: {}".format(seed, traceback.format_exc()))
                latents = []
                obs_names = []
                for mod, adata in adata_dic.items():
                    latents.append(glue.encode_data(mod, adata))
                    obs_names.append(adata.obs.index)
                latents_df = pd.DataFrame(np.concatenate(latents, axis=0), index=np.concatenate(obs_names, axis=0))
                latents_df.to_csv(os.path.join(self.save_dir, "latents_{}.csv".format(seed)))
                if len(self.imputation_genes) > 0:
                    imputed = glue.decode_data(source_key="fish", target_key="rna", adata=adata_dic["fish"],
                                               graph=guidance_graph)
                    imputation_idxes = [list(adata_dic["rna"].var_names).index(gene) for gene in self.imputation_genes]
                    imputed_df = pd.DataFrame(imputed[:, imputation_idxes], index=adata_dic["fish"].obs_names,
                                              columns=self.imputation_genes)
                    imputed_df.to_csv(os.path.join(self.save_dir, "imputations_{}.csv".format(seed)))
            except:
                print("Run of scGLUE on seed {} failed with error message: {}".format(seed, traceback.format_exc()))
        if self.rm_data:
            for name in adata_paths:
                os.remove(os.path.join(self.save_dir, name))
            os.remove(os.path.join(self.save_dir, "guidance.graphml.gz"))
