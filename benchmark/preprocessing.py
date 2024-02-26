import scipy.sparse
from scipy.sparse import csr_matrix, dia_matrix, issparse, block_diag
import numpy as np
import pandas as pd
import scanpy as sc
import muon as mu
import anndata
from anndata import AnnData, concat
from mudata import MuData
from scipy.spatial.distance import cdist

THREE_OMICS_ORDER = {"rna": ['CD4 T', 'CD8 T', 'gd T', 'B', 'PB', 'Granulocyte',
                             'NK', 'Platelet', 'RBC', 'CD14 Monocyte', 'CD16 Monocyte', 'DC', 'pDC'],
                     "adt": ['Treg CD4 T cells', 'Cytotoxic CD4 T cells', 'Activated CD4 T cells', 'Memory CD4 T cells',
                             'Naive CD4 T cells', 'Cytotoxic CD8 T cells', 'Memory CD8 T cells', 'Naive CD8 T cells',
                             'Activated CD8 T cells', 'DN T cells', 'Non-activated gd T cells',
                             'MAIT cells', 'B cells', 'CD16negCD56dim NK', 'CD16negCD56hi NK', 'CD16posCD56dim NK',
                             'cDC2', 'cMono', 'intMono', 'ncMono', 'Basophil', 'pDC'],
                     "atac": ['Naive CD4 T1', 'Naive CD4 T2', 'Memory CD4 T', 'Central memory CD8 T',
                              'Effector memory CD8 T', 'Naive CD8 T1', 'Naive CD8 T3', 'Naive Treg', 'Treg',
                              'Gamma delta T', 'Mature NK1', 'Memory B', 'Naive B', 'Plasma cell', 'Basophil',
                              'Monocyte 1', 'Monocyte 2', 'cDC', 'pDC']}

MODALITY_TAGS = {"rna": "GEX", "atac": "ATAC", "adt": "ADT", "fish": "FISH", "morph": "IMG"}


def subset_data(mdata, spl_pct, seed):
    copy_uns = mdata.uns.copy()
    if spl_pct < 1.0:
        print("subsampling {}% of cells".format(spl_pct * 100))
        np.random.seed(seed)
        mdata = mu.pp.sample_obs(mdata, frac=spl_pct).copy()  # , groupby="celltype")
    mdata.uns = copy_uns
    return mdata


def qc_filter(adata: AnnData,
              min_cells_ratio: float = 0.01, max_cells_ratio: float = 99.,
              min_features: int = 200, max_features: int = 20000):
    """

    :param adata: Unimodal data
    :param min_cells_ratio:
    :param max_cells_ratio:
    :param min_features:
    :param max_features:
    :return:
    """
    n_samples = adata.shape[0]

    # filter features
    sc.pp.filter_genes(adata, min_cells=int(min_cells_ratio * n_samples))
    sc.pp.filter_genes(adata, max_cells=int(max_cells_ratio * n_samples))

    # filter cells
    sc.pp.filter_cells(adata, min_genes=min_features)
    sc.pp.filter_cells(adata, max_genes=max_features)


def apply_qc_filter(umodal_adata, mod, qc_cfg):
    if mod == "adt":
        print("No addtional QC needed for now on adt data")
    else:
        print("Applying QC filters to data from the {} modality".format(mod))
        init_shape = umodal_adata.shape
        qc_filter(umodal_adata, **qc_cfg)
        end_shape = umodal_adata.shape
        print("Started with {} cells and {} features and ended with {} "
              "cells and {} features".format(init_shape[0], init_shape[1],
                                             end_shape[0], end_shape[1]))


def normalize_rna(adata, target_sum=None):
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)


def tf_idf_atac(adata, idf=None, log_tf=True, log_idf=True, log_tfidf=False,
                scale_factor=10000.0):
    """
    Transform peak counts with TF-IDF (Term Frequency - Inverse Document Frequency).
    TF: peak counts are normalised by total number of counts per cell
    DF: total number of counts for each peak
    IDF: number of cells divided by DF
    By default, log(TF) * log(IDF) is returned.
    Parameters
    ----------
    data
            AnnData object with peak counts or multimodal MuData object with 'atac'
            modality.
    log_idf
            Log-transform IDF term (True by default)
    log_tf
            Log-transform TF term (True by default)
    log_tfidf
            Log-transform TF*IDF term (False by default)
            Can only be used when log_tf and log_idf are False
    scale_factor
            Scale factor to multiply the TF-IDF matrix by (1e4 by default)
    """
    if log_tfidf and (log_tf or log_idf):
        raise AttributeError(
            "When returning log(TF*IDF), \
            applying neither log(TF) nor log(IDF) is possible."
        )

    if issparse(adata.X):
        n_peaks = np.asarray(adata.X.sum(axis=1)).reshape(-1)
        n_peaks = dia_matrix((1.0 / n_peaks, 0), shape=(n_peaks.size, n_peaks.size))
        # This prevents making TF dense
        tf = np.dot(n_peaks, adata.X)
    else:
        n_peaks = np.asarray(adata.X.sum(axis=1)).reshape(-1, 1)
        tf = adata.X / n_peaks

    if scale_factor is not None and scale_factor != 0 and scale_factor != 1:
        tf = tf * scale_factor
    if log_tf:
        tf = np.log1p(tf)

    if idf is None:
        idf = np.asarray(adata.shape[0] / adata.X.sum(axis=0)).reshape(-1)
        if log_idf:
            idf = np.log1p(idf)
        if issparse(tf):
            idf = dia_matrix((idf, 0), shape=(idf.size, idf.size))
        else:
            idf = csr_matrix(np.diag(idf))

    if not issparse(tf):
        tf = csr_matrix(tf)

    tf_idf = np.dot(tf, idf)
    if log_tfidf:
        tf_idf = np.log1p(tf_idf)

    adata.X = np.nan_to_num(tf_idf, 0)


def normalize_fish(array, col_factor=None):
    array = array / array.sum(axis=1, keepdims=True)  # Corrected for total molecules per gene
    if col_factor is None:
        col_factor = array.mean(axis=0, keepdims=True)
    array = array / col_factor  # * array.shape[1]
    return array


def normalize_data(umodal_adata, mod, **kwargs):
    if mod == "rna":
        normalize_rna(umodal_adata, **kwargs)
    elif mod == "fish":
        umodal_adata.X = normalize_fish(umodal_adata.X)
        umodal_adata.layers["norm_ints"] = np.ceil(umodal_adata.X)
    elif mod == "atac":
        tf_idf_atac(umodal_adata, **kwargs)
    elif mod == "adt":
        mu.prot.pp.clr(umodal_adata, axis=1)


def select_features(umodal_data, cfg):
    if cfg["raw"]["n_top_features"] is not None:
        umodal_data.var["raw_hvg"] = sc.pp.highly_variable_genes(umodal_data, layer="counts", inplace=False,
                                                                 n_top_genes=cfg["raw"]["n_top_features"],
                                                                 flavor="seurat_v3", span=1.,
                                                                 batch_key="batch")["highly_variable"].values
    else:
        umodal_data.var["raw_hvg"] = np.array(umodal_data.shape[1] * [False])
    norm_hvg = sc.pp.highly_variable_genes(umodal_data, inplace=False,
                                           n_top_genes=cfg["normalized"]["n_top_features"], span=1., batch_key="batch")
    umodal_data.var["norm_hvg"] = norm_hvg["highly_variable"].values
    if "keep_genes" in cfg.keys():
        umodal_data.var["raw_hvg"][cfg["keep_genes"]] = True


def transfer_features_validation(umodal_tr_data, umodal_val_data):
    umodal_val_data.var["raw_hvg"] = umodal_tr_data.var["raw_hvg"]
    umodal_val_data.var["norm_hvg"] = umodal_tr_data.var["norm_hvg"]


def umodal_process(umodal_adata, mod, umodal_data_cfg):
    # Normalize data
    if not issparse(umodal_adata.X):
        umodal_adata.X = umodal_adata.X.astype("float32")
    umodal_adata.layers["counts"] = umodal_adata.X.copy()

    normalize_data(umodal_adata, mod=mod, **umodal_data_cfg["normalization"])

    # Do variable selection on raw and normalized
    if mod in ["adt", "fish", "morph"]:
        umodal_adata.var["raw_hvg"] = np.array(umodal_adata.shape[1] * [True])
        umodal_adata.var["norm_hvg"] = np.array(umodal_adata.shape[1] * [True])
    else:
        select_features(umodal_adata, umodal_data_cfg["feature_selection"])

    # Store pca for both modalities
    if umodal_data_cfg["n_pca"] is not None:
        sc.tl.pca(umodal_adata, n_comps=umodal_data_cfg["n_pca"], zero_center=None)


def update_final_size(umodal_adata, data_cfg, mod):
    data_cfg[mod]["full_dim"] = umodal_adata.shape[1]
    data_cfg[mod]["hv_dim"] = int((np.logical_or(umodal_adata.var["raw_hvg"], umodal_adata.var["norm_hvg"])).sum())
    if "gene_activities_norm" in umodal_adata.obsm.keys():
        data_cfg[mod]["n_gene_act"] = umodal_adata.obsm["gene_activities_norm"].shape[1]
    return data_cfg


def mmodal_to_umodal_mtx(n_1, n_2, mtx_1=None, mtx_2=None):
    if mtx_1 is None and mtx_2 is None:
        raise ValueError("both matrices can't be None")
    if mtx_1 is None or mtx_2 is None:
        if mtx_2 is None:
            sparse = issparse(mtx_1)
            if sparse:
                mtx_2 = csr_matrix(np.zeros((n_2, mtx_1.shape[1])))
            else:
                mtx_2 = np.zeros((n_2, mtx_1.shape[1]))
        elif mtx_1 is None:
            sparse = issparse(mtx_2)
            if sparse:
                mtx_1 = csr_matrix(np.zeros((n_1, mtx_2.shape[1])))
            else:
                mtx_1 = np.zeros((n_1, mtx_2.shape[1]))
    else:
        d1 = mtx_1.shape[1]
        d2 = mtx_2.shape[1]
        sparse = False
        if issparse(mtx_1):
            sparse = True
            mtx_1 = scipy.sparse.hstack((mtx_1, csr_matrix(np.zeros((n_1, d2)))))
        else:
            mtx_1 = np.concatenate((mtx_1, np.zeros((n_1, d2))), axis=1)

        if issparse(mtx_2):
            sparse = True
            mtx_2 = scipy.sparse.hstack((csr_matrix(np.zeros((n_2, d1))), mtx_2))
        else:
            mtx_2 = np.concatenate((np.zeros((n_2, d1)), mtx_2), axis=1)
    if sparse:
        return scipy.sparse.vstack((mtx_1, mtx_2)).tocsr()
    else:
        return np.concatenate((mtx_1, mtx_2), axis=0)


def mdata_to_adata(mdata):
    modalities = list(mdata.mod.keys())

    X = np.zeros((sum([mdata[mod].shape[0] for mod in modalities]), 1))

    adata_obs = pd.concat((mdata[mod].obs for mod in modalities), axis=0, join="outer")
    adata_obs["modality"] = np.concatenate(([[MODALITY_TAGS[mod]] * mdata[mod].shape[0] for mod in modalities]))

    adata_uns = {"batch_indexes": {**mdata.mod[modalities[0]].uns["batch_indexes"],
                                   **mdata.mod[modalities[1]].uns["batch_indexes"]}}
    for mod in modalities:
        for k, v in mdata[mod].uns.items():
            if k != "batch_indexes":
                adata_uns["{}|{}".format(k, MODALITY_TAGS[mod])] = v

    adata = AnnData(X, obs=adata_obs, uns=adata_uns)

    return adata


def save_batch_info_umodal(adata, suffix, data_cfg):
    if "batch" in adata.obs.keys():
        adata.obs["batch"] = [str(b) + suffix for b in adata.obs["batch"].values]
        batch_values = np.unique(adata.obs["batch"].values)
        adata.uns["batch_indexes"] = {b: k for k, b in enumerate(batch_values)}
    else:
        adata.obs["batch"] = np.array(adata.shape[0] * ["uniquebatch" + suffix])
        adata.uns["batch_indexes"] = {"uniquebatch" + suffix: 0}
    data_cfg["batches"] = list(adata.uns["batch_indexes"].keys())


def get_gene_prots(gene_prot_names, prot_prefix="", all_genes=True):
    gene_names, prot_names = [], []

    for k, val in gene_prot_names.items():
        if type(val) == str:
            gene_names.append(val)
            prot_names.append(prot_prefix + k)
        else:
            if all_genes:
                gene_names += val
                prot_names += len(val) * [prot_prefix + k]
            else:
                gene_names.append(val[0])
                prot_names.append(prot_prefix + k)
    return gene_names, prot_names


def rename_gene_prots(gene_prot_names, prots, data_name):
    if data_name == "bmcite":
        rm_prots = ["CD45RO"]
    else:
        rm_prots = ["CD45RO", "CD45RA"]
    converted_prots = [gene_prot_names[prot_name.split("_")[-1]]
                       if (prot_name.split("_")[-1] in gene_prot_names.keys() and
                           not prot_name.split("_")[-1] in rm_prots)
                       else prot_name
                       for prot_name in prots]
    converted_prots = [name if type(name) == str else name[0] for name in converted_prots]
    return converted_prots


def get_common_features_mdata(mdata, modalities, atac_key=None, gene_prot_names=None, prot_prefix=""):
    modalities = modalities.split("+")
    common_features = {mod: [] for mod in modalities}
    if set(modalities) == {"rna", "atac"}:
        genes = mdata.mod["rna"].var_names
        atac_genes = list(mdata.mod["atac"].uns['gene_activities_var_names'])
        for gene in genes:
            if gene in atac_genes:
                common_features["rna"].append(gene)
        common_features["atac"] = common_features["rna"]
        ga_genes_idx = np.array([atac_genes.index(gene)
                                 for gene in common_features["atac"]])
        cross_mdata = mu.MuData({"rna": anndata.AnnData(mdata["rna"][:, common_features["rna"]].layers["counts"]),
                                 "atac": anndata.AnnData(mdata["atac"].obsm[atac_key][:, ga_genes_idx])})
    elif set(modalities) == {"rna", "adt"}:
        common_features["rna"], common_features["adt"] = get_gene_prots(gene_prot_names=gene_prot_names,
                                                                        prot_prefix=prot_prefix, all_genes=True)
        cross_mdata = mu.MuData({"rna": anndata.AnnData(mdata["rna"][:, common_features["rna"]].layers["counts"]),
                                 "adt": anndata.AnnData(mdata["adt"][:, common_features["adt"]].layers["counts"])})
    elif set(modalities) == {"rna", "fish"}:
        genes = mdata.mod["rna"].var_names
        fish_genes = mdata.mod["fish"].var_names
        for gene in fish_genes:
            if gene in genes:
                common_features["rna"].append(gene)
        common_features["fish"] = common_features["rna"]

        cross_mdata = mu.MuData({"rna": anndata.AnnData(mdata["rna"][:, common_features["rna"]].layers["counts"]),
                                 "fish": anndata.AnnData(mdata["fish"][:, common_features["fish"]].layers["counts"])})
    elif set(modalities) == {"atac", "adt"}:
        atac_genes = list(mdata.mod["atac"].uns['gene_activities_var_names'])
        for k, val in gene_prot_names.items():
            if type(val) == str:
                common_features["atac"].append(val)
                common_features["adt"].append(prot_prefix + k)
            else:
                common_features["atac"] += val
                common_features["adt"] += len(val) * [prot_prefix + k]
        ga_genes_idx = np.array([atac_genes.index(gene)
                                 for gene in common_features["atac"]])
        cross_mdata = mu.MuData({"atac": anndata.AnnData(mdata["atac"].obsm[atac_key][:, ga_genes_idx]),
                                 "adt": anndata.AnnData(mdata["adt"][:, common_features["adt"]].layers["counts"])})
    elif set(modalities) == {"rna", "morph"}:
        morph_genes = list(mdata["morph"].uns["gene_names"])
        morph_genes_idx = np.array([morph_genes.index(gene) for gene in mdata["rna"].var_names])
        cross_mdata = mu.MuData({"rna": anndata.AnnData(mdata["rna"].layers["counts"]),
                                 "morph": anndata.AnnData(mdata["morph"].obsm["rna"][:, morph_genes_idx])})
        common_features["rna"] = mdata["rna"].var_names
        common_features["morph"] = mdata["rna"].var_names
    else:
        raise ValueError("Unrecognized modality pair: {}".format(modalities))
    for mod in modalities:
        cross_mdata[mod].obs_names = mdata[mod].obs_names
        cross_mdata[mod].var_names = common_features[mod]
    cross_mdata.update()

    return cross_mdata


def prepro_comm_features_mdata(mdata, cross_prepro_cfg):
    feature_mask = None
    for mod, cfg in cross_prepro_cfg.items():
        if cfg["norm_method"] == "log_norm":
            sc.pp.normalize_total(mdata[mod], **cfg["norm_params"])
            sc.pp.log1p(mdata[mod])
        elif cfg["norm_method"] == "clr":
            mu.prot.pp.clr(mdata[mod], **cfg["norm_params"])
        elif cfg["norm_method"] == "fish_norm":
            mdata[mod].X = normalize_fish(mdata[mod].X)
        else:
            if cfg["norm_method"] is not None:
                raise ValueError("Unrecognized normalization method: {}".format(cfg["norm_method"]))

        if cfg["hvg"] != {}:
            feature_mask = sc.pp.highly_variable_genes(mdata[mod],
                                                       **cfg["hvg"], inplace=False)["highly_variable"].values
    if feature_mask is not None:
        for mod in mdata.mod.keys():
            mdata.mod[mod] = mdata[mod][:, feature_mask].copy()
    for mod, cfg in cross_prepro_cfg.items():
        if cfg["scale"]:
            sc.pp.scale(mdata[mod], **cfg["scale_params"])


def cross_rel_wrapper(mdata, modality, data_cfg):
    mdata.uns["cross_keys"] = []
    modality_pairs = ["{}+{}".format(mod1, mod2)
                      for k, mod1 in enumerate(modality.split("+"))
                      for i, mod2 in enumerate(modality.split("+")) if i != k]
    found_none = True
    for modality_pair in modality_pairs:
        if modality_pair in data_cfg["cross_rel"]:
            found_none = False
            cross_mdata = get_common_features_mdata(mdata, modality_pair,
                                                    **data_cfg["cross_rel"][modality_pair]["common_features"])
            prepro_comm_features_mdata(cross_mdata, data_cfg["cross_rel"][modality_pair]["prepro"])
            for mod in modality_pair.split("+"):
                if issparse(cross_mdata[mod].X):
                    cross_mdata[mod].X = np.array(cross_mdata[mod].X.todense())
                else:
                    cross_mdata[mod].X = cross_mdata[mod].X
            mod_1, mod_2 = modality_pair.split("+")
            mdata.uns["cross_{}".format(modality_pair)] = cdist(cross_mdata[mod_1].X,
                                                                cross_mdata[mod_2].X,
                                                                metric="correlation")
            mdata.uns["cross_keys"].append("cross_{}".format(modality_pair))
    assert not found_none, "No cross-relational data found in data_cfg"


def preprocess_umodal_mdata(mdata, modality, data_cfg, spl_pct):
    # Keep only certain celltypes and subsample data
    mdata = subset_data(mdata, spl_pct, **data_cfg["subsample"])

    data = mdata[modality]

    # Do basic QC filters and handle batch information
    apply_qc_filter(data, modality, data_cfg[modality]["QC"])
    save_batch_info_umodal(adata=data, suffix="", data_cfg=data_cfg[modality])

    heldout_adata = None


    umodal_process(data, modality, data_cfg[modality])
    data_cfg = update_final_size(data, data_cfg, modality)

    return data, data_cfg, heldout_adata


def preprocess_mdata(mdata, modality, data_cfg, spl_pct, baseline=False):
    if "+" in modality:
        return preprocess_mmodal_mdata(mdata=mdata, modality=modality,
                                       data_cfg=data_cfg, spl_pct=spl_pct,
                                       baseline=baseline)
    else:
        return preprocess_umodal_mdata(mdata=mdata, modality=modality, data_cfg=data_cfg, spl_pct=spl_pct)


def preprocess_mmodal_mdata(mdata, modality, data_cfg, spl_pct, baseline=False):
    # Keep only certain celltypes and subsample data
    mdata = subset_data(mdata, spl_pct=spl_pct, seed=data_cfg["subsample_seed"])

    # If we are using only a subset of modalities, then we need to make a new MuData object
    if len(mdata.mod.keys()) > 2 and len(modality.split("+")) == 2:
        mdata = mu.MuData({mod: mdata[mod].copy() for mod in modality.split("+")})

    # Do basic QC filters and handle batch information
    for mod in modality.split("+"):
        apply_qc_filter(mdata[mod], mod, data_cfg[mod]["QC"])
        save_batch_info_umodal(adata=mdata[mod], data_cfg=data_cfg[mod],
                               suffix="|{}".format(MODALITY_TAGS[mod]))
    if modality == "rna+adt":
        prot_genes, _ = get_gene_prots(**data_cfg["cross_rel"]["rna+adt"]["common_features"], all_genes=False)
        mdata.mod["rna"] = mdata["rna"][mdata["rna"][:, prot_genes].X.sum(axis=1) > 0.].copy()
    if data_cfg["paired"]:
        mu.pp.intersect_obs(mdata)
    mdata.update()

    for mod in modality.split("+"):
        mdata[mod].obs.index = mdata[mod].obs.index + "|" + mdata[mod].shape[0] * [MODALITY_TAGS[mod]]
    mdata.update()

    heldout_adata = None
    if "fish" in modality:
        cm_genes = mdata.mod["fish"].var_names
        data_cfg["rna"]["feature_selection"]["keep_genes"] = cm_genes
        if "heldout_fish_genes" in data_cfg.keys():
            cm_genes = [g for g in mdata.mod["fish"].var_names if g not in data_cfg["heldout_fish_genes"]]
        mdata.mod["rna"] = mdata["rna"][mdata["rna"][:, cm_genes].X.sum(1) > 0].copy()
        mdata.update()
        if "heldout_fish_genes" in data_cfg.keys():
            mdata, heldout_adata, data_cfg = handle_imput_genes(mdata=mdata, data_cfg=data_cfg)

    if baseline:
        if modality == "rna+adt":
            mdata.uns["gene_prots"] = data_cfg["cross_rel"]["rna+adt"]["common_features"]["gene_prot_names"]
        return mdata, data_cfg, heldout_adata
    else:
        for mod in modality.split("+"):
            umodal_process(mdata[mod], mod, data_cfg[mod])
            mdata.update()
            data_cfg = update_final_size(mdata[mod], data_cfg, mod)
        cross_rel_wrapper(mdata, modality, data_cfg)

    return mdata, data_cfg, heldout_adata


def handle_imput_genes(mdata, data_cfg):
    # write expression of those heldout genes
    heldout_adata = mdata.copy()
    for mod in ["rna", "fish"]:
        heldout_adata.mod[mod] = heldout_adata.mod[mod][:, data_cfg["heldout_fish_genes"]].copy()
        # heldout_adata.mod[mod].obs.index = heldout_adata.mod[mod].obs.index + "|" + \
        #                                   heldout_adata.mod[mod].shape[0] * [MODALITY_TAGS[mod]]
    heldout_adata = concat([heldout_adata.mod["rna"], heldout_adata.mod["fish"]], join="outer")
    fish_mask = np.array([name.endswith(MODALITY_TAGS["fish"]) for name in heldout_adata.obs_names])
    heldout_adata.obs["modality"] = MODALITY_TAGS["rna"]
    heldout_adata.obs["modality"][fish_mask] = MODALITY_TAGS["fish"]
    heldout_adata.uns["batch_indexes"] = []
    heldout_adata.obs["library_size"] = np.sum(heldout_adata.X, axis=1)
    training_genes = [gene for gene in mdata.mod["fish"].var_names if
                      gene not in data_cfg["heldout_fish_genes"]]
    mdata.mod["fish"] = mdata.mod["fish"][:, training_genes].copy()
    mdata.update()
    return mdata, heldout_adata, data_cfg
