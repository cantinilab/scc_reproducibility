import os
import pickle
import numpy as np
import pandas as pd
import muon as mu
import anndata
from anndata import AnnData
from scipy.stats import spearmanr
from sklearn.neighbors import KNeighborsRegressor

from benchmark.preprocessing import mdata_to_adata
from benchmark.metrics import (compute_knn_metrics)
from benchmark.exp_utils import get_test_data


def eval_wrap(root_path, cfg_name, modality, k_array, resolutions, k_cnct, data_name,
              n_subsamples=None, baseline=False, metrics=None):
    folder_path = os.path.join(root_path, cfg_name)
    file_latents = [file for file in os.listdir(folder_path) if file.startswith("latent")]
    if "test_data" in os.listdir(root_path):
        print("Loading existing test anndata")
        test_adata = anndata.read(os.path.join(root_path, "test_data"))
        seeds = [name.split("_")[1] for name in test_adata.obsm_keys()
                 if "latent" in name and cfg_name in name]
        # keep only the latents for the current config
        for name in test_adata.obsm_keys():
            if "latent" in name:
                if cfg_name in name:
                    test_adata.obsm["latents_{}".format(name.split("_")[1])] = test_adata.obsm[name].copy()
                    del test_adata.obsm[name]
                else:
                    del test_adata.obsm[name]

    else:
        if len(file_latents) == 0:
            return None
        if baseline:
            test_adata = mdata_to_adata(mu.read(os.path.join(root_path,
                                                             "mdata_baseline.h5mu")))
        else:
            test_adata = get_test_data(root_path=root_path, modality=modality,
                                       return_data_path=False)
        for file_name in file_latents:
            latent_name = file_name.split(".")[0]
            test_adata.obsm[latent_name] = pd.read_csv(os.path.join(folder_path,
                                                                    file_name),
                                                       index_col=0).loc[test_adata.obs.index]
        seeds = [(name.split("_")[-1]).split(".")[0] for name in file_latents]
        if "cnct_adata" in os.listdir(folder_path):
            cnct_adata = anndata.read(os.path.join(folder_path, "cnct_adata"))
            for seed in seeds:
                cnct_key = "{}_connectivities".format(seed)
                test_adata.obsp[cnct_key] = cnct_adata[test_adata.obs.index].obsp[cnct_key]
    metrics_wrapper(adata=test_adata, seeds=seeds, k_array=k_array,
                    resolutions=resolutions, multimodal=("+" in modality),
                    save_path=folder_path, k_cnct=k_cnct, data_name=data_name,
                    n_subsamples=n_subsamples, metrics=metrics)


def metrics_wrapper(adata: AnnData, seeds, k_array, resolutions, multimodal: bool, save_path: str,
                    knn_metric: str = "minkowski", k_cnct=[], data_name="", n_subsamples=None, metrics=None):
    """
    Compute metrics to assess the quality of the latent space inferred from each model
    :param adata: Data structure which also contains the latent embeddings for each model in obsm
    :param seeds: list of each run's starting seed
    :param k_array: Array of k values to compute each knn metric for different numbers of neighbors considered
    :param resolutions: Array of resolutions with which to compute louvain clustering
    :param multimodal: Whether the experiment is run on multimodal data
    :param save_path: either None if nothing is to be saved or the path to the directory where the config and results
    will be saved
    :param knn_metric: which distance to use when computing the knn graph
    :param k_cnct: list of number of neighbors used to compute the graph connectivity
    :param data_name: name of the data set being processed
    :param n_subsamples: if not None, number of subsamples to use for the evaluation
    :param metrics: list of metrics to compute
    :return: knn evaluation metrics allowing to compare the different training configs
    """
    model_results = {}
    model_results_cnct = {}
    labels = None
    batch_labels = None
    if "celltype" in adata.obs.keys():
        labels = np.array(adata.obs["celltype"].values)
    if len(adata.uns["batch_indexes"]) > 1 + int(multimodal):
        batch_labels = np.array(adata.obs["batch"].values)

    if multimodal:
        modality_labels = np.array(adata.obs["modality"].values)
        modalities = np.unique(modality_labels)
        metrics_to_compute = ["purity", "foscttm_full",
                              "celltype_transfer"]
        pair_idxs = get_pairs(adata, splitter="|", modalities=modalities)
        if len(pair_idxs[modality_labels[0]]) == 0:
            metrics_to_compute = ["purity", "celltype_transfer"]
        if n_subsamples is not None:
            subsample_idxs = np.random.choice(len(pair_idxs[modalities[0]]), n_subsamples, replace=False)
            subsample_idxs = np.concatenate((np.where(modality_labels == modalities[0])[0][pair_idxs[modalities[0]][subsample_idxs]],
                                             np.where(modality_labels == modalities[1])[0][pair_idxs[modalities[1]][subsample_idxs]]))
            adata = adata[subsample_idxs, :].copy()
            labels = labels[subsample_idxs]
            modality_labels = modality_labels[subsample_idxs]
            if batch_labels is not None:
                batch_labels = batch_labels[subsample_idxs]
            pair_idxs = get_pairs(adata, splitter="|", modalities=modalities)

    else:
        modality_labels = None
        metrics_to_compute = ["purity"]
        pair_idxs = None
    if k_cnct:
        metrics_to_compute.append("graph_connectivity")
    if metrics is not None:
        metrics_to_compute = [n for n in metrics_to_compute if n in metrics]

    for seed in seeds:
        latents_key = "latents_{}".format(seed)
        if latents_key not in adata.obsm.keys():
            continue
        latents = adata.obsm[latents_key]
        model_results[seed] = compute_knn_metrics(latents, k_array=k_array, metrics_to_compute=metrics_to_compute,
                                                  labels=labels, connectivities=None, batch_labels=batch_labels,
                                                  modality_labels=modality_labels, pair_idxs=pair_idxs,
                                                  knn_metric=knn_metric, k_cnct=k_cnct)

        cnct_key = "{}_connectivities".format(seed)
        if cnct_key in adata.obsp.keys():
            model_results_cnct[seed] = compute_knn_metrics(latents, k_array=k_array,
                                                           metrics_to_compute=metrics_to_compute,
                                                           labels=labels, connectivities=adata.obsp[cnct_key],
                                                           batch_labels=batch_labels, k_cnct=k_cnct,
                                                           modality_labels=modality_labels,
                                                           pair_idxs=pair_idxs, knn_metric=knn_metric)

    if save_path is not None:
        with open(os.path.join(save_path, "metric_results.pickle"), 'wb') as handle:
            pickle.dump(model_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
        if len(model_results_cnct) > 0:
            with open(os.path.join(save_path, "metric_results_graph.pickle"), 'wb') as handle:
                pickle.dump(model_results_cnct, handle, protocol=pickle.HIGHEST_PROTOCOL)


def get_pairs(adata, modalities, splitter="|"):
    # For each modality, the idxes correpond to the indx of the cell among all cells from the same modality,
    # not among all cells in the adata.
    obs_names_1 = np.array([name.split(splitter)[0] for name in adata.obs_names
                            if name.split(splitter)[1] == modalities[0]])
    obs_names_2 = np.array([name.split(splitter)[0] for name in adata.obs_names
                            if name.split(splitter)[1] == modalities[1]])
    _, pair_x_idxs, pair_y_idxs = np.intersect1d(obs_names_1, obs_names_2, assume_unique=True, return_indices=True)
    pair_idxs = {modalities[0]: pair_x_idxs, modalities[1]: pair_y_idxs}
    return pair_idxs


def aggreg_seed_results(results, metrics=["accuracy", "mixing", "purity"]):
    median_res = {}
    std_res = {}
    for model_name, model_seeds in results.items():
        if len(model_seeds) == 0:
            continue
        median_res[model_name] = {}
        std_res[model_name] = {}

        first_seed = list(model_seeds.keys())[0]
        k_array = model_seeds[first_seed]["k values"]
        median_res[model_name]["k values"] = k_array
        if "k_cnct" in model_seeds[first_seed].keys():
            k_cnct = model_seeds[first_seed]["k_cnct"]
            median_res[model_name]["k_cnct"] = k_cnct
        if "ARI" in metrics:
            resolutions = model_seeds[first_seed]["resolutions"]
            median_res[model_name]["resolutions"] = resolutions

        for metric_name in metrics:
            if metric_name in ["ARI", "NMI"]:
                n_points = len(resolutions)
            elif metric_name == "graph_connectivity":
                n_points = len(k_cnct)
            else:
                n_points = len(k_array)
            array_res = np.zeros((len(model_seeds.keys()), n_points))
            if metric_name == "celltype_transfer":
                for i, seed_res in enumerate(model_seeds.values()):
                    array_res[i, :] = np.mean([seed_res[n] for n in seed_res.keys()
                                               if "transfer" in n and len(seed_res[n]) > 0], axis=0)
            elif "foscttm" in metric_name:
                array_res = [seed_res[metric_name] for seed_res in model_seeds.values()]
            else:
                for i, seed_res in enumerate(model_seeds.values()):
                    array_res[i, :] = seed_res[metric_name]

            median_res[model_name][metric_name] = np.median(array_res, axis=0)
            std_res[model_name][metric_name] = np.std(array_res, axis=0)
    return median_res, std_res


def eval_imputations_wrap(save_path, remove_data, res_dirs, k_neighbors):
    full_results = {model: {"decoder": {}, "latents": {}} for model in res_dirs}
    median_res = {model: {"decoder": {}, "latents": {}} for model in res_dirs}
    mean_res = {model: {"decoder": {}, "latents": {}} for model in res_dirs}
    heldout_adata = anndata.read(os.path.join(save_path, "heldout_adata"))
    fish_mask = np.array([name.endswith("FISH") for name in heldout_adata.obs_names])
    for model in res_dirs:
        seeds = [name.split("_")[1].split(".")[0]
                 for name in os.listdir(os.path.join(save_path, model))
                 if name.startswith("latents_")]
        for seed in seeds:
            latents_key = "latents_{}".format(seed)
            heldout_adata.obsm[latents_key] = pd.read_csv(os.path.join(save_path, model, "latents_{}.csv".format(seed)),
                                                          index_col=0, header=0).loc[heldout_adata.obs.index]
            imputations = {}
            if os.path.exists(os.path.join(save_path, model, "imputations_latent_{}.csv".format(seed))):
                imputations["latent"] = np.array(pd.read_csv(os.path.join(save_path, model,
                                                                          "imputations_latent_{}.csv".format(seed)),
                                                             index_col=0, header=0)
                                                 .loc[heldout_adata[fish_mask].obs_names,
                heldout_adata.var_names].values)
            else:
                imputations["latents"] = impute_from_latents(adata=heldout_adata, latents_key=latents_key,
                                                             fish_mask=fish_mask, k_neighbors=k_neighbors)
                latents_imputations_df = pd.DataFrame(imputations["latents"],
                                                      index=heldout_adata[fish_mask].obs_names,
                                                      columns=heldout_adata.var_names)
                latents_imputations_df.to_csv(os.path.join(save_path, model, "imputations_latent_{}.csv".format(seed)))
            if os.path.exists(os.path.join(save_path, model, "imputations_{}.csv".format(seed))):
                imputations["decoder"] = np.array(pd.read_csv(os.path.join(save_path, model,
                                                                           "imputations_{}.csv".format(seed)),
                                                              index_col=0, header=0)
                                                  .loc[heldout_adata[fish_mask].obs_names,
                heldout_adata.var_names].values)

            for method, imputed_values in imputations.items():
                scaled = False
                if method == "decoder" and "scaled.txt" in os.listdir(os.path.join(save_path, model)):
                    scaled = True

                full_results[model][method][seed] = eval_imputations(
                    imputations=imputed_values, heldout_adata=heldout_adata[fish_mask],
                    scaled=scaled
                )

        for method in imputations.keys():
            median_res[model][method], mean_res[model][method] = aggreg_imput_results(full_results[model][method])
    if remove_data:
        rm_anndatas = ["heldout_adata", "data_processed.h5mu", "mdata_baseline.h5mu"]
        for anndata_name in rm_anndatas:
            if os.path.exists(os.path.join(save_path, anndata_name)):
                os.remove(os.path.join(save_path, anndata_name))

    with open(os.path.join(save_path, "full_results.pickle"), 'wb') as handle:
        pickle.dump(full_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(save_path, "median_results.pickle"), 'wb') as handle:
        pickle.dump(median_res, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(save_path, "mean_results.pickle"), 'wb') as handle:
        pickle.dump(mean_res, handle, protocol=pickle.HIGHEST_PROTOCOL)


def aggreg_imput_results(seed_results):
    # Take median over seed runs
    model_scenar_res = np.median(np.array([list(seed_res.values())
                                           for seed_res in seed_results.values()]), axis=0)
    # Aggregate over the different genes that compose the scenario
    return np.median(model_scenar_res), np.mean(model_scenar_res)


def eval_imputations(imputations, heldout_adata, scaled):
    ground_truth = heldout_adata.X
    if scaled:
        ground_truth = ground_truth / (1 + np.array(heldout_adata.obs["library_size"].values)[:, np.newaxis])
    rez = {gene: spearmanr(imputations[:, j], ground_truth[:, j]).correlation
           for j, gene in enumerate(heldout_adata.var_names)}
    return rez


def impute_from_latents(adata, latents_key, fish_mask, k_neighbors):
    neigh = KNeighborsRegressor(n_neighbors=k_neighbors)
    neigh.fit(adata[~fish_mask].obsm[latents_key], adata[~fish_mask].X)
    imputations = neigh.predict(adata[fish_mask].obsm[latents_key])
    return imputations


def local_postprocess_wrap(root_path, modality, k_array, resolutions, data_name,
                           n_subsamples, baseline, list_methods, metrics=None,
                           k_cnct=[]):
    for k, cfg_name in enumerate(list_methods):
        eval_wrap(root_path=root_path, cfg_name=cfg_name, modality=modality,
                  k_array=k_array, resolutions=resolutions, k_cnct=k_cnct,
                  data_name=data_name, baseline=baseline, n_subsamples=n_subsamples,
                  metrics=metrics)
