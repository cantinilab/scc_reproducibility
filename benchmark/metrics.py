import numpy as np
import scipy
import scanpy as sc
import pandas as pd
from scipy.sparse import csr_matrix
import anndata
from scib.metrics import graph_connectivity

from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

VALID_METRICS = {"accuracy", "purity", "mixing", "overlap", "spearman", "batch_entropy", "modality_entropy",
                 "celltype_transfer", "foscttm", "foscttm_full", "graph_connectivity"}


def get_joint_knn_graph(latents_1, latents_2, k, knn_metric):
    r"""
    Computes k nearest neighbors of latents_1 among latents_2
    :param latents_1: array of points
    :param latents_2: array of points or None if base and query are the same
    :param k: number of neighbors to keep
    :param knn_metric: metric used to measure distances (must be a sklearn metric keyword like manhattan)
    :return:
    """
    exclude_same = False
    if latents_2 is None:
        latents_2 = latents_1
        exclude_same = True
    k = min(k, len(latents_2))
    if exclude_same:
        k += 1
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree', metric=knn_metric).fit(latents_2)
    _, indices = nbrs.kneighbors(latents_1)
    if exclude_same:
        indices = indices[:, 1:]
    return indices


def get_knn_classif(latents_query, latents_ref, labels_ref, k):
    knn_graph = get_joint_knn_graph(latents_1=latents_query, latents_2=latents_ref, k=k, knn_metric="minkowski")
    labels_neighbors = labels_ref[knn_graph]
    pred_counts = np.concatenate([(labels_neighbors == i).sum(1, keepdims=True) for i in np.unique(labels_ref)], axis=1)
    pred_labels = np.unique(labels_ref)[np.argmax(pred_counts, axis=1)]
    return pred_labels


def get_knn_accuracy(knn_indices, matching_idxs):
    """
    Measures the proportion of cells for which the embedding of one modality is among the k nearest neighbours of the
    embedding of its other modality
    Higher is better
    """
    grid = matching_idxs.reshape((-1, 1))
    grid = grid.repeat(knn_indices.shape[1], axis=1)
    matches = np.sum(knn_indices == grid, axis=1)
    acc = matches.mean()
    return acc


def get_knn_purity(knn_indices, labels_x, labels_y):
    """
    Measure for each cell the proportion of its kneighbors that belong to the same class and average this over all cells
    """
    neighbor_labels = labels_y[knn_indices]
    same_class_match = (labels_x[:, np.newaxis] == neighbor_labels)
    return np.mean(same_class_match)


def get_knn_clf_accuracy(knn_indices, true_labels, labels_neighbors):
    """

    """
    pred_labels = labels_neighbors[knn_indices]
    pred_counts = np.concatenate([(pred_labels == i).sum(1, keepdims=True) for i in np.unique(true_labels)], axis=1)
    pred_labels = np.unique(true_labels)[np.argmax(pred_counts, axis=1)]
    acc = np.mean(true_labels == pred_labels)
    return acc


def get_nearest_neighbor_overlap(x1, x2, k=100):
    """
    Compute the overlap between the k-nearest neighbor graph of x1 and x2.
    Using Spearman correlation of the adjacency matrices.
    Compute the overlap fold enrichment between the protein and mRNA-based cell 100-nearest neighbor
        graph and the Spearman correlation of the adjacency matrices.
    """
    if len(x1) != len(x2):
        raise ValueError("len(x1) != len(x2)")
    n_samples = len(x1)
    k = min(k, n_samples - 1)
    nne = NearestNeighbors(n_neighbors=k + 1)  # "n_jobs=8
    nne.fit(x1)
    kmatrix_1 = nne.kneighbors_graph(x1) - scipy.sparse.identity(n_samples)
    nne.fit(x2)
    kmatrix_2 = nne.kneighbors_graph(x2) - scipy.sparse.identity(n_samples)

    # 1 - spearman correlation from knn graphs
    spearman_correlation = scipy.stats.spearmanr(
        kmatrix_1.A.flatten(), kmatrix_2.A.flatten()
    )[0]
    # 2 - fold enrichment
    set_1 = set(np.where(kmatrix_1.A.flatten() == 1)[0])
    set_2 = set(np.where(kmatrix_2.A.flatten() == 1)[0])
    fold_enrichment = (
        len(set_1.intersection(set_2))
        * n_samples ** 2
        / (float(len(set_1)) * len(set_2))
    )
    return spearman_correlation, fold_enrichment


def get_mix_entropy(knn_indices, batch_labels):
    """returns the normalized (max score is 1) batch entropy score"""
    neighbor_batches = batch_labels[knn_indices]
    batch_values, batch_counts = np.unique(batch_labels, return_counts=True)
    batch_prop = (batch_counts / np.sum(batch_counts))[np.newaxis]
    prob_mtx = np.concatenate([np.mean(neighbor_batches == batch_idx, axis=1, keepdims=True)
                               for batch_idx in batch_values], axis=1)
    # renormalize probabilities by taking account proportion of batches in full data set
    norm_prob = (prob_mtx / batch_prop) / np.sum(prob_mtx / batch_prop, axis=1, keepdims=True)
    # replace 0 proportions with 1. to avoid warning when computing log(0)
    norm_prob[norm_prob == 0.] = 1.
    local_entropies = (-np.log(norm_prob) * norm_prob)
    local_entropies[np.isnan(local_entropies)] = 0.
    local_entropies = np.sum(local_entropies, axis=1)
    return np.mean(local_entropies) / np.log(len(batch_values))


def get_foscttm(knn_indices, pair_idxes):
    n = knn_indices.shape[1]
    pair_pos = pair_idxes.reshape((-1, 1)).repeat(repeats=n, axis=1)
    pos_mask = (knn_indices == pair_pos)
    pos_rank = np.arange(n).reshape((1, -1)).repeat(axis=0, repeats=knn_indices.shape[0])[pos_mask] / n
    return pos_rank


def knn_graphs_wrapper(latents, modalities, pair_idxs, max_k, metrics_to_compute, modality_labels, connectivities,
                       knn_metric, spl_pair_idx=None):
    if connectivities is not None:
        try:
            knn_graph_max = connectivities.todense().argsort(axis=1)[:, ::-1][:, :max_k]
        except:
            knn_graph_max = connectivities.argsort(axis=1)[:, ::-1][:, :max_k]
    else:
        knn_graph_max = get_joint_knn_graph(latents, None, max_k, knn_metric)

    umodal_graphs_max = None
    if modality_labels is not None and "batch_entropy" in metrics_to_compute:
        umodal_graphs_max = {}
        for mod in modalities:
            if connectivities is not None:
                try:
                    umodal_graphs_max[mod] = connectivities[modality_labels == mod, :][:, modality_labels == mod].todense().\
                                                 argsort(axis=1)[:, ::-1][:, :max_k]
                except:
                    umodal_graphs_max[mod] = connectivities[modality_labels == mod, :][:, modality_labels == mod]. \
                                                 argsort(axis=1)[:, ::-1][:, :max_k]
            else:
                umodal_graphs_max[mod] = get_joint_knn_graph(latents[modality_labels == mod], None, max_k, knn_metric)

    crossmod_graphs_max = None
    if modality_labels is not None and ("celltype_transfer" in metrics_to_compute or "accuracy" in metrics_to_compute):
        if len(modalities) > 2:
            raise ValueError("Celltype transfer not implemented for more than 2 modalities")
        crossmod_graphs_max = {}
        for mod in modalities:
            if connectivities is not None:
                try:
                    crossmod_graphs_max[mod] = connectivities[modality_labels == mod, :][:, modality_labels != mod].\
                                                   todense().argsort(axis=1)[:, ::-1][:, :max_k]
                except:
                    crossmod_graphs_max[mod] = connectivities[modality_labels == mod, :][:, modality_labels != mod]. \
                                                   argsort(axis=1)[:, ::-1][:, :max_k]
            else:
                crossmod_graphs_max[mod] = get_joint_knn_graph(latents[modality_labels == mod],
                                                               latents[modality_labels != mod], max_k, knn_metric)

    crossmod_graphs_full = None
    if modality_labels is not None and "foscttm" in metrics_to_compute:
        if len(modalities) > 2:
            raise ValueError("FOSCTTM not implemented for more than 2 modalities")
        crossmod_graphs_full = {}
        for mod1, mod2 in [(modalities[0], modalities[1]), (modalities[1], modalities[0])]:
            if connectivities is not None:
                try:
                    crossmod_graphs_full[mod1] = connectivities[modality_labels == mod1][pair_idxs[mod1]][:, modality_labels == mod2].\
                                                   todense().argsort(axis=1)[:, ::-1]
                except:
                    crossmod_graphs_full[mod1] = connectivities[modality_labels == mod1][pair_idxs[mod1]][:, modality_labels == mod2].\
                                                   argsort(axis=1)[:, ::-1]
            else:
                crossmod_graphs_full[mod1] = get_joint_knn_graph(latents[modality_labels == mod1].iloc[pair_idxs[mod1]],
                                                                 latents[modality_labels == mod2],
                                                                 sum(modality_labels == mod2), knn_metric)

    sampled_graphs_full = None
    if modality_labels is not None and "foscttm_full" in metrics_to_compute:
        sampled_graphs_full = {}
        for mod1, mod2 in [(modalities[0], modalities[1]), (modalities[1], modalities[0])]:
            if connectivities is not None:
                try:
                    sampled_graphs_full[mod1] = connectivities[modality_labels == mod1][pair_idxs[mod1][spl_pair_idx]].\
                                                   todense().argsort(axis=1)[:, ::-1]
                except:
                    sampled_graphs_full[mod1] = connectivities[modality_labels == mod1][pair_idxs[mod1][spl_pair_idx]].\
                                                   argsort(axis=1)[:, ::-1]
            else:
                sampled_graphs_full[mod1] = get_joint_knn_graph(latents[modality_labels == mod1].iloc[pair_idxs[mod1][spl_pair_idx]],
                                                                latents, len(modality_labels), knn_metric)
    return knn_graph_max, umodal_graphs_max, crossmod_graphs_max, crossmod_graphs_full, sampled_graphs_full


def compute_knn_metrics(latents, k_array, metrics_to_compute, labels=None, connectivities=None,
                        batch_labels=None, modality_labels=None, pair_idxs=None, knn_metric="manhattan", k_cnct=None):
    if pair_idxs is None:
        pair_idxs = []
    modalities = np.unique(modality_labels)
    for name in metrics_to_compute:
        assert name in VALID_METRICS, "the metric called <{}> is not implemented".format(name)
    if "graph_connectivity" in metrics_to_compute:
        assert k_cnct is not None, "k_cnct should be provided in order to compute graph connectivity"
    results = {"accuracy": [], "modality_entropy": [], "batch_entropy": [], "purity": [],
               "k values": k_array, "batch_entropy|ATAC": [], "batch_entropy|ADT": [], "batch_entropy|GEX": [],
               "batch_entropy|FISH": [], "celltype_transfer|FISH": [], "celltype_transfer|IMG": [],
               "celltype_transfer|ATAC": [], "celltype_transfer|ADT": [], "celltype_transfer|GEX": [],
               "k_cnct": k_cnct, "graph_connectivity": []}
    max_k = max(k_array)

    spl_pair_idx = None
    if "foscttm_full" in metrics_to_compute:
        spl_pair_idx = np.random.choice(len(pair_idxs[modalities[0]]), size=min(1000, len(pair_idxs[modalities[0]])),
                                        replace=False)
    knn_graph_max, umodal_graphs_max, crossmod_graphs_max, crossmod_graphs_full, sampled_graphs_full = \
        knn_graphs_wrapper(latents=latents, modalities=modalities, pair_idxs=pair_idxs, max_k=max_k,
                           metrics_to_compute=metrics_to_compute, modality_labels=modality_labels,
                           connectivities=connectivities, knn_metric=knn_metric, spl_pair_idx=spl_pair_idx)

    if "foscttm" in metrics_to_compute and crossmod_graphs_full is not None:
        foscttm = []
        for mod1, mod2 in [(modalities[0], modalities[1]), (modalities[1], modalities[0])]:
            foscttm.append(get_foscttm(crossmod_graphs_full[mod1], pair_idxs[mod2]))
        results["foscttm"] = np.mean(foscttm)

    if "foscttm_full" in metrics_to_compute and sampled_graphs_full is not None:
        foscttm_full = []
        for mod1, mod2 in [(modalities[0], modalities[1]), (modalities[1], modalities[0])]:
            # pair_idx are indexes inside the samples of one modality
            foscttm_full.append(get_foscttm(sampled_graphs_full[mod1],
                                            np.where(modality_labels == mod2)[0][pair_idxs[mod2][spl_pair_idx]]))
        results["foscttm_full"] = np.mean(foscttm_full)

    for k in k_array:
        knn_graph = knn_graph_max[:, :k]

        if "accuracy" in metrics_to_compute:
            if len(modalities) > 2:
                raise ValueError("Matching accuracy is not implemented for more than 2 modalities")
            if len(pair_idxs[modalities[0]]) > 0:
                acc1 = get_knn_accuracy(crossmod_graphs_max[modalities[0]][:, :k][pair_idxs[modalities[0]]],
                                        pair_idxs[modalities[1]])
                acc2 = get_knn_accuracy(crossmod_graphs_max[modalities[1]][:, :k][pair_idxs[modalities[1]]],
                                        pair_idxs[modalities[0]])
                results["accuracy"].append(0.5 * (acc1 + acc2))
            else:
                assert False, "paired samples should be provided in order to compute knn accuracy"

        if "modality_entropy" in metrics_to_compute:
            if modality_labels is not None:
                results["modality_entropy"].append(get_mix_entropy(knn_graph, modality_labels))
            else:
                assert False, "class labels should be provided in order to compute knn purity"

        if "batch_entropy" in metrics_to_compute:
            if batch_labels is not None:
                if modality_labels is None:
                    results["batch_entropy"].append(get_mix_entropy(knn_graph, batch_labels))
                else:
                    for mod in umodal_graphs_max.keys():
                        umodal_knn_graph = umodal_graphs_max[mod][:, :k]
                        results["batch_entropy|{}".format(mod)].append(
                            get_mix_entropy(umodal_knn_graph, batch_labels[modality_labels == mod]))
            else:
                assert False, "class labels should be provided in order to compute knn purity"

        if "purity" in metrics_to_compute:
            if labels is not None:
                purity = get_knn_purity(knn_graph, labels, labels)
                results["purity"].append(purity)
            else:
                assert False, "class labels should be provided in order to compute knn purity"

        if "celltype_transfer" in metrics_to_compute:
            if labels is not None:
                for mod in crossmod_graphs_max.keys():
                    crossmod_graph = crossmod_graphs_max[mod][:, :k]
                    results["celltype_transfer|{}".format(mod)].append(
                        get_knn_clf_accuracy(crossmod_graph, labels[modality_labels == mod],
                                             labels[modality_labels != mod]))
            else:
                assert False, "class labels should be provided in order to compute knn celltype transfer"
    if "graph_connectivity" in metrics_to_compute:
        results["graph_connectivity"] = []
        adata = anndata.AnnData(X=latents)
        adata.obs["celltype"] = pd.Categorical(labels)
        for k in k_cnct:
            del adata.obsp
            if connectivities is not None:
                x_idx = np.repeat(np.arange(adata.shape[0]).reshape((-1, 1)), k, axis=1)
                adata.obsp["connectivities"] = csr_matrix((np.ones(adata.shape[0] * k),
                                                           (x_idx.flatten(), np.array(knn_graph_max[:, :k]).flatten())),
                                                          shape=(adata.shape[0], adata.shape[0]))
                adata.uns["neighbors"] = []
            else:
                sc.pp.neighbors(adata, n_neighbors=k)
            results["graph_connectivity"].append(graph_connectivity(adata, label_key="celltype"))
    return results


def get_clustering_metrics(adata, resolutions, cnct_key=None):
    results = {"ARI": [], "NMI": [], "resolutions": resolutions}
    if cnct_key is None:
        sc.pp.neighbors(adata, n_neighbors=15)
        cnct_key = "connectivities"
    classes = list(np.unique(adata.obs["celltype"].values))
    true_labels = np.array([classes.index(ctype) for ctype in adata.obs["celltype"].values])
    for res in resolutions:
        sc.tl.louvain(adata, resolution=res, key_added="louvain", obsp=cnct_key)
        pred_labels = np.array(adata.obs["louvain"].values).astype(int)
        del adata.obs["louvain"]
        results["ARI"].append(adjusted_rand_score(true_labels, pred_labels))
        results["NMI"].append(normalized_mutual_info_score(true_labels, pred_labels))
    return results

