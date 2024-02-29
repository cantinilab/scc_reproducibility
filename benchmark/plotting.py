import os

import matplotlib.pyplot as plt
import scanpy as sc
import numpy as np

from benchmark.eval import aggreg_seed_results


def bar_plot(ax, res, std_res, metric_name, method_names, colors, ylim):
    mmap_index = method_names.index("MultiMAP")
    pivot = np.delete(np.arange(len(method_names)), mmap_index)
    ax.bar(pivot, [res[n][metric_name] for n in method_names if n != "MultiMAP"],
           color=[colors[n] for n in method_names if n != "MultiMAP"])
    ax.errorbar(pivot, [res[n][metric_name] for n in method_names if n != "MultiMAP"],
                yerr=[std_res[n][metric_name] for n in method_names if n != "MultiMAP"],
                fmt="o", color='black', elinewidth=1, capsize=2, ms=3)
    ax.set_ylim(0., ylim)
    ax.set_xticks(np.arange(len(method_names)), labels=method_names)


def plot_metrics(results, knn_metrics, bar_metrics, metrics_to_plot, method_names,
                 colors, k_array, save, save_path,
                 bar_ylim=0.95, knn_ylim=(0.5, 0.95), figsize=(10, 8), title=None):
    fig, axes = plt.subplots(len(results), len(metrics_to_plot), figsize=figsize)
    if len(results) == 1:
        if len(metrics_to_plot) == 1:
            axes = np.array([axes])
        axes = axes[np.newaxis, :]
    elif len(metrics_to_plot) == 1:
        axes = axes[:, np.newaxis]
    for i, (ds_name, ds_results) in enumerate(results.items()):
        median_res, std_res = aggreg_seed_results(ds_results, metrics=metrics_to_plot)
        for k, metric_name in enumerate(knn_metrics):
            for method_name, color in colors.items():
                axes[i, k].plot(k_array, median_res[method_name][metric_name], c=color,
                                label=method_name)
                axes[i, k].errorbar(k_array, median_res[method_name][metric_name],
                                    yerr=std_res[method_name][metric_name],
                                    fmt="o", color='black', ecolor=color, elinewidth=1,
                                    capsize=2, ms=3)
            axes[i, k].set_ylim(*knn_ylim)
            axes[i, k].set_facecolor((0, 0, 0, 0))
            if i != len(results) - 1:
                axes[i, k].xaxis.set_ticklabels([])
            else:
                axes[i, k].set_xlabel("Number of neighbors")
        for k, metric_name in enumerate(bar_metrics):
            bar_plot(axes[i, k + len(knn_metrics)], res=median_res, std_res=std_res,
                     metric_name=metric_name, method_names=method_names, colors=colors,
                     ylim=bar_ylim)
            axes[i, k + len(knn_metrics)].set_facecolor((0, 0, 0, 0))
            if i != len(results) - 1:
                axes[i, k + len(knn_metrics)].xaxis.set_ticklabels([])
    for k in range(len(knn_metrics), len(metrics_to_plot)):
        axes[-1, k].set_xticklabels(method_names, rotation=90)
    axes[0, 0].legend()
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(save_path, "rez.pdf" if title is None else title
                                                                            + ".pdf"))
    else:
        plt.show()


def plot_bench_umaps(adatas, plots, s=15, n_neigh=20, alpha=0.7, save_path=None,
                     legend=True, keep_previous=False, method_name="Ours"):
    for ds_name, adata in adatas.items():
        if not keep_previous:
            adata.uns = {}
            sc.pp.neighbors(adata, use_rep=f"latents_{method_name}",
                            n_neighbors=n_neigh)
            sc.tl.umap(adata)
        for color_cat in plots[ds_name]:
            fig = plt.figure(constrained_layout=not legend, figsize=(7, 5))
            ax = fig.add_subplot()
            spls = list(adata.obs_names)
            np.random.shuffle(spls)
            plot_one_umap(adata[spls], s=s, ax=ax, color_cat=color_cat, legend=legend,
                          alpha=alpha,
                          save_path=os.path.join(save_path, "{}{}_{}{}.pdf".format(
                              "legend_" if legend else "",
                              ds_name, color_cat,
                              "_" + method_name
                              if method_name != "Ours" else "")))


def plot_one_umap(adata, s, alpha, ax, color_cat, save_path, legend):
    sc.pl.umap(adata, ax=ax, color=color_cat, s=s, alpha=alpha, title="", show=False)
    if not legend:
        ax.get_legend().remove()
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_facecolor((0, 0, 0, 0))
    plt.savefig(save_path, bbox_inches='tight')
