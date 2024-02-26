import os
import traceback
import anndata
import torch.cuda
import numpy as np
import muon as mu

from scconfluence.model import ScConfluence
from scconfluence.unimodal import AutoEncoder
from benchmark.preprocessing import preprocess_mdata, mdata_to_adata


def load_data(data_name, pathprefix=""):
    datasets = ["cell_lines_0", "cell_lines_1", "cell_lines_2", "cell_lines_3",
                "pbmc10X", "OP_Multiome", "bmcite", "OP_Cite", "smFISH", "3omics",
                "Patch"]
    if data_name in datasets:
        mdata = mu.read_h5mu(os.path.join(pathprefix,
                                          "data/{}.h5mu.gz".format(data_name)))
    else:
        raise ValueError(f"data name \'{data_name}\' not recognized")
    mdata.uns["data_name"] = data_name
    return mdata


def keep_batches(mdata, batches):
    for mod, batch_list in batches.items():
        mask = np.any([mdata[mod].obs["batch"] == b for b in batch_list], axis=0)
        mdata.mod[mod] = mdata[mod][mask].copy()
    mdata.update()


def set_exp_dir(data_name, exp_name):
    save_path_root = os.path.join("exp_results", data_name, exp_name)
    if not os.path.exists(os.path.join("exp_results", data_name)):
        os.mkdir(os.path.join("exp_results", data_name))
    k = 0
    save_path = "{}_v{}".format(save_path_root, k)
    while os.path.exists(save_path):
        k += 1
        save_path = "{}_v{}".format(save_path_root, k)
    os.mkdir(save_path)
    return save_path


def get_test_data(root_path, modality, return_data_path):
    data_path = os.path.join(root_path, "data_processed{}".format(".h5mu" if "+" in modality else ""))
    if "+" in modality:
        data = mdata_to_adata(mu.read(data_path))
    else:
        data = anndata.read(data_path)

    if return_data_path:
        return data, data_path
    else:
        return data


def prepro_wrap(data_name, spl_pct, modality, data_cfg, save_path,
                imput_scenar=None, baseline=False):
    mdata = load_data(data_name=data_name)
    data_cfg["name"] = data_name
    if imput_scenar is not None:
        data_cfg["heldout_fish_genes"] = imput_scenar
    data, data_cfg, heldout_adata = preprocess_mdata(mdata=mdata, modality=modality,
                                                     data_cfg=data_cfg, spl_pct=spl_pct,
                                                     baseline=baseline)
    if baseline:
        data.write(os.path.join(save_path, "mdata_baseline.h5mu"))
    else:
        data.write(os.path.join(save_path,
                                "data_processed{}".format(".h5mu"
                                                          if "+" in modality else "")))
    if not (imput_scenar is None or os.path.exists(os.path.join(save_path, "heldout_adata"))):
        heldout_adata.write(os.path.join(save_path, "heldout_adata"))


def keep_hvg_only(data, input_cfg, modality):
    if "+" in modality:
        for mod in modality.split("+"):
            data.mod[mod] = hvg_adata(adata=data[mod], input_cfg=input_cfg[mod])
    else:
        data = hvg_adata(adata=data, input_cfg=input_cfg[modality])
    return data


def hvg_adata(adata, input_cfg):
    if input_cfg["hvg_only"]:
        features_mask = np.logical_or(adata.var["raw_hvg"], adata.var["norm_hvg"])
    else:
        features_mask = np.ones(adata.shape[1]).astype(bool)
    return adata[:, features_mask].copy()


def run_model_wrap(tr_cfg, modality, data_path, save_path: str,
                   test_mode, seed, imputation_genes=[]):
    """
    Initializes and trains different models to compare configurations
    :param tr_cfg: Dictionary of model and training parameters
    :param data_path: Data path
    :param modality: indicates which modality this experiment runs on
    :param save_path: string indicating the folder where the results should be saved
    :param test_mode: boolean indicating model must be trained on fewer examples just for testing purposes
    :param seed: random seed used for this run
    :param imputation_genes: list of genes to impute, if not empty then instead of computing imputations instead of
    latent embeddings at inference time
    :return: Trained models and their training loss histories
    """

    if "+" in modality:
        data = mu.read(data_path)
    else:
        data = anndata.read(data_path)
    data = keep_hvg_only(data=data, input_cfg=tr_cfg["input_choice"], modality=modality)

    print("Retrieving data set")

    torch.manual_seed(seed)
    try:
        print("Initializing model with seed {}".format(seed))
        if "+" in modality:
            umodal_cfgs = {mod: {**tr_cfg["model"][mod]} for mod in modality.split("+")}
            for mod in modality.split("+"):
                umodal_cfgs[mod]["n_latent"] = tr_cfg["model"]["n_latent"]
                umodal_cfgs[mod]["modality"] = mod
                for n in ["rep_in", "rep_out"]:
                    umodal_cfgs[mod][n] = tr_cfg["input_choice"][mod][n]
            unimodal_aes = {mod: AutoEncoder(data.mod[mod], **umodal_cfgs[mod])
                            for mod in modality.split("+")}

            model = ScConfluence(mdata=data, unimodal_aes=unimodal_aes,
                                 **tr_cfg["model"]["mmae_args"])
        else:
            umodal_cfg = dict()
            umodal_cfg["n_latent"] = tr_cfg["model"]["n_latent"]
            umodal_cfg["modality"] = modality
            umodal_cfg.update(tr_cfg["model"][modality])
            model = AutoEncoder(data, **umodal_cfg)
        print("Starting training")
        model.fit(save_path=save_path, test_mode=test_mode, **tr_cfg["train"])
        print("Training completed, starting prediction on full data set.")
        model_latents = model.get_latent(use_cuda=tr_cfg["train"]["use_cuda"],
                                         batch_size=tr_cfg["train"]["batch_size"],
                                         pin_memory=tr_cfg["train"]["pin_memory"],
                                         num_workers=tr_cfg["train"]["num_workers"])
        model_latents.to_csv(os.path.join(save_path, "latents_{}.csv".format(seed)))
        if len(imputation_genes) > 0:
            model_imputations = model.get_imputation(impute_from="fish", impute_to="rna",
                                                     use_cuda=tr_cfg["train"]["use_cuda"],
                                                     batch_size=tr_cfg["train"]["batch_size"],
                                                     pin_memory=tr_cfg["train"]["pin_memory"],
                                                     num_workers=tr_cfg["train"]["num_workers"])
            rna_imput_idxes = np.array([list(data["rna"].var_names).index(gene) for gene in imputation_genes])
            imputations = model_imputations.iloc[:, rna_imput_idxes]
            imputations.columns = imputation_genes
            imputations.to_csv(os.path.join(save_path, "imputations_{}.csv".format(seed)))
        elif modality == "rna+fish" and tr_cfg["model"]["impute"]:
            model_imputations = model.get_imputation(impute_from="fish", impute_to="rna",
                                                     use_cuda=tr_cfg["train"]["use_cuda"],
                                                     batch_size=tr_cfg["train"]["batch_size"],
                                                     pin_memory=tr_cfg["train"]["pin_memory"],
                                                     num_workers=tr_cfg["train"]["num_workers"])
            if tr_cfg["input_choice"]["rna"]["hvg_only"]:
                features_mask = np.logical_or(data["rna"].var["raw_hvg"], data["rna"].var["norm_hvg"])
            else:
                features_mask = np.ones(data["rna"].shape[1]).astype(bool)
            model_imputations.columns = data["rna"][:, features_mask].var_names
            model_imputations.to_csv(os.path.join(save_path, "imputations_{}.csv".format(seed)))


    except:
        error_msg = "The training or initialization of the model with seed {} failed with the following message: " \
                    "\n {}".format(seed, traceback.format_exc())
        print(error_msg)
        with open(os.path.join(save_path, "error_traceback_{}".format(seed)), "w") as f:
            f.write(error_msg)


def local_training_wrap(modality, save_path, model_cfg, test_mode, n_seeds,
                        imputation_genes=[]):
    data_path = os.path.join(save_path,
                             "data_processed{}".format(".h5mu"
                                                       if "+" in modality else ""))
    np.random.seed()
    train_seeds = np.random.choice(10000, n_seeds, replace=False).tolist()
    save_path_model = os.path.join(save_path, "run-scConfluence")
    os.mkdir(save_path_model)
    for seed in train_seeds:
        run_model_wrap(tr_cfg=model_cfg, modality=modality, data_path=data_path,
                       save_path=save_path_model, test_mode=test_mode, seed=seed,
                       imputation_genes=imputation_genes)
