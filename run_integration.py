import yaml
import os

import argparse

from benchmark.exp_utils import set_exp_dir, prepro_wrap, local_training_wrap
from benchmark.eval import local_postprocess_wrap
from benchmark.baselines.utils import run_baseline_wrap

parser = argparse.ArgumentParser('Running experiment')
parser.add_argument('--dataname', type=str, help='name of the dataset')
parser.add_argument('--baseline', action='store_true', default=False,
                    help='run baseline methods')
args = parser.parse_args()

data_name = args.dataname
datasets = ["cell_lines_0", "cell_lines_1", "cell_lines_2", "cell_lines_3", "pbmc10X",
            "OP_Multiome", "bmcite", "OP_Cite", "smFISH", "3omics", "Patch"]
assert data_name in datasets, f"data name {data_name} not recognized"
assert not (args.baseline and data_name in ["3omics", "Patch"]), \
    "Baseline methods are not implemented for 3omics and Patch"

cfg_name = data_name
if data_name.startswith("cell_lines"):
    cfg_name = "cell_lines"
cfg_path = f"configs/cfg_{cfg_name}.yml"
with open(cfg_path) as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)
if args.baseline:
    exp_name = "baselines"
else:
    exp_name = "scConfluence"
save_path_root = set_exp_dir(data_name=data_name, exp_name=exp_name)

print(f"Everything related to this experiment will be saved in {save_path_root}")

prepro_wrap(data_name=data_name, spl_pct=cfg["spl_pct"], modality=cfg["modality"],
            save_path=save_path_root, data_cfg=cfg["data"], baseline=args.baseline)

if not args.baseline:
    local_training_wrap(modality=cfg["modality"], save_path=save_path_root,
                        model_cfg=cfg["model"], **cfg["train"])
    list_methods = ["run-scConfluence"]
else:
    mdata_path = os.path.join(save_path_root, "mdata_baseline.h5mu")
    for name, model_cfg in cfg["baseline_cfgs"].items():
        save_path_model = os.path.join(save_path_root, "run-{}".format(name))
        os.mkdir(save_path_model)
        run_baseline_wrap(name=name, save_path_model=save_path_model,
                          mdata_path=mdata_path, model_cfg=model_cfg,
                          imputation_genes=[])
    list_methods = ["run-{}".format(name) for name in cfg["baseline_cfgs"].keys()]
if data_name not in ["3omics", "Patch"]:
    local_postprocess_wrap(root_path=save_path_root, modality=cfg["modality"],
                           data_name=data_name, baseline=args.baseline,
                           list_methods=list_methods, **cfg["eval"])
