import yaml
import os
import argparse
from benchmark.exp_utils import set_exp_dir, prepro_wrap, local_training_wrap
from benchmark.eval import eval_imputations_wrap
from benchmark.baselines.utils import run_baseline_wrap


parser = argparse.ArgumentParser('Running experiment')
parser.add_argument('--baseline', action='store_true', default=False,
                    help='run baseline methods')
args = parser.parse_args()

data_name = "smFISH"
cfg_name = "smFISH_imput"
if args.baseline:
    exp_name = "baselines_imput"
else:
    exp_name = "scConfluence_imput"
cfg_path = f"configs/cfg_{cfg_name}.yml"
with open(cfg_path) as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

save_path_root = set_exp_dir(data_name=data_name, exp_name=exp_name)

print(f"Everything related to this experiment will be saved in {save_path_root}")

scenarios = cfg["scenarios"]
for j, scenar in enumerate(scenarios):
    scenar_path = os.path.join(save_path_root, "scenario_{}".format(j))
    os.mkdir(scenar_path)
    prepro_wrap(data_name=data_name, spl_pct=cfg["spl_pct"], imput_scenar=scenar,
                modality=cfg["modality"], save_path=scenar_path,
                data_cfg=cfg["data"], baseline=args.baseline)

    if not args.baseline:
        local_training_wrap(modality=cfg["modality"], save_path=scenar_path,
                            imputation_genes=scenar, model_cfg=cfg["model"],
                            **cfg["train"])
        res_dirs = ["run-scConfluence"]
    else:
        mdata_path = os.path.join(
            os.path.join(save_path_root, "scenario_{}".format(j)),
            "mdata_baseline.h5mu")
        for name, model_cfg in cfg["baseline_cfgs"].items():
            save_path_model = os.path.join(save_path_root, "scenario_{}".format(j),
                                           "run-{}".format(name))
            os.mkdir(save_path_model)
            run_baseline_wrap(name=name, save_path_model=save_path_model,
                              mdata_path=mdata_path, model_cfg=model_cfg,
                              imputation_genes=scenarios[j])
        res_dirs = [name
                    for name in os.listdir(os.path.join(save_path_root,
                                                        "scenario_0"))
                    if name.startswith("run-")]
    eval_imputations_wrap(save_path=scenar_path, res_dirs=res_dirs,
                          **cfg["eval"])
