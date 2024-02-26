import muon as mu

from benchmark.baselines.multimap import MultimapEvaluator
from benchmark.baselines.scglue import scGlueEvaluator
from benchmark.baselines.uniport import UniportEvaluator
from benchmark.baselines.gimvi import GimVIEvaluator
from benchmark.baselines.rmethod import RmethodEvaluator


def split_baselines(baseline_cfgs):
    baseline_cpu = dict()
    baseline_gpu = dict()
    for name, dic in baseline_cfgs.items():
        if name in ["Uniport", "scGLUE", "GIMVI"]:
            baseline_gpu[name] = dic
        elif name in ["MultiMAP", "Seurat", "Liger", "MaxFuse"]:
            baseline_cpu[name] = dic
        else:
            raise ValueError("Baseline method {} was not recognized".format(name))
    return baseline_gpu, baseline_cpu


def get_baseline_evaluator(name, model_cfg, save_dir, imputation_genes):
    if name == "Uniport":
        return UniportEvaluator(save_dir=save_dir, imputation_genes=imputation_genes, **model_cfg)
    elif name == "MultiMAP":
        return MultimapEvaluator(save_dir=save_dir, imputation_genes=imputation_genes, **model_cfg)
    elif name == "scGLUE":
        return scGlueEvaluator(save_dir=save_dir, imputation_genes=imputation_genes, **model_cfg)
    elif name == "GIMVI":
        return GimVIEvaluator(save_dir=save_dir, imputation_genes=imputation_genes, **model_cfg)
    elif name == "Seurat":
        return RmethodEvaluator(method_name="seurat", save_dir=save_dir, imputation_genes=imputation_genes, **model_cfg)
    elif name == "Liger":
        return RmethodEvaluator(method_name="liger", save_dir=save_dir, imputation_genes=imputation_genes, **model_cfg)
    else:
        raise ValueError("Evaluator not implemented for {}".format(name))


def run_baseline_wrap(name, save_path_model, mdata_path, model_cfg, imputation_genes):
    evaluator = get_baseline_evaluator(name=name, model_cfg=model_cfg,
                                       save_dir=save_path_model,
                                       imputation_genes=imputation_genes)
    mdata = mu.read(mdata_path)
    evaluator.preprocess_write(mdata)
    evaluator.run()
