from abc import abstractmethod


class BaselineEvaluator():
    def __init__(self, save_dir, seeds, no_mix=False, rm_data=True, tr_length=None, imputation_genes=[]):
        self.seeds = seeds
        self.save_dir = save_dir
        self.rm_data = rm_data
        self.no_mix = no_mix
        self.tr_length = tr_length
        self.imputation_genes = imputation_genes

    @abstractmethod
    def preprocess_write(self, mdata):
        pass

    @abstractmethod
    def run(self):
        pass
