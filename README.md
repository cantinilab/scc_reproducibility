# scConfluence reproducibility

scConfluence is a novel method for the integration of unpaired multiomics data combining 
uncoupled autoencoders and Inverse Optimal Transport to learn low-dimensional cell 
embeddings. These embeddings can then be used for visualization and clustering, 
useful for discovering subpopulations of cells, and for imputation of features across 
modalities.
This is the code used to perform the experiments and generate the figures in our 
manuscript. If you are looking for the Python package, 
[click here](https://github.com/cantinilab/scconfluence)!

## Installation
install the scconfluence package from Pypi:
```bash
pip install scconfluence
```

## Usage

### Downloading the data
All the data used in the paper has been formatted and can be downloaded from 
[here](https://figshare.com/s/ff75c0bfecd89468dc8d). All *.h5mu.gz files should then be 
placed in the data folder of this repository.

### Obtaining each method's outputs
To obtain the outputs of scConfluence, run the following command:
```bash
python run_integration.py --dataname <dataname>
```
`<dataname>` is the name of the dataset which must be one of the following:
- cell_lines_<k> for k=0,1,2,3 where k is the number of the simulation scenario for the 
unbalanced celllines experiment
- pbmc10X
- OP_Multiome
- bmcite
- OP_Cite
- smFISH
- 3omics
- Patch

To obtain the outputs of the baseline methods, add the `--baseline` flag.
```bash
python run_integration.py --dataname <dataname> --baseline
```

Similarly, to obtain the imputation results on the smFISH dataset, run the following 
command:
```bash
python run_imputation.py --dataname <dataname> (--baseline)
```

### Generating the figures
Before running any notebook, download the pickles from [there](https://figshare.com/s/10860420658b68bb1eb5)
 and place them inside a folder entitled 
`exp_results` in the root of this directory. These files contain saved results (latent 
embeddings and their evaluations) of each method run with the provided configurations as 
the training of all methods is very long. Then run the notebooks to obtain the plots 
from the paper.
As of now, only cell lines and benchmark plots are available. The rest will be added 
shortly.

## Our preprint
https://www.biorxiv.org/content/10.1101/2024.02.26.582051v1
```bibtex
@article {Samaran2024unpaired,
  author = {Jules Samaran and Gabriel Peyre and Laura Cantini},
  title = {scConfluence : single-cell diagonal integration with regularized Inverse Optimal Transport on weakly connected features},
  year = {2024},
  doi = {10.1101/2024.02.26.582051},
  publisher = {Cold Spring Harbor Laboratory},
  journal = {bioRxiv}
}
```