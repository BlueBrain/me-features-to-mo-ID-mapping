# BBP-me-type-to-mol-ID

## Description:
This a repository containing configuration code that use me-types-mapper to reproduce results from Roussel et al., 
2021 (https://www.biorxiv.org/content/10.1101/2021.11.24.469815v1)

## Installation:

To install this tool, run the following steps in your shell:

```
git clone https://bbpgitlab.epfl.ch/molsys/bbp_me_type_to_mol_id.git
pip install -r bbp_me_type_to_mol_id/requirements.txt
pip install git+https://bbpgitlab.epfl.ch/molsys/me_types_mapper.git
pip install git+https://github.com/BlueBrain/BluePyEfe@da783256a4212b14d4f152687238754a99d1a78b
```

## How to use:

### Step 1: Download the data
Go into `downloader` directory
####  Morphologies and electrophysiological recording from Gouwens et al., 2019:
Due to open source license issues, the python script for downloading AIBS data from Gouwens et al. (2019) could not be 
included.
Please refer to `allensdk` from Allen Institute for Brain Science to download data from the Gouwens et al paper.
Data should be stored in the `downloader` directory in a new folder named after the dataset name (i.e. `Gouwens_2019`)
and organized  into two sub-directories: one named as `ephys_traces` containing the raw traces in nwb format and another
named `morphologies` containing the morphology reconstruction in swc format.
Downloading of AIBS data should be done in a dedicated virtual environment.
####  Morphologies and electrophysiological recording from Blue Brain Project:
Create a dedicated virtual environment for downloading BBP data. Install dependencies using
`pip install -r requirements_BBP_dl.txt`. Finally, run `python BBP_downloader.py`.

### Step 2: Extract features
Go into the `feature_extraction` directory and run first `python efeatures_extractor.py` 
and then `python mfeatures_extractor.py`.

### Step 3: Select common features
Go into the `feature_selection` directory and run first `python feature_selector.py`. It should create two directories named
`figures` and `filtered_datasets`. Then run `python compute_dataset_labels.py`.

### Step 4: Mapping
Go into the `mapping` directory and run `python clustering.py` to reproduce the main results of the paper 
(https://www.biorxiv.org/content/10.1101/2021.11.24.469815v1). To reproduce 
the other figures run `python dataset_overlap_validation.py` for figure 2, `python alpha_optimization_validation.py` for
figure 3 and `python pipeline_validation.py` for Table 1.

##  Funding & Acknowledgment:

The development of this software was supported by funding to the Blue Brain Project, a research center of the École polytechnique fédérale de Lausanne (EPFL), from the Swiss government’s ETH Board of the Swiss Federal Institutes of Technology.

Copyright (c) 2022 Blue Brain Project/EPFL
