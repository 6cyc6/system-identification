# System Identification 
## Installation
### Conda environment:
Download anaconda or miniconda from https://docs.conda.io/en/latest/miniconda.html and install it following the online instructions 
#### Add addtional channel for package installation:
```
conda config --append channels conda-forge
```
#### Create an virtual environment with anaconda:
```
conda env create -f environment.yml
```
### Install package
```
pip install -e .
```
## Running experiments 
### Active the virtual environment: 
```
conda activate sysid
```
### Run the script
All scripts are in experiments, for exacmple, generate excitation for identifying the systems, you can run:
```
python experiments/generate_excitation.py
```