# CondMacroAssetAlloc
Conditional macroeconomic condition encoding and asset allocation

This is part of our HKU MSc CS final year project. The project has mainly two parts - macroeconomic condition encoding and conditional asset allocation based on said macroeconomic condition encoding.

## Set up
Python 3.10

- Run `conda env create -f environments.yml`
- After creating the environment, run `conda activate macroalloc_env`

## Set up (linear)
- Make sure your working directory is `./linear` for proper relative imports.
- To run the linear model example, run `python baseline_dummy.py` inside the linear folder.
- If dependencies are missing, you are encouraged to install them using `./linear/requirements.txt` altogether, or:
    1. Install critical libraries with `python -m pip install pathos cvxpy pandas numpy tabulate statsmodels more_itertools`.
    2. Install research/visualization libraries with `python -m pip install matplotlib scikit_learn scipy`.
    3. Optionally, install the rest of the libraries mostly for data downloading.

## Get data


## Macroeconomic condition encoding

### Data
Timeseries data on a monthly frequency from FRED and Bloomberg are obtained.

### Methodology

#### Factor models

#### Machine leanring


## Conditional asset allocation
