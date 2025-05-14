# XAI-experiments
Required: Python 3.10

## SHAP
Implemented

## SHAP-IQ
* max order = 1

Implemented

* max order => 2

Implemented

## LIME
Implemented - M

## MMACE
Implemented - M

## MEG
Required: seperate conda env with python 3.7

Implemented - A


# Folder structure:
- *data* - raw data
- *SHAP, SHAP-IQ, LIME, MMACE, MEG* - seperate folders for methods
- *XAIFlow* - collective running for SHAP, SHAP-IQ, LIME, MMACE
- *RFReg* - saved model and split data
- *results* - subfolders seperated by methods

# Running the experiments:

The script is at the mottom of the file available at:
**"MEG/meg_master/main_meg.py"**

#### Args:
- bool: do you want to train the model during this run
- dataset name (only roks for battery)
- experiment name: for directory structure
- number of folds
- path to raw data file

#### Example:
```python
mainXaiFlow(True, 'battery', 'rf_test', 5, os.path.join(os.getcwd(), 'data', 'new_maccs_merged.csv'))
```

### Note: 
Make sure you run this from conda env, which can be installed using scripts in: **"MEG/ meg_master/setup"**. 

The script was only checked for windows installation.