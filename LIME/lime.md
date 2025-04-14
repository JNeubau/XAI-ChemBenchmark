# LIME

LIME (Local Interpretable Model-agnostic Explanations) is a technique used to explain the predictions of any classifier in a local, interpretable manner. It works by perturbing the input data and observing the changes in the predictions, allowing for the construction of a local surrogate model that approximates the behavior of the complex model in the vicinity of the instance being explained.

# LIME Cross-Validation Pipeline

---

## **LIME Cross-Validation Pipeline**

### **Purpose**
The `CrossValidationLIMEPipeline` class is designed to:
1. Train machine learning models using cross-validation.
2. Evaluate model performance using specified metrics.
3. Generate LIME explanations for individual predictions.
4. Visualize and save the explanations and chemical space.

---

### **Key Components**

#### **1. Initialization**
The pipeline is initialized with the following parameters:
- **`X`**: DataFrame containing feature data.
- **`y`**: DataFrame containing the target variable.
- **`z`**: DataFrame containing SMILES strings (chemical representations).
- **`folds`**: List of cross-validation folds.
- **`metrics`**: List of evaluation metrics (e.g., SMAPE, RMSE).
- **`save_dir`**: Directory to save results and plots.
- **`data_name`**: Name of the dataset.
- **`hyperparam_opt`**: Boolean indicating whether to perform hyperparameter optimization.
- **`verbose`**: Boolean for printing detailed logs.

#### **2. Model Training**
The `train_pipeline` method:
1. Splits the data into training and testing sets for each fold.
2. Tunes the model using grid search (if `hyperparam_opt` is enabled).
3. Trains the model on the training set.
4. Evaluates the model on the test set using the specified metrics.
5. Generates LIME explanations for the test set.

#### **3. LIME Explanation Generation**

#### **Usage Example**
```python
beta = exmol.lime_explain(
    examples,
    descriptor_type='MACCS',
    return_beta=True
)
```

The `generate_lime_explanations` method:
1. Uses the `exmol` library to sample the chemical space around a given molecule.
2. Generates MACCS fingerprints for the sampled molecules.
3. Filters the fingerprints based on selected keys from a reference file (`maccs_merged.csv`).
4. Uses the filtered fingerprints to make predictions with the trained model.
5. Generates LIME explanations and visualizes them using `exmol.plot_descriptors`.

#### **4. Visualization**
The pipeline saves the following visualizations:
- **Descriptor Plots**: Visualize the importance of features (e.g., MACCS keys) for individual predictions.
- **Chemical Space Plots**: Visualize the sampled chemical space and counterfactuals using `exmol.plot_space`.

---

## **Workflow**

### **1. Data Preparation**
- Input data includes:
  - Feature matrix (`X`).
  - Target variable (`y`).
  - SMILES strings (`z`).
- The data is split into training and testing sets using cross-validation.

### **2. Model Training**
- A machine learning model (e.g., Random Forest) is trained on the training set.
- Hyperparameter optimization is performed using grid search.

### **3. Explanation Generation**
- For each test instance:
  1. The chemical space is sampled using `exmol.sample_space`.
  2. MACCS fingerprints are generated for the sampled molecules.
  3. The fingerprints are filtered based on the selected keys.
  4. Predictions are made using the filtered fingerprints.
  5. LIME explanations are generated using `exmol.lime_explain`.

### **4. Visualization**
- Descriptor plots are saved as `.svg` files.

---

## **Key Methods**

### **1. `generate_lime_explanations`**
Generates LIME explanations for a given test set.

#### **Parameters**
- `model`: Trained machine learning model.
- `X_test`: DataFrame containing test features.
- `list_smiles`: Series containing SMILES strings for the test set.
- `fold`: Fold number (used for naming files).

#### **Returns**
- `lime_explanations`: List of explanations for each test instance.
- `samples`: List of sampled molecules for each test instance.

---

### **2. `local_predict_fn`**
A local prediction function used by LIME to make predictions for sampled molecules.

#### **Steps**
1. Generate MACCS fingerprints for the input SMILES strings.
2. Filter the fingerprints based on the selected keys.
3. Convert the filtered fingerprints to a NumPy array.
4. Use the trained model to make predictions.

---

### **3. `plot_descriptors`**
Visualizes the importance of descriptors for a given molecule.

#### **Usage**
```python
exmol.plot_descriptors(
    samples,
    title="Molecule SMILES",
    output_file="path/to/save/plot.svg"
)
```

---

# Documentation exmol

https://ur-whitelab.github.io/exmol/api.html#exmol.exmol.cf_explain


functions:
<!-- ### sample_space -->


# `sample_space` Function

The `sample_space` function is used to sample chemical space around a given SMILES string. It evaluates a provided function over the chemical space and generates molecules using methods like STONED, CHEMED, or SYNSPACE.

## **Function Signature**
```python
sample_space(
    origin_smiles, 
    f, 
    batched=True, 
    preset='medium', 
    data=None, 
    method_kwargs=None, 
    num_samples=None, 
    stoned_kwargs=None, 
    quiet=False, 
    use_selfies=False, 
    sanitize_smiles=True
)
```

---

## **Description**
This function samples chemical space around a given molecule represented by its SMILES string. It evaluates the provided function `f` on the generated molecules and returns a list of `Example` objects.

- By default, the number of samples (`num_samples`) is:
  - **3,000** for STONED.
  - **150** for CHEMED.
  - **1,000** for SYNSPACE.
  - If using a custom dataset, `num_samples` is set to the length of the provided data list.

- SYNSPACE is a package that generates synthetically feasible molecules from a given SMILES. Learn more at [synspace](https://github.com/whitead/synspace).

---

## **Parameters**

### **1. `origin_smiles`**
- **Type**: `str`
- **Description**: The starting SMILES string for sampling chemical space.

### **2. `f`**
- **Type**: `Union[Callable[[str, str], List[float]], Callable[[str], List[float]], Callable[[List[str], List[str]], List[float]], Callable[[List[str]], List[float]]]`
- **Description**: A function that takes SMILES or SELFIES as input and returns predicted values. It is assumed to work with lists of SMILES/SELFIES unless `batched=False`.

### **3. `batched`**
- **Type**: `bool`
- **Default**: `True`
- **Description**: Indicates whether the function `f` is batched.

### **4. `preset`**
- **Type**: `str`
- **Default**: `'medium'`
- **Description**: Determines how far across chemical space is sampled. Options include:
  - `"wide"`
  - `"medium"`
  - `"narrow"`
  - `"chemed"`
  - `"custom"`
  - `"synspace"`

  Use `"chemed"` to sample only PubChem compounds.

### **5. `data`**
- **Type**: `List[Union[str, Mol]]`
- **Default**: `None`
- **Description**: If not `None` and `preset="custom"`, this data will be used instead of generating new molecules.

### **6. `method_kwargs`**
- **Type**: `Dict`
- **Default**: `None`
- **Description**: Provides additional control over STONED


# `run_stoned` Function

The `run_stoned` function implements the STONED SELFIES algorithm to generate molecules by mutating a starting SMILES string. This function is typically not called directly; instead, use the `sample_space()` function for higher-level functionality.

---

## **Function Signature**
```python
run_stoned(
    start_smiles, 
    fp_type='ECFP4', 
    num_samples=2000, 
    max_mutations=2, 
    min_mutations=1, 
    alphabet=None, 
    return_selfies=False, 
    _pbar=None
)
```

---

## **Description**
This function generates molecules by applying mutations to a starting SMILES string. It uses the STONED SELFIES algorithm to explore chemical space. The generated molecules can be returned as SMILES strings, SELFIES strings, or both, along with their scores.

---

## **Parameters**

### **1. `start_smiles`**
- **Type**: `str`
- **Description**: The starting SMILES string for generating molecules.

### **2. `fp_type`**
- **Type**: `str`
- **Default**: `'ECFP4'`
- **Description**: The type of fingerprint to use for scoring the generated molecules.

### **3. `num_samples`**
- **Type**: `int`
- **Default**: `2000`
- **Description**: The total number of molecules to generate.

### **4. `max_mutations`**
- **Type**: `int`
- **Default**: `2`
- **Description**: The maximum number of mutations to apply to the starting SMILES string.

### **5. `min_mutations`**
- **Type**: `int`
- **Default**: `1`
- **Description**: The minimum number of mutations to apply to the starting SMILES string.

### **6. `alphabet`**
- **Type**: `Union[List[str], Set[str]]`
- **Default**: `None`
- **Description**: The alphabet to use for mutations. Typically obtained from `get_basic_alphabet()`.

### **7. `return_selfies`**
- **Type**: `bool`
- **Default**: `False`
- **Description**: If `True`, returns SELFIES strings along with SMILES strings.

### **8. `_pbar`**
- **Type**: `Optional`
- **Default**: `None`
- **Description**: Internal parameter for progress bar handling (not typically used by the user).

---

## **Return Type**
- **Type**: 
  - `Union[Tuple[List[str], List[float]], Tuple[List[str], List[str], List[float]]]`
- **Description**: Returns either:
  1. A tuple of SMILES strings and their scores.
  2. A tuple of SELFIES strings, SMILES strings, and their scores (if `return_selfies=True`).

---

## **Returns**
- **SELFIES**: The generated SELFIES strings (if `return_selfies=True`).
- **SMILES**: The generated SMILES strings.
- **SCORES**: The scores of the generated molecules.

---

## **Usage Example**
```python
from exmol import run_stoned

# Starting SMILES string
start_smiles = "CCO"

# Generate molecules using STONED
smiles, scores = run_stoned(
    start_smiles=start_smiles,
    fp_type="ECFP4",
    num_samples=100,
    max_mutations=3,
    min_mutations=1
)

# Print the first few generated molecules and their scores
for s, score in zip(smiles[:5], scores[:5]):
    print(f"SMILES: {s}, Score: {score}")
```

# `lime_explain` Function

The `lime_explain` function calculates descriptor t-statistics from a given list of `Example` objects. It is typically used to analyze the importance of molecular descriptors for explaining predictions.

---

## **Function Signature**
```python
lime_explain(
    examples, 
    descriptor_type='MACCS', 
    return_beta=True
)
```

---

## **Description**
This function processes a list of `Example` objects (generated using `sample_space`) to compute descriptor t-statistics. It can also return regression coefficient values if specified.

---

## **Parameters**

### **1. `examples`**
- **Type**: `List[Example]`
- **Description**: A list of `Example` objects, typically the output from the `sample_space` function.

### **2. `descriptor_type`**
- **Type**: `str`
- **Default**: `'MACCS'`
- **Description**: The type of molecular descriptors to use. Options include:
  - `'Classic'`
  - `'ECFP'`
  - `'MACCS'`

### **3. `return_beta`**
- **Type**: `bool`
- **Default**: `True`
- **Description**: If `True`, the function returns regression coefficient values along with the t-statistics.

---

## **Return Type**
- **Type**: `Union[Dict, Tuple[Dict, np.ndarray]]`
- **Description**: Returns either:
  1. A dictionary of descriptor t-statistics.
  2. A tuple containing the t-statistics dictionary and regression coefficient values (if `return_beta=True`).

---

## **Returns**
- **T-Statistics**: A dictionary containing the t-statistics for each descriptor.
- **Beta Coefficients**: (Optional) Regression coefficient values for the descriptors.

---

## **Usage Example**
```python
import exmol

# Generate examples using sample_space
examples = exmol.sample_space(
    origin_smiles="CCO",
    f=lambda x: [len(smile) for smile in x],  # Example prediction function
    num_samples=100
)

# Perform LIME explanation
t_statistics, beta_coefficients = exmol.lime_explain(
    examples,
    descriptor_type='MACCS',
    return_beta=True
)

# Print the t-statistics and beta coefficients
print("T-Statistics:", t_statistics)
print("Beta Coefficients:", beta_coefficients)
```

