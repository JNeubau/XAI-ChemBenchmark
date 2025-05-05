from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
import copy
from typing import Any, Iterable
import numpy as np
import pandas as pd
import os
from datetime import datetime
from XAIFlow.AI_models.models import Models
from XAIFlow.AI_models.eval_metrics import EvalMetrics, smape
from XAIFlow.utils.data_split import custom_data_split
# from XAIFlow.utils.fingerprints import Fingerprints
import exmol
from rdkit import Chem
import signal
import matplotlib.pyplot as plt
from MMACE.timeoutexception import timeout

class CrossValidationMMACEPipeline:
    """
    Cross-validation pipeline for training and evaluating models with MMACE explanations.
    """

    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        z: pd.DataFrame,
        folds: list,
        metrics: list,
        save_dir: str,
        data_name: str,
        hyperparam_opt: bool = True,
        verbose: bool = False,
        custom_alphabet: set = None,  # Add this parameter
    ):
        """
        Initialize the cross-validation pipeline.
        :param X: dataframe with features.
        :param y: dataframe with target variable.
        :param z: dataframe with SMILES strings.
        :param folds: list with cross-validation folds.
        :param metrics: list with metrics to evaluate.
        :param save_dir: path to save scores.
        :param data_name: name of the dataset.
        :param verbose: whether to print model scores.
        """
        self.X = X
        self.y = y
        self.z = z
        self.folds = folds
        self.data_name = data_name
        self.save_dir = save_dir
        self.verbose = verbose
        self.metrics = metrics
        self.hyperparam_opt = hyperparam_opt
        self.scores = None
        self.MMACE_results = None
        self.cfs = None
        self.samples = None
        self.custom_alphabet = custom_alphabet

    def tune_model(self, X_train: pd.DataFrame, y_train: pd.DataFrame, model: object, param_grid: dict | None) -> object:
        """
        Perform model tuning.
        :param X_train: training data.
        :param y_train: target variable data.
        :param model: prediction model.
        :param param_grid: dictionary with parameter grid.
        :return: model with optimized parameters.
        """
        if self.hyperparam_opt:
            scorer = make_scorer(smape, greater_is_better=False)
            folds = custom_data_split(X_train, y_train, train_size=0.6)
            opt = GridSearchCV(estimator=model, param_grid=param_grid, cv=folds, scoring=scorer, refit=True, n_jobs=-1,
                               return_train_score=True)
            opt.fit(X_train.to_numpy(), y_train[y_train.columns[0]].to_numpy())
            best_score, best_params = opt.best_score_, opt.best_params_
            model.set_params(**best_params)
            if self.verbose:
                print(f"Best score: {best_score}\nBest params: {best_params}")
        return model

    def eval_model(self, y_pred: np.array, y_test: np.array) -> dict:
        """
        Evaluate the model.
        :param y_pred: predicted values.
        :param y_test: true values.
        :return: dictionary with evaluation metrics.
        """
        partial_scores = {}
        for metric in self.metrics:
            partial_scores[metric] = EvalMetrics().evaluate(metric, y_test, y_pred)
        return partial_scores

    def init_scores_MMACE(self):
        """
        Initialize scores dictionary and MMACE results.
        """
        scores = {}
        for metric in self.metrics:
            scores[metric] = []
        self.scores = scores
        self.MMACE_results = []
        self.cfs = []
        self.samples = []

    def update_scores(self, model_scores: dict):
        """
        Update scores dictionary.
        :param model_scores: dictionary with model scores.
        """
        for metric in self.metrics:
            self.scores[metric].append(model_scores[metric])

    def aggregate_scores(self):
        """
        Aggregate scores.
        :return: aggregated scores.
        """
        results = {}
        for metric in self.metrics:
            metric_scores = self.scores[metric]
            results[metric] = round(sum(metric_scores) / len(metric_scores), 4)
        return results

    def save_results(self, results: dict, model_name: str, model_params: Any):
        """
        Save results to a file.
        :param results: dictionary with results.
        :param model_name: name of the model.
        :param model_params: model parameters.
        """
        save_scores_path = os.path.join(self.save_dir, f"results_{model_name}.txt")
        metric_lines = [f"{metric}: {score}\n" for metric, score in results.items()]
        lines = [
            "\n******************************************************\n",
            f"{datetime.now().strftime('%d-%m-%Y %H:%M:%S')}\n",
            f"{model_name}\nModel parameters: {model_params}\n",
            f"Training data: {self.data_name}\n",
            *metric_lines,
            "******************************************************\n",
        ]
        if self.verbose:
            for l in lines:
                print(l)
        with open(save_scores_path, "a", encoding="utf-8") as f:
            f.writelines(lines)

    def generate_MMACE_explanations(self, model, X_test: pd.DataFrame, list_smiles: pd.Series, y_pred: np.array, fold: int = 0) -> tuple:
        """
        Generate MMACE explanations using the exmol library.
        :param model: trained model.
        :param X_test: test data.
        :param list_smiles: series with SMILES string dla każdej instancji.
        :return: lista słowników z wyjaśnieniami dla poszczególnych instancji.
        """
        MMACE_explanations = []
        datetime_now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        def local_predict_fn(x):
            # Generate MACCS fingerprints for the input SMILES
            # print(f"Generating MACCS fingerprints for SMILES: {x}")
            fps = [list(Chem.MACCSkeys.GenMACCSKeys(Chem.MolFromSmiles(x)).ToBitString())]
            fps_df = pd.DataFrame(fps, columns=[f'maccsfingerprint{i}' for i in range(0, len(fps[0]))])

            # Load the MACCS merge file and filter columns based on selected keys
            parent_dir = os.path.dirname(os.getcwd())
            maccs_merge_path = os.path.join(parent_dir, 'data', 'maccs_merged.csv')

            if not os.path.exists(maccs_merge_path):
                raise FileNotFoundError(f"MACCS merge file not found: {maccs_merge_path}")

            maccs_merge = pd.read_csv(maccs_merge_path)
            maccs_merge = maccs_merge.loc[:, maccs_merge.columns.str.contains('maccs', case=False)]
            selected_keys = maccs_merge.columns.tolist()

            selected_keys = [key for key in selected_keys if key in fps_df.columns]

            # Filter the fingerprints DataFrame to include only the selected keys
            filtered_fps = fps_df[selected_keys]

            # Convert the filtered DataFrame to a labeled DataFrame for prediction
            labeled_fps = pd.DataFrame(filtered_fps, columns=selected_keys)

            # Use the labeled DataFrame for prediction
            prediction = model.predict(labeled_fps).flatten()

            # print(f"Prediction for SMILES {x}: {prediction}")

            return prediction
            # # Check if the prediction for the newly generated SMILES matches the original SMILES
            # original_prediction = y_pred[X_test.index[i]]
            # print(f"Comparing predictions {i}: {prediction} vs {original_prediction}")
            # if not np.array_equal(prediction, original_prediction):
            #     return 0

            # return 1
        @timeout(60)
        def plot_with_timeout(cfs):
            fkw = {"figsize": (10, 3)}
            exmol.plot_cf(cfs, figure_kwargs=fkw, mol_size=(450, 400), nrows=1)

        def export_plots_exmol(cfs, fold, i):
            
            datetime_now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            try:
                # plot_path = os.path.join(os.path.join(
                #         os.path.dirname(os.getcwd())), 'results', 'plots', 'MMACE', "local", datetime.today().strftime("%d-%m-%Y"))
                plot_dir = os.path.join(
                        os.path.dirname(os.getcwd()), 'results', 'plots', 'MMACE', "local","counterfactuals", datetime.today().strftime("%d-%m-%Y")
                    )
                os.makedirs(plot_dir, exist_ok=True)
                plot_path = os.path.join(
                        plot_dir, f"explanation_fold_{fold}_instance_{i}_{datetime_now}.png"
                    )
                # fkw = {"figsize": (10, 3)}
                print(f"fkw")
                plot_with_timeout(cfs)
                print(f"exmol plot_cf")
                plt.savefig(plot_path, bbox_inches="tight", dpi=180)
                print(f"Plot saved to {plot_path}")
            except Exception as e:
                print(f"An error occurred while plotting CFS: {e}")
            finally:
                plt.close('all')
            
            # exmol.plot_descriptors(sample_space)
            # plt.savefig("my_descriptor_plot.png", bbox_inches="tight")
            # plt.close()
            return 0

        @timeout(60)
        def plot_space_with_timeout(space, cfs):
            fkw = {"figsize": (10, 3)}
            exmol.plot_space(space, cfs, figure_kwargs=fkw, mol_size=(200, 200), offset=1)

        def export_plots_exmol_space(space,cfs, fold, i):
            datetime_now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            try:
                plot_dir = os.path.join(
                        os.path.dirname(os.getcwd()), 'results', 'plots', 'MMACE', "local","space", datetime.today().strftime("%d-%m-%Y")
                    )
                os.makedirs(plot_dir, exist_ok=True)
                plot_path = os.path.join(
                        plot_dir, f"explanation_space_fold_{fold}_instance_{i}_{datetime_now}.png"
                    )
                # fkw = {"figsize": (10, 3)}
                print(f"fkw")
                fkw = {"figsize": (8, 6)}
                font = {"family": "normal", "weight": "normal", "size": 22}


                # exmol.plot_space(space, cfs, figure_kwargs=fkw, mol_size=(200, 200), offset=1)
                plot_space_with_timeout(space, cfs)
                ax = plt.gca()
                plt.colorbar(ax.get_children()[1], ax=[ax], location="left")
                plt.savefig(plot_path, bbox_inches="tight", dpi=180)
                print(f"Plot saved to {plot_path}")
            except Exception as e:
                print(f"An error occurred while plotting CFS: {e}")
            finally:
                plt.close('all')
            return 0

        for i, instance in X_test.iterrows():
            samples_fold = []
            cfs_fold = []
            smiles = list_smiles.iloc[i]
            print(f"Processing instance {i} with SMILES: {smiles}")
            try:
                stoned_kwargs = {
                    "num_samples": 25,
                    # "alphabet": self.custom_alphabet if self.custom_alphabet else exmol.get_basic_alphabet(),
                    "max_mutations": 1,
                }
                
                samples = exmol.sample_space(
                    smiles, 
                    local_predict_fn,
                    stoned_kwargs=stoned_kwargs, 
                    quiet=True,
                    batched=False)
                samples_fold.append(samples)
                    
                # samples = exmol.sample_space(
                #     smiles, 
                #     local_predict_fn, 
                #     batched=False,
                #     # preset='chemed',
                #     num_samples=4,
                #     quiet=True,
                #     use_selfies=False)
            except Exception as e:
                print(f"An error occurred while sampling space: {e}")
                return [], [], []
            
            print(f"Samples: {len(samples)}")
            cfs = exmol.rcf_explain(
                samples,
                filter_nondrug = False,
                # delta=[-0.5,0.5],
                delta=0.5,
                nmols=4
                )
            
            export_plots_exmol(cfs, fold, i)
            # export_plots_exmol_space(samples, cfs, fold, i)
            cfs_fold.append(cfs)
           

        return MMACE_explanations, samples_fold, cfs_fold

    def train_pipeline(self, model_name: str, model_path: str | None = None) -> tuple:
        """
        Train the model.
        :param model_name: name of the model.
        :param model_path: path to saved model.
        :return: tuple with results, scores, and MMACE results.
        """
        proper_model_name, model, param_grid = Models().get_model(model_name, model_path=model_path)
        self.init_scores_MMACE()

        if self.verbose:
            print(f"Training model {proper_model_name}")
        foldid=0

        for fold in self.folds:
            train_idx, test_idx = fold

            X_train = copy.deepcopy(self.X.loc[train_idx, :]).reset_index(drop=True)
            y_train = copy.deepcopy(self.y.loc[train_idx, :]).reset_index(drop=True)
            X_test = copy.deepcopy(self.X.loc[test_idx, :]).reset_index(drop=True)
            y_test = copy.deepcopy(self.y.loc[test_idx, :]).reset_index(drop=True)
            smiles = copy.deepcopy(self.z.loc[test_idx, :]).reset_index(drop=True)

            model = self.tune_model(X_train, y_train, model, param_grid)

            model.fit(X_train, y_train[y_train.columns[0]])
            y_pred = model.predict(X_test).flatten()

            y_test_numpy = y_test.to_numpy().flatten()
            model_scores = self.eval_model(y_pred, y_test_numpy)
            self.update_scores(model_scores)

            # if i == 0:
            MMACE_explanations,samples,cfs = self.generate_MMACE_explanations(model, X_test, smiles['smiles'], fold=foldid,y_pred=y_pred)
            # self.MMACE_results.append({"fold": fold[1], "explanations": MMACE_explanations})
            # i+=1
            self.samples.append(samples)
            self.cfs.append(cfs)
            foldid+=1

        results = self.aggregate_scores()

        if len(self.save_dir) > 0:
            self.save_results(results, proper_model_name, model.get_params())

        return results, self.scores, self.cfs, self.samples
