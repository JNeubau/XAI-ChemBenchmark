import sys

import numpy as np

from src.ranking.mean import aggregate_rankings_by_mean_position
from src.ranking.rra import aggregate_rankings_by_rra
from src.analysis.processing import lime_ranking, shap_ranking, shapiq_ranking, meg_ranking, mmace_ranking, \
    meg_cf_percent, mmace_cf_percent
from src.analysis.xai_eval import pgi, pgu
import pickle
import os
import joblib
from joblib import Parallel, delayed
import pandas as pd

def convert_term_ranking_to_feature_ranking(ranking_with_interactions: pd.DataFrame) -> list:
    """
    Converts a ranking of terms (features + interactions) into a ranking of
    only features based on their earliest appearance.
    """
    ranking_with_interactions['rank'] = ranking_with_interactions['abs_ranking'].rank(method='min', ascending=False).astype(int)
    feature_to_best_rank = {}
    for i in range(len(ranking_with_interactions)):
        term = ranking_with_interactions.iloc[i]['features']
        idx = ranking_with_interactions.iloc[i]['rank']
        constituent_features = term.split(' x ')
        for feature in constituent_features:
            if feature not in feature_to_best_rank:
                feature_to_best_rank[feature] = idx
            else:
                feature_to_best_rank[feature] = min(feature_to_best_rank[feature], idx)

    return feature_to_best_rank


def compute_aggregated_ranking(ranking_per_fold_dict: dict, dataset_names, dataset_name, rank_type='mean_position'):
    model_dir = dataset_names[dataset_name][1]
    target = dataset_names[dataset_name][2]
    results_dir = dataset_names[dataset_name][0]
    pgis, pgus = [], []
    rankings_per_fold = []

    with open(os.path.join(results_dir, 'lime_results.pickle'), 'rb') as f:
        results = pickle.load(f)

    for i in range(len(ranking_per_fold_dict['lime'])):
        print(rank_type, i)
        model = os.path.join(model_dir, f'model_{i}.joblib')
        model = joblib.load(model)
        test_examples = results['test_data'][i].drop(columns=[target])
        train_examples = results['training_data'][i].drop(columns=[target])

        rankings = []
        for key in ranking_per_fold_dict.keys():
            ranking_current = ranking_per_fold_dict[key][i]
            ranking_current = convert_term_ranking_to_feature_ranking(ranking_current)
            rankings.append(ranking_current)

        if rank_type == 'rra':
            aggregated_ranking, _ = aggregate_rankings_by_rra(rankings)
        else:
            aggregated_ranking, _ = aggregate_rankings_by_mean_position(rankings)
        rankings_per_fold.append(aggregated_ranking)
        _, pgi_one = pgi(test_examples, aggregated_ranking, model, train_examples)
        _, pgu_one = pgu(test_examples, aggregated_ranking, model, train_examples)

        pgis.extend(pgi_one)
        pgus.extend(pgu_one)

    pgi_mean = np.mean(pgis)
    pgu_mean = np.mean(pgus)
    pgi_std = np.std(pgis)
    pgu_std = np.std(pgus)

    return f'aggregated_{rank_type}', {
        'pgi_mean': pgi_mean,
        'pgu_mean': pgu_mean,
        'pgi_std': pgi_std,
        'pgu_std': pgu_std,
    }, rankings_per_fold


def run_experiment_on_dataset(dataset_name, datasets_names, results_dict):
    """
    Runs a full experiment for a given dataset, calculating rankings and metrics
    for multiple XAI methods in parallel.
    """
    print('aaa')
    results_dir, model_dir, target = datasets_names[dataset_name]

    def process_method(key):
        """
        This is the function that will be parallelized. It processes ONE XAI method (key)
        and returns ALL of its calculated data.
        """
        # --- 1. Load data for the method ---
        file_name, ranking_func, *cf_info = results_dict[key]
        cf_func = cf_info[0] if cf_info else None

        with open(os.path.join(results_dir, file_name), 'rb') as f:
            results = pickle.load(f)

        # --- 2. Calculate Rankings and Counterfactuals ---
        ranking, rankings_per_fold = ranking_func(results, target)
        print(f"Ranking for {key} calculated.")

        cf_validity, cf_similarity, cf_similarity_std = None, None, None
        if cf_func:
            cf_validity, cf_similarity, cf_similarity_std = cf_func(results, target=target)

        # --- 3. Calculate PGI/PGU per fold ---
        pgis, pgus = [], []

        for i in range(len(rankings_per_fold)):
            print(key, i)
            model = joblib.load(os.path.join(model_dir, f'model_{i}.joblib'))
            test_examples = results['test_data'][i].drop(columns=[target])
            train_examples = results['training_data'][i].drop(columns=[target])
            ranking_current = list(rankings_per_fold[i]['features'])

            _, pgi_one = pgi(test_examples, ranking_current, model, train_examples)
            _, pgu_one = pgu(test_examples, ranking_current, model, train_examples)

            #print(pgi_one, pgu_one)

            pgis.extend(pgi_one)
            pgus.extend(pgu_one)

        if key in ["mmace", "meg"]:
            print(key, np.argsort(-np.array(pgus))[:10])

        # --- 4. Aggregate metrics and package all results for returning ---
        metrics = {
            'pgi_mean': np.mean(pgis),
            'pgu_mean': np.mean(pgus),
            'pgi_std': np.std(pgis),
            'pgu_std': np.std(pgus),
        }
        print(np.mean(pgis), np.mean(pgus))
        print(f"Finished processing: {key}")
        # Return everything needed from this run
        return key, {
            'ranking': ranking,
            'rankings_per_fold': rankings_per_fold,
            'cf_validity': cf_validity,
            'cf_similarity': cf_similarity,
            'cf_similarity_std': cf_similarity_std,
            'metrics': metrics
        }

    # --- Execute the parallel jobs ---
    # `results_per_method` will be a list of tuples, e.g., [('shap', {...}), ('lime', {...})]
    results_per_method = Parallel(n_jobs=6)(delayed(process_method)(key) for key in results_dict.keys())

    # --- Post-process the results in the main thread ---
    # Now, we safely populate the final dictionaries
    ranking_dict = {}
    ranking_per_fold_dict = {}
    cf_similarity_dict = {}
    cf_validity_dict = {}
    metrics_dict = {}

    for key, method_results in results_per_method:
        ranking_dict[key] = method_results['ranking']
        ranking_per_fold_dict[key] = method_results['rankings_per_fold']
        metrics_dict[key] = method_results['metrics']

        if method_results['cf_validity'] is not None:
            cf_validity_dict[key] = method_results['cf_validity']
            cf_similarity_dict[key] = (method_results['cf_similarity'], method_results['cf_similarity_std'])

    # You can now use these fully populated dictionaries
    return {
        'rankings': ranking_dict,
        'rankings_per_fold': ranking_per_fold_dict,
        'metrics': metrics_dict,
        'cf_validity': cf_validity_dict,
        'cf_similarity': cf_similarity_dict
    }


if __name__ == "__main__":

    datasets_names = {
        'cnohf_ecfp': ['../results/cnohf_data/cnohf_ecfp/explanations/', '../results/cnohf_data/cnohf_ecfp/', 'detonation_velocity'],
        'cof_ecfp_descriptor': ['../results/cof_data/cof_ecfp_descriptor/explanations/', '../results/cof_data/cof_ecfp_descriptor/', 'capacity_max'],
        'photoswitch_ecfp': ['../results/photoswitch_data/photoswitch_ecfp/explanations/', '../results/photoswitch_data/photoswitch_ecfp/', 'e_isomer_pi_pi'],
        'polymers_ecfp': ['../results/polymers_data/polymers_ecfp/explanations/', '../results/polymers_data/polymers_ecfp/', 'Tg'],
        'redox_ecfp': ['../results/redox_data/redox_ecfp/explanations/', '../results/redox_data/redox_ecfp/', 'dGox'],
        'herg_ecfp_linear': ['../results/synthetic_data/herg_ecfp_linear/explanations/', '../results/synthetic_data/herg_ecfp_linear/', 'target'],
        'herg_ecfp_nonlinear': ['../results/synthetic_data/herg_ecfp_nonlinear/explanations/', '../results/synthetic_data/herg_ecfp_nonlinear/', 'target'],
        'herg_ecfp_piecewise': ['../results/synthetic_data/herg_ecfp_piecewise/explanations/', '../results/synthetic_data/herg_ecfp_piecewise/', 'target'],
    }

    results_dict = {
        'lime': ('lime_results.pickle', lime_ranking),
        'shap': ('shap_results.pickle', shap_ranking),
        'shapiq1': ('shapiq1_results.pickle', shapiq_ranking),
        'shapiq2': ('shapiq2_results.pickle', shapiq_ranking),
        'meg': ('meg2_results.pickle', meg_ranking, meg_cf_percent),
        'mmace': ('mmace_results.pickle', mmace_ranking, mmace_cf_percent),
    }

    # Run the experiment for each dataset
    for dataset_name in datasets_names.keys():
        results_dir, model_dir, target = datasets_names[dataset_name]
        print(f"Processing dataset: {dataset_name}")
        results = run_experiment_on_dataset(dataset_name, datasets_names, results_dict)
        print(results['metrics'])

        metrics_dict = results['metrics']
        ranking_dict = results['rankings']
        ranking_per_fold_dict = results['rankings_per_fold']
        cf_validity_dict = results['cf_validity']
        cf_similarity_dict = results['cf_similarity']
        os.makedirs(os.path.join(results_dir, 'analysis_global2'), exist_ok=True)
        with open(os.path.join(results_dir, 'analysis_global2', 'metrics_results.pickle'), 'wb') as f:
            pickle.dump(metrics_dict, f)
        with open(os.path.join(results_dir, 'analysis_global2', 'ranking_results.pickle'), 'wb') as f:
            pickle.dump(ranking_dict, f)
        with open(os.path.join(results_dir, 'analysis_global2', 'ranking_per_fold_results.pickle'), 'wb') as f:
            pickle.dump(ranking_per_fold_dict, f)
        with open(os.path.join(results_dir, 'analysis_global2', 'cf_validity_results.pickle'), 'wb') as f:
            pickle.dump(cf_validity_dict, f)
        with open(os.path.join(results_dir, 'analysis_global2', 'cf_similarity_results.pickle'), 'wb') as f:
            pickle.dump(cf_similarity_dict, f)

        aggregated_ranking_rra, agg_metrics_rra, agg_rankings_per_fold_rra = compute_aggregated_ranking(
            results['rankings_per_fold'], datasets_names, dataset_name, rank_type='rra')
        print(f"Aggregated ranking for {dataset_name}: {aggregated_ranking_rra}")
        aggregated_ranking, agg_metrics, agg_rankings_per_fold = compute_aggregated_ranking(
            results['rankings_per_fold'], datasets_names, dataset_name, rank_type='mean')
        print(f"Aggregated ranking for {dataset_name}: {aggregated_ranking}")

        results['metrics'][aggregated_ranking] = agg_metrics
        results['rankings_per_fold'][aggregated_ranking] = agg_rankings_per_fold
        results['metrics'][aggregated_ranking_rra] = agg_metrics_rra
        results['rankings_per_fold'][aggregated_ranking_rra] = agg_rankings_per_fold_rra

        metrics_dict = results['metrics']
        ranking_dict = results['rankings']
        ranking_per_fold_dict = results['rankings_per_fold']
        cf_validity_dict = results['cf_validity']
        cf_similarity_dict = results['cf_similarity']

        os.makedirs(os.path.join(results_dir, 'analysis_global2'), exist_ok=True)
        with open(os.path.join(results_dir, 'analysis_global2', 'metrics_results.pickle'), 'wb') as f:
            pickle.dump(metrics_dict, f)
        with open(os.path.join(results_dir, 'analysis_global2', 'ranking_results.pickle'), 'wb') as f:
            pickle.dump(ranking_dict, f)
        with open(os.path.join(results_dir, 'analysis_global2', 'ranking_per_fold_results.pickle'), 'wb') as f:
            pickle.dump(ranking_per_fold_dict, f)
        with open(os.path.join(results_dir, 'analysis_global2', 'cf_validity_results.pickle'), 'wb') as f:
            pickle.dump(cf_validity_dict, f)
        with open(os.path.join(results_dir, 'analysis_global2', 'cf_similarity_results.pickle'), 'wb') as f:
            pickle.dump(cf_similarity_dict, f)

        print('the end')