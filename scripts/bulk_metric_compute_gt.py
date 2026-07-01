import sys

import numpy as np

from src.ranking.mean import aggregate_rankings_by_mean_position
from src.ranking.rra import aggregate_rankings_by_rra
from scripts.bulk_explain_gt import GtModel
from src.analysis.processing import lime_ranking, shap_ranking, shapiq_ranking, meg_ranking, mmace_ranking, \
    meg_cf_percent, mmace_cf_percent
from src.analysis.xai_eval import pgi, pgu, feature_agreement
import pickle
import os
import joblib
from joblib import Parallel, delayed
import pandas as pd


def compute_reference_list(rankings_per_instance_dict):
    all_methods_lists = list(rankings_per_instance_dict.values())
    if not all_methods_lists:
        return []
    reference_list = []
    for i in range(len(rankings_per_instance_dict['lime'])):
        fold = []
        for j in range(len(rankings_per_instance_dict['lime'][i])):
            value = 1
            for key in rankings_per_instance_dict:
                if rankings_per_instance_dict[key][i][j] is None:
                    value = None
            fold.append(value)
        reference_list.append(fold)
    return reference_list


def convert_term_ranking_to_feature_ranking(ranking_with_interactions: pd.DataFrame) -> list:
    """
    Converts a ranking of terms (features + interactions) into a ranking of
    only features based on their earliest appearance.
    """
    ranking_with_interactions['rank'] = ranking_with_interactions['abs_ranking'].rank(method='min',
                                                                                      ascending=False).astype(int)
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


def compute_aggregated_ranking(ranking_per_fold_dict: dict, ranking_per_instance_dict: dict, dataset_names,
                               dataset_name, rank_type='mean_position'):
    model_dir = dataset_names[dataset_name][1]
    target = dataset_names[dataset_name][2]
    results_dir = dataset_names[dataset_name][0]
    rankings_per_instance = []
    reference_list = []

    with open(os.path.join(results_dir, 'lime_results.pickle'), 'rb') as f:
        results = pickle.load(f)

    for i in range(len(ranking_per_fold_dict['lime'])):
        print(f'Aggregation for fold: {i}')
        rankings_per_fold = []
        for j in range(len(ranking_per_instance_dict['lime'][i])):
            rankings = []
            for key in ranking_per_instance_dict.keys():
                ranking_instance = ranking_per_instance_dict[key][i][j]
                if ranking_instance is not None:
                    ranking_instance = convert_term_ranking_to_feature_ranking(ranking_instance)
                    rankings.append(ranking_instance)
            if rank_type == 'rra':
                aggregated_ranking, _ = aggregate_rankings_by_rra(rankings)
            else:
                aggregated_ranking, _ = aggregate_rankings_by_mean_position(rankings)
            aggregated_ranking = {'features': aggregated_ranking}
            rankings_per_fold.append(aggregated_ranking)
        rankings_per_instance.append(rankings_per_fold)

    metrics_no_reference_list = compute_metrics_from_rankings(f'aggregated_{rank_type}', rankings_per_instance,
                                                              rankings_per_instance, results, model_dir=model_dir,
                                                              target=target, ref_list=None)

    return f'aggregated_{rank_type}', metrics_no_reference_list, rankings_per_instance


def compute_metrics_from_rankings(key, method_instance_rankings, method_fold_rankings, results, target, model_dir,
                                  ref_list=None, interactions=''):
    pgis_ratio, pgus_ratio, pgis_max = [], [], []
    pgis_raw, pgus_raw, pgus_max = [], [], []
    fas = []
    fas_interaction = []

    feature_name = 'ecfp_feature'
    important_features = [726, 456, 893, 428]
    important_features_simple = [f'{feature_name}_{i}' for i in important_features]
    important_features_piecewise_interactions = [f'{feature_name}_{i}' for i in important_features] + [
        f'{feature_name}_{j} x {feature_name}_893' for j in important_features if j != 893]
    important_feature_nonlinear_interactions = [f'{feature_name}_{i}' for i in important_features] + [
        f'{feature_name}_428 x {feature_name}_726', f'{feature_name}_456 x {feature_name}_726']
    choices = {
        'linear': important_features_simple,
        'piecewise': important_features_piecewise_interactions,
        'nonlinear': important_feature_nonlinear_interactions
    }

    for i in range(len(method_fold_rankings)):
        if ref_list is not None:
            ref_l = ref_list[i]
        else:
            ref_l = None
        print(key, i)
        sys.modules['__main__'].GtModel = GtModel
        model = joblib.load(os.path.join(model_dir, f'model_{i}.joblib'))
        test_examples = results['test_data'][i].drop(columns=[target])
        train_examples = results['training_data'][i].drop(columns=[target])
        rankings_current = list(method_instance_rankings[i])

        pgi_ratios, pgi_raw, pgi_max = pgi(test_examples, rankings_current, model, train_examples, reference_list=ref_l)
        pgu_ratios, pgu_raw, pgu_max = pgu(test_examples, rankings_current, model, train_examples, reference_list=ref_l)

        pgis_ratio.extend(pgi_ratios)
        pgus_ratio.extend(pgu_ratios)
        pgis_raw.extend(pgi_raw)
        pgus_raw.extend(pgu_raw)
        pgis_max.extend(pgi_max)
        pgus_max.extend(pgu_max)

        fa = feature_agreement(choices['linear'], rankings_current, list(test_examples.columns), remove_inter=True)
        fas.extend(fa)

        if len(interactions):
            fa_int = feature_agreement(choices[interactions], rankings_current, list(test_examples.columns), remove_inter=False)
            fas_interaction.extend(fa_int)

    # --- 4. Aggregate metrics and package all results for returning ---
    metrics = {
        'pgis_raw': pgis_raw,
        'pgus_raw': pgus_raw,
        'pgis_ratio': pgis_ratio,
        'pgus_ratio': pgus_ratio,
        'pgis_max': pgis_max,
        'pgus_max': pgus_max,
        'fa': fas,
        'fa_interaction': fas_interaction
    }
    return key, metrics


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

        sys.modules['__main__'].GtModel = GtModel
        with open(os.path.join(results_dir, file_name), 'rb') as f:
            results = pickle.load(f)

        # --- 2. Calculate Rankings and Counterfactuals ---
        ranking, rankings_per_fold, rankings_per_instance = ranking_func(results, target)
        print(f"Ranking for {key} calculated.")

        cf_validity, cf_similarity, cf_similarity_std = None, None, None
        if cf_func:
            cf_validity, cf_similarity, cf_similarity_std = cf_func(results, target=target)

        if key == 'shapiq2':
            interactions = dataset_name
        else:
            interactions = ''
        metrics = compute_metrics_from_rankings(key, rankings_per_instance, rankings_per_fold, results, target=target,
                                                model_dir=model_dir, interactions=interactions)
        print(metrics)
        print(f"Finished processing: {key}")
        # Return everything needed from this run
        return key, {
            'ranking': ranking,
            'rankings_per_fold': rankings_per_fold,
            'rankings_per_instance': rankings_per_instance,
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
    ranking_per_instance_dict = {}
    cf_similarity_dict = {}
    cf_validity_dict = {}
    metrics_dict = {}
    results_org = {}

    for key, method_results in results_per_method:
        ranking_dict[key] = method_results['ranking']
        ranking_per_fold_dict[key] = method_results['rankings_per_fold']
        ranking_per_instance_dict[key] = method_results['rankings_per_instance']
        metrics_dict[key] = method_results['metrics']

        if method_results['cf_validity'] is not None:
            cf_validity_dict[key] = method_results['cf_validity']
            cf_similarity_dict[key] = (method_results['cf_similarity'], method_results['cf_similarity_std'])

        file_name, _, *_ = results_dict[key]
        with open(os.path.join(results_dir, file_name), 'rb') as f:
            r = pickle.load(f)
        results_org[key] = r

    reference_list = compute_reference_list(ranking_per_instance_dict)

    return {
        'rankings': ranking_dict,
        'rankings_per_fold': ranking_per_fold_dict,
        'rankings_per_instance': ranking_per_instance_dict,
        'metrics': metrics_dict,
        'reference_list': reference_list,
        'cf_validity': cf_validity_dict,
        'cf_similarity': cf_similarity_dict
    }

if __name__ == "__main__":

    datasets_names = {
        'linear': [f'../results/gt_synthetic_data/herg_ecfp_linear/explanations/',
                   '../results/gt_synthetic_data/herg_ecfp_linear/', 'target'],
        'piecewise': [f'../results/gt_synthetic_data/herg_ecfp_piecewise/explanations/',
                      '../results/gt_synthetic_data/herg_ecfp_piecewise/', 'target'],
        'nonlinear': [f'../results/gt_synthetic_data/herg_ecfp_nonlinear/explanations/',
                      '../results/gt_synthetic_data/herg_ecfp_nonlinear/', 'target'],
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

        analysis_output_dir = 'analysis_local'

        metrics_dict = results['metrics']
        ranking_dict = results['rankings']
        ranking_per_fold_dict = results['rankings_per_fold']
        reference_list = results['reference_list']
        cf_validity_dict = results['cf_validity']
        cf_similarity_dict = results['cf_similarity']
        os.makedirs(os.path.join(results_dir, analysis_output_dir), exist_ok=True)
        with open(os.path.join(results_dir, analysis_output_dir, 'metrics_results.pickle'), 'wb') as f:
            pickle.dump(metrics_dict, f)
        with open(os.path.join(results_dir, analysis_output_dir, 'ranking_results.pickle'), 'wb') as f:
            pickle.dump(ranking_dict, f)
        with open(os.path.join(results_dir, analysis_output_dir, 'ranking_per_fold_results.pickle'), 'wb') as f:
            pickle.dump(ranking_per_fold_dict, f)
        with open(os.path.join(results_dir, analysis_output_dir, 'cf_validity_results.pickle'), 'wb') as f:
            pickle.dump(cf_validity_dict, f)
        with open(os.path.join(results_dir, analysis_output_dir, 'cf_similarity_results.pickle'), 'wb') as f:
            pickle.dump(cf_similarity_dict, f)
        with open(os.path.join(results_dir, analysis_output_dir, 'reference_lists.pickle'), 'wb') as f:
            pickle.dump(reference_list, f)


        aggregated_ranking_rra, agg_metrics_rra,  agg_rankings_per_fold_rra = compute_aggregated_ranking(
            results['rankings_per_fold'], results['rankings_per_instance'], datasets_names, dataset_name, rank_type='rra')
        print(f"Aggregated ranking for {dataset_name}: {aggregated_ranking_rra}")
        aggregated_ranking, agg_metrics,  agg_rankings_per_fold = compute_aggregated_ranking(
            results['rankings_per_fold'], results['rankings_per_instance'], datasets_names, dataset_name, rank_type='mean')
        print(f"Aggregated ranking for {dataset_name}: {aggregated_ranking}")

        results['metrics'][aggregated_ranking] = agg_metrics
        results['rankings_per_instance'][aggregated_ranking] = agg_rankings_per_fold
        results['metrics'][aggregated_ranking_rra] = agg_metrics_rra
        results['rankings_per_instance'][aggregated_ranking_rra] = agg_rankings_per_fold_rra

        metrics_dict = results['metrics']
        ranking_dict = results['rankings']
        ranking_per_fold_dict = results['rankings_per_fold']
        cf_validity_dict = results['cf_validity']
        cf_similarity_dict = results['cf_similarity']
        ranking_per_instance_dict = results['rankings_per_instance']

        os.makedirs(os.path.join(results_dir, analysis_output_dir), exist_ok=True)
        with open(os.path.join(results_dir, analysis_output_dir, 'metrics_results.pickle'), 'wb') as f:
            pickle.dump(metrics_dict, f)
        with open(os.path.join(results_dir, analysis_output_dir, 'ranking_results.pickle'), 'wb') as f:
            pickle.dump(ranking_dict, f)
        with open(os.path.join(results_dir, analysis_output_dir, 'ranking_per_fold_results.pickle'), 'wb') as f:
            pickle.dump(ranking_per_fold_dict, f)
        with open(os.path.join(results_dir, analysis_output_dir, 'cf_validity_results.pickle'), 'wb') as f:
            pickle.dump(cf_validity_dict, f)
        with open(os.path.join(results_dir, analysis_output_dir, 'cf_similarity_results.pickle'), 'wb') as f:
            pickle.dump(cf_similarity_dict, f)
        with open(os.path.join(results_dir, analysis_output_dir, 'reference_lists.pickle'), 'wb') as f:
            pickle.dump(reference_list, f)
        with open(os.path.join(results_dir, analysis_output_dir, 'rankings_per_instance_results.pickle'), 'wb') as f:
            pickle.dump(ranking_per_instance_dict, f)

