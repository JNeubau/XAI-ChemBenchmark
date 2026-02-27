import numpy as np
import pandas as pd
from shapiq.interaction_values import InteractionValues, aggregate_interaction_values
from shapiq.plot.utils import format_labels


def shap_ranking(shap_results, target):
    abs_ranking = []
    features_names = shap_results['test_data'][0].drop(columns=[target]).columns
    shap_values = shap_results['shap_values']
    ranking_per_fold = []
    for i in range(len(shap_values)):
        fold_ranking = []
        for sv in shap_values[i]:
            abs_ranking.append(np.abs(sv))
            fold_ranking.append(np.abs(sv))

        fold_ranking = np.array(fold_ranking)
        fold_ranking = fold_ranking.mean(axis=0)
        fold_rank = pd.DataFrame({'features': features_names, 'abs_ranking': fold_ranking})
        fold_rank = fold_rank.sort_values(by='abs_ranking', ascending=False).reset_index(drop=True)
        ranking_per_fold.append(fold_rank)

    abs_ranking = np.array(abs_ranking)
    abs_ranking = abs_ranking.mean(axis=0)
    abs_ranking = pd.DataFrame({'features': features_names, 'abs_ranking': abs_ranking})
    abs_ranking = abs_ranking.sort_values(by='abs_ranking', ascending=False).reset_index(drop=True)
    return abs_ranking, ranking_per_fold


def shapiq_ranking(shapiq_results, target):
    features_names = shapiq_results['test_data'][0].drop(columns=[target]).columns
    feature_mapping = dict(enumerate(features_names))
    shapiq_values = shapiq_results['interactions']
    ranking_per_fold = []
    interaction_values = []
    for i in range(len(shapiq_values)):
        fold_iv = []
        for iv in shapiq_values[i]:
            fold_iv.append(InteractionValues.from_dict(iv))
            interaction_values.append(InteractionValues.from_dict(iv))

        fold_iv = [abs(iv) for iv in fold_iv]
        fold_ranking = aggregate_interaction_values(fold_iv, "mean")
        interaction_list = fold_ranking.interaction_lookup.keys()
        feature_labels = [format_labels(feature_tuple=iv, feature_mapping=feature_mapping) for iv in interaction_list]
        fold_ranking = pd.DataFrame({'features': feature_labels, 'abs_ranking': fold_ranking.values})
        fold_ranking = fold_ranking[~fold_ranking['features'].str.contains('Base Value')]
        fold_ranking = fold_ranking.sort_values(by='abs_ranking', ascending=False). reset_index(drop=True)
        ranking_per_fold.append(fold_ranking)

    interaction_values = [abs(iv) for iv in interaction_values]
    abs_ranking = aggregate_interaction_values(interaction_values, "mean")
    interaction_list = abs_ranking.interaction_lookup.keys()
    feature_labels = [format_labels(feature_tuple=iv, feature_mapping=feature_mapping) for iv in interaction_list]
    abs_ranking = pd.DataFrame({'features': feature_labels, 'abs_ranking': abs_ranking.values})
    abs_ranking = abs_ranking[~abs_ranking['features'].str.contains('Base Value')]
    abs_ranking = abs_ranking.sort_values(by='abs_ranking', ascending=False).reset_index(drop=True)
    return abs_ranking, ranking_per_fold


def lime_ranking(lime_results, target):
    feature_names = lime_results['test_data'][0].drop(columns=[target]).columns
    abs_ranking = {f: [] for f in feature_names}
    ranking_per_fold = []
    lime_values = lime_results['lime_values']
    for i in range(len(lime_values)):
        fold_ranking = {f: [] for f in feature_names}
        for lv in lime_values[i]:
            for entry in lv:
                abs_ranking[entry[0]].append(np.abs(entry[1]))
                fold_ranking[entry[0]].append(np.abs(entry[1]))

        fold_ranking = {f: np.mean(v) for f, v in fold_ranking.items()}
        fold_rank = pd.DataFrame({'features': list(fold_ranking.keys()), 'abs_ranking': list(fold_ranking.values())})
        fold_rank = fold_rank.sort_values(by='abs_ranking', ascending=False).reset_index(drop=True)
        ranking_per_fold.append(fold_rank)

    abs_ranking = {f: np.mean(v) for f, v in abs_ranking.items()}
    abs_ranking = pd.DataFrame({'features': list(abs_ranking.keys()), 'abs_ranking': list(abs_ranking.values())})
    abs_ranking = abs_ranking.sort_values(by='abs_ranking', ascending=False).reset_index(drop=True)
    return abs_ranking, ranking_per_fold


def meg_cf_percent(meg_results, target):
    counterfactual_reward = meg_results['counterfactuals_pred_reward']
    counterfactuals = 0
    all = 0
    similarities = []
    for i in range(len(counterfactual_reward)):
        cf_r = np.array([item for sublist in counterfactual_reward[i] for item in sublist])
        all += 25 * len(counterfactual_reward[i])
        real_cfs = np.sum(cf_r >= 1)
        sim = np.array([item for sublist in meg_results['counterfactuals_similarity'][i] for item in sublist])
        sim = sim[cf_r >= 1]
        similarities += list(sim)
        counterfactuals += real_cfs
    return counterfactuals / all, np.mean(similarities), np.std(similarities)


def meg_ranking(meg_results, target):
    feature_names = meg_results['test_data'][0].drop(columns=[target]).columns
    abs_ranking = {f: [] for f in feature_names}

    ranking_per_fold = []

    for i in range(len(meg_results['test_data'])):
        fold_ranking = {f: [] for f in feature_names}

        for j in range(len(meg_results['test_data'][i])):
            example = meg_results['test_data'][i].iloc[j].drop(columns=[target]).values
            cfs = meg_results['counterfactuals_encoding'][i][j]
            for n in range(len(cfs)):
                cf = cfs[n].flatten()
                is_cf = meg_results['counterfactuals_pred_reward'][i][j][n] >= 1
                if is_cf:
                    for k in range(len(feature_names)):
                        if example[k] != cf[k]:
                            abs_ranking[feature_names[k]].append(1)
                            fold_ranking[feature_names[k]].append(1)
                        else:
                            abs_ranking[feature_names[k]].append(0)
                            fold_ranking[feature_names[k]].append(0)

        fold_ranking = {f: np.mean(v) for f, v in fold_ranking.items()}
        fold_rank = pd.DataFrame({'features': list(fold_ranking.keys()), 'abs_ranking': list(fold_ranking.values())})
        fold_rank = fold_rank.sort_values(by='abs_ranking', ascending=False).reset_index(drop=True)
        ranking_per_fold.append(fold_rank)

    abs_ranking = {f: np.mean(v) for f, v in abs_ranking.items()}
    abs_ranking = pd.DataFrame({'features': list(abs_ranking.keys()), 'abs_ranking': list(abs_ranking.values())})
    abs_ranking = abs_ranking.sort_values(by='abs_ranking', ascending=False).reset_index(drop=True)
    return abs_ranking, ranking_per_fold


def mmace_cf_percent(mmace_results, target, min_diff=0):
    all = 0
    counterfactuals = 0
    similarities = []
    for i in range(len(mmace_results['test_data'])):
        train_data_median = mmace_results['training_data'][i][target].median()
        pred_original = np.array(mmace_results['pred_original'][i])
        pred_counterfactuals = mmace_results['pred_counterfactual'][i]
        for j in range(len(pred_original)):
            pred_counterfactuals[j] = np.array(pred_counterfactuals[j])
            all += 25
            for k in range(len(pred_counterfactuals[j])):
                if pred_original[j] > train_data_median > pred_counterfactuals[j][k] and np.abs(
                        pred_original[j] - pred_counterfactuals[j][k]) >= min_diff:
                    counterfactuals += 1
                    similarities.append(mmace_results['counterfactuals_similarity'][i][j][k])
                elif pred_original[j] < train_data_median < pred_counterfactuals[j][k] and np.abs(
                        pred_original[j] - pred_counterfactuals[j][k]) >= min_diff:
                    counterfactuals += 1
    return counterfactuals / all, np.mean(similarities), np.std(similarities)


def mmace_ranking(mmace_results, target):
    feature_names = mmace_results['test_data'][0].drop(columns=[target]).columns
    abs_ranking = {f: [] for f in feature_names}
    ranking_per_fold = []

    for i in range(len(mmace_results['test_data'])):
        fold_ranking = {f: [] for f in feature_names}

        for j in range(len(mmace_results['test_data'][i])):
            example = mmace_results['test_data'][i].iloc[j].drop(columns=[target]).values
            cfs = mmace_results['counterfactuals_encoding'][i][j]
            for n in range(len(cfs)):
                cf = cfs[n].flatten()
                for k in range(len(feature_names)):
                    if example[k] != cf[k]:
                        abs_ranking[feature_names[k]].append(1)
                        fold_ranking[feature_names[k]].append(1)
                    else:
                        abs_ranking[feature_names[k]].append(0)
                        fold_ranking[feature_names[k]].append(0)
        fold_ranking = {f: np.mean(v) for f, v in fold_ranking.items()}
        fold_rank = pd.DataFrame({'features': list(fold_ranking.keys()), 'abs_ranking': list(fold_ranking.values())})
        fold_rank = fold_rank.sort_values(by='abs_ranking', ascending=False).reset_index(drop=True)
        ranking_per_fold.append(fold_rank)

    abs_ranking = {f: np.mean(v) for f, v in abs_ranking.items()}
    abs_ranking = pd.DataFrame({'features': list(abs_ranking.keys()), 'abs_ranking': list(abs_ranking.values())})
    abs_ranking = abs_ranking.sort_values(by='abs_ranking', ascending=False).reset_index(drop=True)
    return abs_ranking, ranking_per_fold
