import copy
from itertools import combinations

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import auc
from joblib import Parallel, delayed


def perturb(examples, features, training_examples, seed=42):
    rng = np.random.default_rng(seed)

    features_flat = [f for sublist in features for f in sublist]
    continuous_features = [f for f in features_flat if 'ecfp' not in f]
    continuous_features = list(set(continuous_features))
    discrete_features = [f for f in features_flat if 'ecfp' in f]
    discrete_features = list(set(discrete_features))
    continuous_stds = {f: training_examples[f].std() for f in continuous_features}
    discrete_features_probs = {f: training_examples[f].value_counts(normalize=True) for f in discrete_features}

    perturbed_examples = copy.deepcopy(examples)
    for feature in continuous_features:
        std = continuous_stds[feature]
        perturbation = rng.normal(0, std, size=examples.shape[0])
        perturbed_examples[feature] = examples[feature] + perturbation

    for feature in discrete_features:
        probs = discrete_features_probs[feature]
        new_values = []
        for original_value in examples[feature]:
            temp_probs = probs.drop(original_value, errors='ignore')
            renormalized_probs = temp_probs / temp_probs.sum()
            new_value = rng.choice(
                renormalized_probs.index,
                size=1,
                p=renormalized_probs.values,
            )[0]
            new_values.append(new_value)
        perturbed_examples[feature] = new_values

    return perturbed_examples


def calculate_max_diff(examples, model, training_examples, old_preds):
    all_features = [[f] for f in training_examples.columns]
    perturbed_all = perturb(examples, all_features, training_examples)
    preds_all_perturbed = model.predict(perturbed_all.to_numpy())
    max_diff = np.abs(preds_all_perturbed - old_preds).max()
    return max_diff


def pgi(examples, ranking, model, training_examples, len_max=None):

    old_preds = model.predict(examples.to_numpy())
    num_unique_features = [f.split('x') for f in ranking]
    num_unique_features = list(set([f.replace(' ', '') for sublist in num_unique_features for f in sublist]))
    non_present = [f for f in training_examples.columns if f not in num_unique_features]
    calculate_non = False
    if len_max is None:
        calculate_non = True
        len_max = len(num_unique_features)

    max_diff = calculate_max_diff(examples, model, training_examples, old_preds)

    def compute_results(i):
        results = []
        current_len = 0
        for k in range(1, len(ranking) + 1):
            top_k = ranking[:k]
            top_k = [f.split(' x ') for f in top_k]
            topk_flat = list(set([f.replace(' ', '') for sublist in top_k for f in sublist]))
            if len(topk_flat) >= len_max:
                break
            if current_len == len(topk_flat):
                continue
            current_len = len(topk_flat)
            perturbed_examples = perturb(examples, top_k, training_examples, seed=i)
            new_preds = model.predict(perturbed_examples.to_numpy())
            diff = np.abs(new_preds - old_preds).mean()
            for _ in range(len(top_k[-1])):
                results.append(diff)

        # account for features not in explanation - all ex equo get last place
        if calculate_non:
            features = [[f] for f in non_present]
            for j in range(1, len(num_unique_features) + 1):
                perturbed_examples = perturb(examples, [[p] for p in num_unique_features] + features[:j], training_examples, seed=i)
                new_preds = model.predict(perturbed_examples.to_numpy())
                diff = np.abs(new_preds - old_preds).mean()
                results.append(diff)

        return results

    results = Parallel(n_jobs=25)(delayed(compute_results)(i) for i in range(25))
    results = np.array(results)
    results = np.mean(results, axis=0)

    auc_results = auc(np.arange(len(results)) / (len(results) - 1), results)
    auc_max = auc(np.arange(len(results)) / (len(results) - 1), [max_diff] * len(results))
    return auc_results / auc_max, auc_results


def pgu(examples, ranking, model, training_examples, len_max=None):
    old_preds = model.predict(examples.to_numpy())
    num_unique_features = [f.split('x') for f in ranking]
    num_unique_features = list(set([f.replace(' ', '') for sublist in num_unique_features for f in sublist]))
    non_present = [f for f in training_examples.columns if f not in num_unique_features]
    if len_max is None:
        len_max = len(num_unique_features) + len(non_present)
    else:
        len_max = len(num_unique_features) + len(non_present) - len_max

    max_diff = calculate_max_diff(examples, model, training_examples, old_preds)

    def compute_results(i):
        results = []
        current_len = 0
        features = [[f] for f in non_present]
        for j in range(1, len(num_unique_features)+1):
            perturbed_examples = perturb(examples, features[:j], training_examples, seed=i)
            new_preds = model.predict(perturbed_examples.to_numpy())
            diff = np.abs(new_preds - old_preds).mean()
            results.append(diff)
        for k in range(1, len(ranking) + 1):
            top_k = ranking[-k:]
            top_k = [f.split(' x ') for f in top_k]
            topk_flat = list(set([f.replace(' ', '') for sublist in top_k for f in sublist]))
            if len(topk_flat) + len(non_present) >= len_max:
                break
            if current_len == len(topk_flat):
                continue
            current_len = len(topk_flat)
            perturbed_examples = perturb(examples, top_k + features, training_examples, seed=i)
            new_preds = model.predict(perturbed_examples.to_numpy())
            diff = np.abs(new_preds - old_preds).mean()
            results.append(diff)
            for _ in range(len(top_k[-1])):
                results.append(diff)
        return results

    results = Parallel(n_jobs=25)(delayed(compute_results)(i) for i in range(25))
    results = np.array(results)
    results = np.mean(results, axis=0)

    auc_results = auc(np.arange(len(results)) / (len(results) - 1), results[::-1])
    auc_max = auc(np.arange(len(results)) / (len(results) - 1), [max_diff] * len(results))
    return auc_results / auc_max, auc_results


def feature_agreement(gt, ranking2):
    percent = []
    for k in range(len(gt) + 1, len(ranking2) + 1):
        top_k2 = ranking2[:k]
        metric = len(set(gt).intersection(set(top_k2))) / len(gt)
        percent.append(metric)
    not_in_ranking = len(set(gt) - set(ranking2))
    for _ in range(not_in_ranking):
        percent.append(0.0)
    auc_results = auc(np.arange(len(percent)) / (len(percent) - 1), percent)
    return auc_results


def rank_correlation(ranking1: dict, ranking2: dict, k=None):
    r1 = {feature: rank for feature, rank in ranking1.items() if rank <= k} if k is not None else ranking1
    r2 = {feature: rank for feature, rank in ranking2.items() if rank <= k} if k is not None else ranking2

    all_features = set(r1.keys()) | set(r2.keys())

    penalty1 = (max(r1.values()) + 1)
    penalty2 = (max(r2.values()) + 1)

    ranks1 = [ranking1.get(feature, penalty1) for feature in all_features]
    ranks2 = [ranking2.get(feature, penalty2) for feature in all_features]

    correlation = spearmanr(ranks1, ranks2)
    return correlation

