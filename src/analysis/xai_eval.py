from itertools import combinations

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import auc


def perturb(examples, features, training_examples):

    features_flat = [f for sublist in features for f in sublist]
    continuous_features = [f for f in features_flat if 'ecfp' not in f]
    continuous_features = list(set(continuous_features))
    discrete_features = [f for f in features_flat if 'ecfp' in f]
    discrete_features = list(set(discrete_features))
    continuous_stds = {f: training_examples[f].std() for f in continuous_features}
    discrete_features_probs = {f: training_examples[f].value_counts(normalize=True) for f in discrete_features}

    perturbed_examples = examples.copy()
    for feature in continuous_features:
        np.random.seed(42)
        std = continuous_stds[feature]
        perturbation = np.random.normal(0, std, size=examples.shape[0])
        perturbed_examples[feature] = examples[feature] + perturbation

    for feature in discrete_features:
        probs = discrete_features_probs[feature]
        new_values = []
        for original_value in examples[feature]:
            np.random.seed(42)
            temp_probs = probs.drop(original_value, errors='ignore')
            renormalized_probs = temp_probs / temp_probs.sum()
            new_value = np.random.choice(
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
    results = []
    old_preds = model.predict(examples.to_numpy())
    num_unique_features = [f.split('x') for f in ranking]
    num_unique_features = list(set([f.replace(' ', '') for sublist in num_unique_features for f in sublist]))
    non_present = [f for f in training_examples.columns if f not in num_unique_features]
    calculate_non = False
    if len_max is None:
        calculate_non = True
        len_max = len(num_unique_features)

    max_diff = calculate_max_diff(examples, model, training_examples, old_preds)
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
        perturbed_examples = perturb(examples, top_k, training_examples)
        new_preds = model.predict(perturbed_examples.to_numpy())
        diff = np.abs(new_preds - old_preds).mean()
        for _ in range(len(top_k[-1])):
            results.append(diff)

    # account for features not in explanation - all ex equo get last place
    if calculate_non:
        features = [[p] for p in num_unique_features] + [[f] for f in non_present]
        perturbed_examples = perturb(examples, features, training_examples)
        new_preds = model.predict(perturbed_examples.to_numpy())
        diff = np.abs(new_preds - old_preds).mean()
        for _ in non_present:
            results.append(diff)
    auc_results = auc(np.arange(len(results)) / (len(results) - 1), results)
    auc_max = auc(np.arange(len(results)) / (len(results) - 1), [max_diff] * len(results))
    return auc_results / auc_max, auc_results


def pgu(examples, ranking, model, training_examples, len_max=None):
    results = []
    old_preds = model.predict(examples.to_numpy())
    num_unique_features = [f.split('x') for f in ranking]
    num_unique_features = list(set([f.replace(' ', '') for sublist in num_unique_features for f in sublist]))
    non_present = [f for f in training_examples.columns if f not in num_unique_features]
    if len_max is None:
        len_max = len(num_unique_features) + len(non_present)
    else:
        len_max = len(num_unique_features) + len(non_present) - len_max

    max_diff = calculate_max_diff(examples, model, training_examples, old_preds)
    current_len = 0

    features = [[f] for f in non_present]
    perturbed_examples = perturb(examples, features, training_examples)
    new_preds = model.predict(perturbed_examples.to_numpy())
    diff = np.abs(new_preds - old_preds).mean()
    for _ in non_present:
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
        perturbed_examples = perturb(examples, top_k + features, training_examples)
        new_preds = model.predict(perturbed_examples.to_numpy())
        diff = np.abs(new_preds - old_preds).mean()
        results.append(diff)
        for _ in range(len(top_k[-1])):
            results.append(diff)

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


def rank_agreement(ranking1, ranking2):
    percent = []
    for k in range(1, len(ranking1) + 1):
        top_k1 = np.array(ranking1[:k])
        top_k2 = np.array(ranking2[:k])
        metric = np.sum(top_k1 == top_k2) / k
        percent.append(metric)
    auc_results = auc(np.arange(len(percent)) / (len(percent) - 1), percent)
    return auc_results


def rank_correlation(ranking1, ranking2, k=None):
    if k is not None:
        ranking1 = ranking1[:k]
        ranking2 = ranking2[:k]
    if len(ranking1) != len(ranking2):
        shorter = min(len(ranking1), len(ranking2))
        ranking1 = ranking1[:shorter]
        ranking2 = ranking2[:shorter]

    all_items = set(ranking1) | set(ranking2)
    rank_map1 = {item: i for i, item in enumerate(ranking1)}
    rank_map2 = {item: i for i, item in enumerate(ranking2)}
    ranks1 = [rank_map1.get(item, len(ranking1)) for item in all_items]
    ranks2 = [rank_map2.get(item, len(ranking2)) for item in all_items]
    correlation = spearmanr(ranks1, ranks2)
    return correlation


def pairwise_agreement(ranking1, ranking2, k=None):
    if k is not None:
        ranking1 = ranking1[:k]
        ranking2 = ranking2[:k]

    common_items = set(ranking1) & set(ranking2)
    common_items_list = list(common_items)

    if len(common_items_list) < 2:
        return 0.0

    rank_map1 = {item: i for i, item in enumerate(ranking1)}
    rank_map2 = {item: i for i, item in enumerate(ranking2)}

    agreements = 0

    for item1, item2 in combinations(common_items_list, 2):
        rank1_item1 = rank_map1[item1]
        rank1_item2 = rank_map1[item2]
        rank2_item1 = rank_map2[item1]
        rank2_item2 = rank_map2[item2]

        is_item1_higher_in_rank1 = rank1_item1 < rank1_item2
        is_item1_higher_in_rank2 = rank2_item1 < rank2_item2
        agreements += (is_item1_higher_in_rank1 == is_item1_higher_in_rank2)

    total_pairs = sum(1 for _ in combinations(common_items_list, 2))
    return agreements / total_pairs
