import numpy as np
from scipy.stats import spearmanr, skellam, weightedtau
from sklearn.metrics import auc
from joblib import Parallel, delayed


def get_dataset_stats(training_examples):
    """Calculate once per fold, not once per perturbation."""
    continuous = ['mol_wt', 'o%', 'n%', 'c%']
    # Filter to only what exists in the data
    existing_cont = [f for f in continuous if f in training_examples.columns]
    existing_count = [f for f in training_examples.columns if f not in existing_cont]
    all_cols = {f: i for i, f in enumerate(training_examples.columns)}

    stats = {
        'cont_stds': training_examples[existing_cont].std().to_dict(),
        'count_vars': training_examples[existing_count].var().clip(lower=1e-5).to_dict(),
        'all_cols_dict': all_cols,
        'all_cols': training_examples.columns.tolist()
    }
    return stats

def perturb(examples, features, stats, seed=42):
    rng = np.random.default_rng(seed)
    perturbed = examples.copy()

    for idx in features:
        feat_name = stats['all_cols'][idx]
        if feat_name in stats['cont_stds']:
            noise = rng.normal(0, stats['cont_stds'][feat_name], size=len(perturbed))
            perturbed[:, idx] += noise
        else:
            mu = stats['count_vars'][feat_name] / 2
            noise = skellam.rvs(mu1=mu, mu2=mu, size=len(perturbed), random_state=rng)
            perturbed[:, idx] += noise

        # Clipping logic (Vectorized)
        if '%' in feat_name:
            perturbed[:, idx] = np.clip(perturbed[:, idx], 0, 1)
        else:
            perturbed[:, idx] = np.clip(perturbed[:, idx], 0, None)

    return perturbed


def calculate_max_diff(examples, model, stats, old_preds):
    all_features = [i for i, f in enumerate(stats['all_cols'])]
    perturbed_all = perturb(examples, all_features, stats)
    preds_all_perturbed = model.predict(perturbed_all)
    max_diff = np.abs(preds_all_perturbed - old_preds).max()
    return max_diff


def pgi(examples, ranking, model, training_examples, len_max=None, num_runs=25):
    examples_np = examples.to_numpy()
    old_preds = model.predict(examples_np)
    stats = get_dataset_stats(training_examples)

    num_unique_features = [f.split('x') for f in ranking]
    num_unique_features = list(set([f.replace(' ', '') for sublist in num_unique_features for f in sublist]))
    non_present = [f for f in training_examples.columns if f not in num_unique_features]
    calculate_non = False
    if len_max is None:
        calculate_non = True
        len_max = len(num_unique_features)

    max_diff = calculate_max_diff(examples_np, model, stats, old_preds)

    schedule = []
    stretch_counts = []  # How many times to repeat the result (for interaction terms)

    current_features_set = set()
    current_unique_names = []

    for term in ranking:
        new_vars = [f.replace(' ', '') for f in term.split('x')]
        new_unique_to_add = [v for v in new_vars if v not in current_features_set]
        if not new_unique_to_add:
            continue
        if len(current_features_set) + len(new_unique_to_add) >= len_max:
            break
        current_features_set.update(new_unique_to_add)
        current_unique_names.extend(new_unique_to_add)
        schedule.append([stats['all_cols_dict'][name] for name in current_unique_names])
        stretch_counts.append(len(new_unique_to_add))

    # Add non-present features to schedule
    if calculate_non:
        # Start with all features from the ranking perturbed
        base_features = [stats['all_cols_dict'][name] for name in num_unique_features]
        for j in range(1, len(non_present) + 1):
            extra_features = [stats['all_cols_dict'][name] for name in non_present[:j]]
            schedule.append(base_features + extra_features)
            stretch_counts.append(1)


    def compute_results(seedi):
        perturbed_all = perturb(examples_np.copy(), [i for i in range(len(stats['all_cols']))], stats, seed=seedi)
        results = []
        for i, feature_list in enumerate(schedule):
            p_data = examples_np.copy()
            p_data[:, feature_list] = perturbed_all[:, feature_list]
            new_preds = model.predict(p_data)
            diff = np.abs(new_preds - old_preds)
            for _ in range(stretch_counts[i]):
                results.append(diff)
        return results

    results = Parallel(n_jobs=25)(delayed(compute_results)(i) for i in range(num_runs))
    results = np.array(results)
    mean_diff_per_step = np.mean(results, axis=0)
    instance_curves = mean_diff_per_step.T
    num_steps = instance_curves.shape[1]
    x_axis = np.arange(num_steps) / (num_steps - 1)
    individual_aucs = np.array([auc(x_axis, curve) for curve in instance_curves])
    auc_max = auc(x_axis, [max_diff] * len(x_axis))
    return auc_max, individual_aucs


def pgu(examples, ranking, model, training_examples, len_max=None, num_runs=25):
    examples_np = examples.to_numpy()
    old_preds = model.predict(examples_np)
    stats = get_dataset_stats(training_examples)

    num_unique_features = [f.split('x') for f in ranking]
    num_unique_features = list(set([f.replace(' ', '') for sublist in num_unique_features for f in sublist]))
    non_present = [f for f in training_examples.columns if f not in num_unique_features]

    if len_max is None:
        len_max = len(num_unique_features) + len(non_present)
    else:
        len_max = len(num_unique_features) + len(non_present) - len_max

    max_diff = calculate_max_diff(examples_np, model, stats, old_preds)

    schedule = []
    stretch_counts = []
    features = [stats['all_cols_dict'][f] for f in non_present]
    for j in range(1, len(features) + 1):
        schedule.append(features[:j])
        stretch_counts.append(1)

    current_features_set = set()
    current_unique_names = []

    ranking_rev = []
    seen_in_forward = set()

    for term in ranking:
        new_vars = [f.strip() for f in term.split('x')]
        # Keep only features we haven't seen in a higher-ranked position
        new_unique_to_add = [v for v in new_vars if v not in seen_in_forward]

        ranking_rev.append(new_unique_to_add)
        seen_in_forward.update(new_unique_to_add)

    for term in reversed(ranking_rev):
        new_unique_to_add = [v for v in term if v not in current_features_set]
        if not new_unique_to_add:
            continue
        if len(current_features_set) + len(new_unique_to_add) + len(non_present) >= len_max:
            break
        current_features_set.update(new_unique_to_add)
        current_unique_names.extend(new_unique_to_add)
        schedule.append([stats['all_cols_dict'][name] for name in current_unique_names] + features)
        stretch_counts.append(len(new_unique_to_add))

    def compute_results(seedi):
        perturbed_all = perturb(examples_np.copy(), [i for i in range(len(stats['all_cols']))], stats, seed=seedi)
        results = []
        for i, feature_list in enumerate(schedule):
            p_data = examples_np.copy()
            p_data[:, feature_list] = perturbed_all[:, feature_list]
            new_preds = model.predict(p_data)
            diff = np.abs(new_preds - old_preds)
            for _ in range(stretch_counts[i]):
                results.append(diff)
        return results

    results = Parallel(n_jobs=25)(delayed(compute_results)(i) for i in range(num_runs))
    results = np.array(results)
    # results = np.mean(results, axis=0)
    mean_diff_per_step = np.mean(results, axis=0)
    instance_curves = mean_diff_per_step.T
    num_steps = instance_curves.shape[1]
    x_axis = np.arange(num_steps) / (num_steps - 1)
    individual_aucs = np.array([auc(x_axis, curve) for curve in instance_curves])
    auc_max = auc(x_axis, [max_diff] * len(x_axis))
    return auc_max, individual_aucs


def feature_agreement(gt, ranking2, all_features, removex=True):
    percent = []
    max_percent = 0.0
    features_current = set()

    if not removex:
        print(gt, ranking2[:len(gt) + 1])

    for k in range(len(gt), len(ranking2) + 1):
        top_k2_all = ranking2[:k]
        if removex:
            top_k2 = []
            for item in top_k2_all:
                parts = item.split('x')
                parts = [p.replace(' ', '') for p in parts]

                if set(parts).issubset(set(gt)):
                    top_k2.extend(parts)
        else:
            top_k2 = top_k2_all
        features_current.update(set(top_k2))
        metric = len(set(gt).intersection(set(top_k2))) / len(gt)
        percent.append(metric)
        if max_percent < metric:
            max_percent = metric
        if removex:
            if len(set(features_current)) == len(set(all_features)):
                break
    if removex:
        not_in_ranking = len(set(all_features) - features_current)
        for _ in range(not_in_ranking):
            percent.append(max_percent)
    else:
        not_in_ranking = len(set(gt) - features_current)
        for _ in range(not_in_ranking):
            percent.append(max_percent)
    if len(percent) == 1:
        return percent[0]
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

