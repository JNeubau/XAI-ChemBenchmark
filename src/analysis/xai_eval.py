import numpy as np
from joblib import Parallel, delayed
from scipy.stats import spearmanr, skellam, weightedtau
from sklearn.metrics import auc


def get_dataset_stats(training_examples):
    """Calculate once per fold, not once per perturbation."""
    continuous = ['mol_wt', 'o%', 'n%', 'c%']
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
            # This automatically generates an independent noise array of length N
            noise = rng.normal(0, stats['cont_stds'][feat_name], size=len(perturbed))
            perturbed[:, idx] += noise
        else:
            mu = stats['count_vars'][feat_name] / 2
            noise = skellam.rvs(mu1=mu, mu2=mu, size=len(perturbed), random_state=rng)
            perturbed[:, idx] += noise

        if '%' in feat_name:
            perturbed[:, idx] = np.clip(perturbed[:, idx], 0, 1)
        else:
            perturbed[:, idx] = np.clip(perturbed[:, idx], 0, None)

    return perturbed


def calculate_max_diff(examples, model, stats, old_preds, num_runs=25):
    all_features = [i for i, f in enumerate(stats['all_cols'])]

    # Create a batch of the same example 'num_runs' times
    batch = np.repeat(examples, num_runs, axis=0)

    # Perturb the entire batch at once. 'perturb' will add independent noise to each row.
    perturbed_batch = perturb(batch, all_features, stats, seed=42)

    # Predict the whole batch in one optimized C-level call
    preds_all_perturbed = model.predict(perturbed_batch)

    # Calculate max difference across the batch
    max_diff = np.abs(preds_all_perturbed - old_preds[0]).max()
    return max_diff


def _evaluate_single_example_pgi(single_example_np, ranking, model, stats, old_pred, len_max, num_runs):
    single_example_2d = single_example_np.reshape(1, -1)
    old_pred_arr = np.array([old_pred])

    num_unique_features = [f.split('x') for f in ranking]
    num_unique_features = list(set([f.replace(' ', '') for sublist in num_unique_features for f in sublist]))
    non_present = [f for f in stats['all_cols'] if f not in num_unique_features]

    calculate_non = False
    if len_max is None:
        calculate_non = True
        len_max = len(num_unique_features)

    max_diff = calculate_max_diff(single_example_2d, model, stats, old_pred_arr, num_runs)

    schedule = []
    stretch_counts = []

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


    if calculate_non:
        base_features = [stats['all_cols_dict'][name] for name in num_unique_features]
        for j in range(1, len(non_present) + 1):
            extra_features = [stats['all_cols_dict'][name] for name in non_present[:j]]
            schedule.append(base_features + extra_features)
            stretch_counts.append(1)

    batch = np.repeat(single_example_2d, num_runs, axis=0)

    # Generate all independent perturbations at once for the entire batch
    perturbed_batch = perturb(batch.copy(), list(range(len(stats['all_cols']))), stats, seed=42)

    results = []
    for i, feature_list in enumerate(schedule):
        p_data = batch.copy()
        # Swap in the perturbed features for the whole batch
        p_data[:, feature_list] = perturbed_batch[:, feature_list]

        # Single fast prediction on 25 rows
        new_preds = model.predict(p_data)

        # Calculate the mean difference over the 25 perturbed variants
        mean_diff = np.abs(new_preds - old_pred).mean()

        for _ in range(stretch_counts[i]):
            results.append(mean_diff)

    auc_results = auc(np.arange(len(results)) / (len(results) - 1), results)
    auc_max = auc(np.arange(len(results)) / (len(results) - 1), [max_diff] * len(results))

    return auc_results / auc_max, auc_results, auc_max


def pgi(examples, rankings, model, training_examples, reference_list=None, len_max=None, num_runs=25, n_jobs=4):
    """
    n_jobs controls outer-loop parallelization. Set to 4 to run 4 examples concurrently.
    """
    examples_np = examples.to_numpy()
    old_preds = model.predict(examples_np)
    stats = get_dataset_stats(training_examples)

    # Helper function to process a single row so joblib can map it cleanly
    def process_single_pgi(idx):
        if reference_list is not None and reference_list[idx] is None:
            return None, None, None
        if rankings[idx] is None:
            return None, None, None

        ranking_features = rankings[idx]['features']
        ratio, raw, amax = _evaluate_single_example_pgi(
            examples_np[idx], ranking_features, model, stats, old_preds[idx], len_max, num_runs
        )
        return ratio, raw, amax

    # Run the outer loop in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_single_pgi)(idx) for idx in range(len(examples_np))
    )

    # Unpack the list of tuples back into three separate lists
    auc_ratios, auc_raws, auc_maxes = zip(*results)

    return list(auc_ratios), list(auc_raws), list(auc_maxes)


def _evaluate_single_example_pgu(single_example_np, ranking, model, stats, old_pred, len_max, num_runs):
    single_example_2d = single_example_np.reshape(1, -1)
    old_pred_arr = np.array([old_pred])

    num_unique_features = [f.split('x') for f in ranking]
    num_unique_features = list(set([f.replace(' ', '') for sublist in num_unique_features for f in sublist]))
    non_present = [f for f in stats['all_cols'] if f not in num_unique_features]

    if len_max is None:
        len_max = len(num_unique_features) + len(non_present)
    else:
        len_max = len(num_unique_features) + len(non_present) - len_max

    max_diff = calculate_max_diff(single_example_2d, model, stats, old_pred_arr, num_runs)

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

    # VECTORIZED COMPUTE
    batch = np.repeat(single_example_2d, num_runs, axis=0)
    perturbed_batch = perturb(batch.copy(), list(range(len(stats['all_cols']))), stats, seed=42)

    results = []
    for i, feature_list in enumerate(schedule):
        p_data = batch.copy()
        p_data[:, feature_list] = perturbed_batch[:, feature_list]

        new_preds = model.predict(p_data)
        mean_diff = np.abs(new_preds - old_pred).mean()

        for _ in range(stretch_counts[i]):
            results.append(mean_diff)

    auc_results = auc(np.arange(len(results)) / (len(results) - 1), results)
    auc_max = auc(np.arange(len(results)) / (len(results) - 1), [max_diff] * len(results))

    return auc_results / auc_max, auc_results, auc_max


def pgu(examples, rankings, model, training_examples, reference_list=None, len_max=None, num_runs=25, n_jobs=4):
    """
    n_jobs controls outer-loop parallelization. Set to 4 to run 4 examples concurrently.
    """
    examples_np = examples.to_numpy()
    old_preds = model.predict(examples_np)
    stats = get_dataset_stats(training_examples)

    # Helper function to process a single row
    def process_single_pgu(idx):
        if reference_list is not None and reference_list[idx] is None:
            return None, None, None
        if rankings[idx] is None:
            return None, None, None

        ranking_features = rankings[idx]['features']
        ratio, raw, amax = _evaluate_single_example_pgu(
            examples_np[idx], ranking_features, model, stats, old_preds[idx], len_max, num_runs
        )
        return ratio, raw, amax

    # Run the outer loop in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_single_pgu)(idx) for idx in range(len(examples_np))
    )

    # Unpack the list of tuples back into three separate lists
    auc_ratios, auc_raws, auc_maxes = zip(*results)

    return list(auc_ratios), list(auc_raws), list(auc_maxes)


def feature_agreement(gt, rankings, all_features, reference_list=None, remove_inter=False):
    auc_results = []
    for idx in range(len(rankings)):
        ranking2 = rankings[idx]
        if reference_list and reference_list[idx] is None:
            auc_results.append(None)
            continue
        else:
            if ranking2 is None:
                auc_results.append(None)
                continue
        ranking2 = rankings[idx]['features']
        percent = []
        max_percent = 0.0
        features_current = set()
        for k in range(len(gt), len(ranking2) + 1):
            top_k2_all = ranking2[:k]
            if remove_inter:
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
            if remove_inter:
                if len(set(features_current)) == len(set(all_features)):
                    break
        if remove_inter:
            not_in_ranking = len(set(all_features) - features_current)
            for _ in range(not_in_ranking):
                percent.append(max_percent)
        else:
            not_in_ranking = len(set(gt) - features_current)
            for _ in range(not_in_ranking):
                percent.append(max_percent)
        if len(percent) == 1:
            auc_results.append(percent[0])
        else:
            auc_results.append(auc(np.arange(len(percent)) / (len(percent) - 1), percent))
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