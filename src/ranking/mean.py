import numpy as np


def aggregate_rankings_by_mean_position(list_of_rankings: list) -> tuple:
    if not list_of_rankings:
        return []
    all_items = set()
    for ranking in list_of_rankings:
        all_items.update(list(ranking.keys()))
    item_scores = {}
    for item in all_items:
        positions = []
        for ranking in list_of_rankings:
            try:
                position = ranking[item]
            except:
                position = len(ranking)
            positions.append(position)
        item_scores[item] = np.mean(positions)
    sorted_items = sorted(item_scores.keys(), key=lambda item: item_scores[item])
    return sorted_items, item_scores