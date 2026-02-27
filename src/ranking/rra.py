"""
Python code for rank aggregation, for both full and partial lists.  For methods/algorithms
I have followed the paper

"Rank aggregation methods for the web" (2001) C. Dwork, R. Kumar, M. Naor, D. Sivakumar.
Proceedings of the 10th international conference on World Wide Web.

Created May 22, 2015

@author: Kevin S. Brown, University of Connecticut

This source code is provided under the BSD-3 license (see LICENSE file).

Copyright (c) 2015, Kevin S. Brown
All rights reserved.
"""

from numpy import zeros,sort,zeros_like

from scipy.stats import binom


def sort_by_value(d,reverse=False):
    """
    One of many ways to sort a dictionary by value.
    """
    return [(k,d[k]) for k in sorted(d,key=d.get,reverse=reverse)]


class RankAggregator(object):
    """
    Base class for full and partial list rank aggregation methods.  Should not be called
    directly except in testing situations; houses shared methods to both.
    """
    def __init__(self):
        pass


    def item_universe(self,rank_list):
        """
        Determines the universe of ranked items (union of all the items ranked by all
        experts).  Not necessary for full lists.
        """
        return list(frozenset().union(*[list(x.keys()) for x in rank_list]))


    def first_order_marginals(self,rank_list):
        """
        Computes m_ik, the fraction of rankers that ranks item i as their kth choice
        (see Ammar and Shah, "Efficient Rank Aggregation Using Partial Data").  Works
        with either full or partial lists.
        """
        # get list of all the items
        all_items = self.item_universe(rank_list)
        # dictionaries for creating the matrix
        self.item_mapping(all_items)
        # create the m_ik matrix and fill it in
        m_ik = zeros((len(all_items),len(all_items)))
        n_r = len(rank_list)
        for r in rank_list:
            for item in r:
                m_ik[self.itemToIndex[item],r[item]-1] += 1
        return m_ik/n_r


    def convert_to_ranks(self,scoreDict):
        """
        Accepts an input dictionary in which they keys are items to be ranked (numerical/string/etc.)
        and the values are scores, in which a higher score is better.  Returns a dictionary of
        items and ranks, ranks in the range 1,...,n.
        """
        # default sort direction is ascending, so reverse (see sort_by_value docs)
        x = sort_by_value(scoreDict,True)
        y = list(zip(list(zip(*x))[0],range(1,len(x)+1)))
        ranks = {}
        for t in y:
            ranks[t[0]] = t[1]
        return ranks


    def item_ranks(self,rank_list):
        """
        Accepts an input list of ranks (each item in the list is a dictionary of item:rank pairs)
        and returns a dictionary keyed on item, with value the list of ranks the item obtained
        across all entire list of ranks.
        """
        item_ranks = {}.fromkeys(rank_list[0])
        for k in item_ranks:
            item_ranks[k] = [x[k] for x in rank_list]
        return item_ranks


    def item_mapping(self,items):
        """
        Some methods need to do numerical work on arrays rather than directly using dictionaries.
        This function maps a list of items (they can be strings, ints, whatever) into 0,...,len(items).
        Both forward and reverse dictionaries are created and stored.
        """
        self.itemToIndex = {}
        self.indexToItem = {}
        indexToItem = {}
        next = 0
        for i in items:
            self.itemToIndex[i] = next
            self.indexToItem[next] = i
            next += 1
        return


class FullListRankAggregator(RankAggregator):
    """
    Performs rank aggregation, using a variety of methods, for full lists
    (all items are ranked by all experts).
    """
    def __init__(self):
        super(RankAggregator,self).__init__()
        # used for method dispatch


    def aggregate_ranks(self,experts,areScores=True,*args):
        """
        Combines the ranks in the list experts to obtain a single
        set of aggregate ranks.  Can operate on either scores
        or ranks; scores are assumed to always mean higher=better.

        INPUT:
        ------
            experts : list of dictionaries, required
                each element of experts should be a dictionary of item:score
                or item:rank pairs

            areScores : bool, optional
                set to True if the experts provided scores, False if they
                provide ranks
        """
        # if the input data is scores, we need to convert
        if areScores:
            ranklist = [self.convert_to_ranks(e) for e in experts]
        else:
            ranklist = experts
        scores,aggRanks = self.robust_aggregation(ranklist)
        return scores,aggRanks

    def robust_aggregation(self,rank_list):
        """
        Implements the robust rank aggregation scheme of Kolde, Laur, Adler,
        and Vilo in "Robust rank aggregation for gene list integration and
        meta-analysis", Bioinformatics 28(4) 2012.  Essentially compares
        order statistics of normalized ranks to a uniform distribution.
        """
        def beta_calc(x):
            bp = zeros_like(x)
            n = len(x)
            for k in range(n):
                b = binom(n,x[k])
                for l in range(k,n):
                    bp[k] += b.pmf(l+1)
            return bp
        scores = {}.fromkeys(rank_list[0])
        item_ranks = self.item_ranks(rank_list)
        N = len(scores)
        # sort and normalize the ranks, and then compute the item score
        for item in item_ranks:
            item_ranks[item] = sort([1.0*x/N for x in item_ranks[item]])
            # the 1.0 here is to make *large* scores correspond to better ranks
            scores[item] = 1.0 - min(beta_calc(item_ranks[item]))
        return scores,self.convert_to_ranks(scores)


def aggregate_rankings_by_rra(list_of_rankings: list) -> tuple:
    if not list_of_rankings:
        return []
    all_ranking_items = set()
    for ranking in list_of_rankings:
        all_ranking_items.update(list(ranking.keys()))
    all_rankings_items = list(all_ranking_items)
    for ranking in list_of_rankings:
        max_pos = len(ranking)
        for item in all_rankings_items:
            if item not in ranking:
                ranking[item] = max_pos
    flra = FullListRankAggregator()
    scores, agg_ranking = flra.aggregate_ranks(list_of_rankings, areScores=False)
    sorted_items = sorted(agg_ranking.keys(), key=lambda item: agg_ranking[item])
    return sorted_items, scores
