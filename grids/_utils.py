from scipy.stats import rankdata, spearmanr
import numpy as np


def mean_reciprocal_rank(log_likelihoods, use_maximum_likelihood=True):

    ranks = rankdata(log_likelihoods, method='min', axis=1)

    if use_maximum_likelihood:
        ranks = len(log_likelihoods[0]) + 1 - ranks
    else:
        # print('using minimum likelihood')
        pass

    scores = []
    hits = 0
    for i, rank in enumerate(ranks):
        score = 1/rank[i]
        scores.append(score)

        if rank[i] == 1:
            hits += 1

    mrr = np.mean(scores)
    accuracy = hits/len(ranks)

    return mrr, accuracy