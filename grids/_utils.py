from scipy.stats import rankdata, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

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


def plot_confusion_clevr(df, xp, use_maximum_likelihood=True):
    colours = df['colour'].unique()
    shapes = df['shape'].unique()

    colour_matrix = np.zeros((len(colours), len(colours)))

    for i, colour in enumerate(colours):
        row = df[df['colour'] == colour][[f'll_{c}' for c in colours]].mean(axis=0)
        colour_matrix[i] = row.to_list()

    shape_matrix = np.zeros((len(shapes), len(shapes)))

    for i, shape in enumerate(shapes):
        row = df[df['shape'] == shape][[f'll_{s}' for s in shapes]].mean(axis=0)
        shape_matrix[i] = row.to_list()

    # compute mean reciprocal rank
    colour_mrr, colour_acc = mean_reciprocal_rank(colour_matrix, use_maximum_likelihood=use_maximum_likelihood)
    shape_mrr, shape_acc = mean_reciprocal_rank(shape_matrix, use_maximum_likelihood=use_maximum_likelihood)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    sns.heatmap(colour_matrix, ax=ax[0], cmap='jet')
    sns.heatmap(shape_matrix, ax=ax[1], cmap='jet')

    # Label x and y ticks for the first heatmap
    ax[0].set_xticks(np.arange(len(colours)) + 0.5)
    ax[0].set_yticks(np.arange(len(colours)) + 0.5)
    ax[0].set_xticklabels(colours)
    ax[0].set_yticklabels(colours)

    # Label x and y ticks for the second heatmap
    ax[1].set_xticks(np.arange(len(shapes))+0.5)
    ax[1].set_yticks(np.arange(len(shapes))+0.5)
    ax[1].set_xticklabels(shapes)
    ax[1].set_yticklabels(shapes)

    ax[0].set_title('colour')
    ax[1].set_title('shape')
    plt.tight_layout()
    plt.savefig(str(xp.folder) + '/loglikelihoods.png')

    return colour_mrr, shape_mrr, colour_acc, shape_acc

def plot_confusion_pacs(df, xp, use_maximum_likelihood=True):
    classes = df['class'].unique()
    domains = df['domain'].unique()

    class_matrix = np.zeros((len(classes), len(classes)))

    for i, class_ in enumerate(classes):
        row = df[df['class'] == class_][[f'll_{c}' for c in classes]].mean(axis=0)
        class_matrix[i] = row.to_list()

    domain_matrix = np.zeros((len(domains), len(domains)))

    for i, domain in enumerate(domains):
        row = df[df['domain'] == domain][[f'll_{d}' for d in domains]].mean(axis=0)
        domain_matrix[i] = row.to_list()

    # compute mean reciprocal rank
    class_mrr, class_acc = mean_reciprocal_rank(class_matrix, use_maximum_likelihood=use_maximum_likelihood)
    domain_mrr, domain_acc = mean_reciprocal_rank(domain_matrix, use_maximum_likelihood=use_maximum_likelihood)

    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    sns.heatmap(class_matrix, ax=ax[0, 0], cmap='jet')
    sns.heatmap(domain_matrix, ax=ax[0, 1], cmap='jet')

    # Label x and y ticks for the first heatmap
    ax[0, 0].set_xticks(np.arange(len(classes)) + 0.5)
    ax[0, 0].set_yticks(np.arange(len(classes)) + 0.5)
    ax[0, 0].set_xticklabels(classes)
    ax[0, 0].set_yticklabels(classes)

    # Label x and y ticks for the second heatmap
    ax[0, 1].set_xticks(np.arange(len(domains))+0.5)
    ax[0, 1].set_yticks(np.arange(len(domains))+0.5)
    ax[0, 1].set_xticklabels(domains)
    ax[0, 1].set_yticklabels(domains)

    ax[0, 0].set_title('class')
    ax[0, 1].set_title('domain')

    # Compute row min-max normalized matrices
    class_matrix_row_norm = (class_matrix - class_matrix.min(axis=1, keepdims=True)) / (class_matrix.max(axis=1, keepdims=True) - class_matrix.min(axis=1, keepdims=True))
    domain_matrix_row_norm = (domain_matrix - domain_matrix.min(axis=1, keepdims=True)) / (domain_matrix.max(axis=1, keepdims=True) - domain_matrix.min(axis=1, keepdims=True))

    sns.heatmap(class_matrix_row_norm, ax=ax[1, 0], cmap='jet')
    sns.heatmap(domain_matrix_row_norm, ax=ax[1, 1], cmap='jet')

    # Label x and y ticks for the row-normalized heatmaps
    ax[1, 0].set_xticks(np.arange(len(classes)) + 0.5)
    ax[1, 0].set_yticks(np.arange(len(classes)) + 0.5)
    ax[1, 0].set_xticklabels(classes)
    ax[1, 0].set_yticklabels(classes)

    ax[1, 1].set_xticks(np.arange(len(domains))+0.5)
    ax[1, 1].set_yticks(np.arange(len(domains))+0.5)
    ax[1, 1].set_xticklabels(domains)
    ax[1, 1].set_yticklabels(domains)

    ax[1, 0].set_title('class (row-normalized)')
    ax[1, 1].set_title('domain (row-normalized)')

    plt.tight_layout()
    plt.savefig(str(xp.folder) + '/loglikelihoods.png')

    return class_mrr, domain_mrr, class_acc, domain_acc
