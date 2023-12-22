import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import rankdata, spearmanr


def mean_reciprocal_rank(log_likelihoods, use_maximum_likelihood=True):

    ranks = rankdata(log_likelihoods, method='min', axis=1)

    if use_maximum_likelihood:
        ranks = len(log_likelihoods[0]) + 1 - ranks

    scores = []
    for rank in ranks:
        score = 1/rank[0]
        scores.append(score)

    mrr = np.mean(scores)

    return mrr


root = '/exp/exp4/acq22mc/outputs/clevr_diff/xps/darkseagreenMantine/'
df = pd.read_csv(root + 'loglikelihoods.csv')

colours = df['colour'].unique()
shapes = df['shape'].unique()

colour_matrix = np.zeros((len(colours), len(colours)))

for i, colour in enumerate(colours):
    row = df[df['colour'] == colour][[f'll_{c}' for c in colours]].mean(axis=0)
    print(f'colour: {colour}')
    print(row.to_list())

    colour_matrix[i] = row.to_list()

shape_matrix = np.zeros((len(shapes), len(shapes)))

for i, shape in enumerate(shapes):
    row = df[df['shape'] == shape][[f'll_{s}' for s in shapes]].mean(axis=0)
    print(f'shape: {shape}')
    print(row.to_list())

    shape_matrix[i] = row.to_list()

# compute mean reciprocal rank
colour_mrr = mean_reciprocal_rank(colour_matrix)
shape_mrr = mean_reciprocal_rank(shape_matrix)

print(f'colour mrr: {colour_mrr}')
print(f'shape mrr: {shape_mrr}')

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
plt.savefig(root + 'loglikelihoods.png')