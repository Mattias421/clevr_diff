import typing as tp
from dora import Explorer
import treetable as tt
from treetable.table import _Node
from typing import List
from itertools import product

import numpy as np
import pandas as pd
from collections import defaultdict
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
    
def plot_confusion(df):
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
    plt.savefig('loglikelihoods.png')

    return colour_mrr, shape_mrr

class MyExplorer(Explorer):

    talks = 'total'

    def get_grid_metrics(self) -> List[_Node]:
        return [
                tt.group('MRR', [
                    tt.leaf('Colour', '.4f'),
                    tt.leaf('Shape', '.4f')])
                ]

    def process_history(self, history: List[dict]) -> dict:
        if history == []:
            return {}
        
        df = pd.DataFrame(history)

        colour_mrr, shape_mrr = plot_confusion(df)

        return {'MRR': {'Colour': colour_mrr, 'Shape': shape_mrr}}

@MyExplorer
def explorer(launcher):

    # stanage slurm options
    launcher.slurm_(partition='gpu',
              qos='gpu',
              mem_per_gpu='82',
              setup=['module unload Anaconda3',
                  'module load Anaconda3/2022.10',
                     'source activate diff-ll'],
              srun_args=['--export=ALL'],
              time=3*24*60,)

    pipe_options = {'height':1024, 'width':1024, 
                    'num_inference_steps':50}
    ode_options = {'num_inference_steps':50,
                   'atol':1e-3,
                   'rtol':1e-3,
                   'method':'dopri5'
                   }   
    sub = launcher.bind({'model':'sdxl',
                        'pipe':pipe_options,
                        'ode_options':ode_options,
                        'll_ode_options':ode_options,
                        'data.path':'/mnt/parscratch/users/acq22mc/data/clevr/single_object/images'
    })

    with launcher.job_array():
        for guidance_scale, reconstruct in product([0.0, 3.0, 5.0, 7.0], [True, False]):
            sub({'pipe':{'guidance_scale':guidance_scale,
                        'reconstruct':reconstruct,
                        'll_guidance_scale':guidance_scale,
                        }})
         