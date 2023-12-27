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


def mean_reciprocal_rank(log_likelihoods, use_maximum_likelihood=False):

    ranks = rankdata(log_likelihoods, method='min', axis=1)

    if use_maximum_likelihood:
        ranks = len(log_likelihoods[0]) + 1 - ranks
    else:
        print('using minimum likelihood')

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
    
def plot_confusion(df, xp):
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
    class_mrr, class_acc = mean_reciprocal_rank(class_matrix)
    domain_mrr, domain_acc = mean_reciprocal_rank(domain_matrix)

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

class MyExplorer(Explorer):

    def get_grid_metrics(self) -> List[_Node]:
        return [
                tt.group('MRR', [
                    tt.leaf('Class', '.4f'),
                    tt.leaf('Domain', '.4f')]),
                tt.group('Accuracy', [
                    tt.leaf('Class', '.4f'),
                    tt.leaf('Domain', '.4f')])
                ]

    def process_sheep(self, sheep, history: List[dict]) -> dict:
        if history == []:
            return {}

        df = pd.DataFrame(history)

        class_mrr, domain_mrr, class_acc, domain_acc = plot_confusion(df, sheep.xp)

        return {'MRR': {'Class': class_mrr, 'Domain': domain_mrr},
                'Accuracy': {'Class': class_acc, 'Domain': domain_acc}}

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
    turbo_pipe_options = {'height':512, 'width':512, 
                    'num_inference_steps':4}
    turbo_sub = launcher.bind({'model':'sdxl_turbo',
                        'pipe':turbo_pipe_options,
                        'data.path':'/mnt/parscratch/users/acq22mc/data/PACS',
                        'dora.dir':'outputs/sdxl_pacs',
                        '+data.name':'pacs',
    })

    sub = launcher.bind({'model':'sdxl',
                        'pipe':pipe_options,
                        'data.path':'/mnt/parscratch/users/acq22mc/data/PACS',
                        'dora.dir':'outputs/sdxl_pacs',
                        '+data.name':'pacs',
    })

    with launcher.job_array():
        for guidance_scale, tol, reconstruct in product([0.0, 3.0, 5.0, 7.0], [1e-3, 1e-5], [True, False]):
            ode_options['atol'] = tol
            ode_options['rtol'] = tol

            sub({'pipe':{'guidance_scale':guidance_scale,
                        'reconstruct':reconstruct,
                        'll_guidance_scale':guidance_scale,
                        },
                'ode_options':ode_options,
                'll_ode_options':ode_options,})

            turbo_sub({'pipe':{'guidance_scale':guidance_scale,
                        'reconstruct':reconstruct,
                        'll_guidance_scale':guidance_scale,
                        },
                'ode_options':ode_options,
                'll_ode_options':ode_options,})
         