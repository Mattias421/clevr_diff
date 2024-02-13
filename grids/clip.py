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
from . import plot_confusion_pacs as plot_confusion_pacs
from . import plot_confusion_clevr as plot_confusion_clevr 


    
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

    def process_sheep(self, sheep, history: List[dict], use_maximum_likelihood=True) -> dict:
        if history == []:
            return {}

        df = pd.DataFrame(history)

        if 'clevr' in sheep.xp.cfg.data.name:
            class_mrr, domain_mrr, class_acc, domain_acc = plot_confusion_clevr(df, sheep.xp, use_maximum_likelihood=use_maximum_likelihood)
        else:
            class_mrr, domain_mrr, class_acc, domain_acc = plot_confusion_pacs(df, sheep.xp, use_maximum_likelihood=use_maximum_likelihood)

        latex = False

        if latex:
            args = dict(sheep.xp.delta)
            tol = 1e-3 if 'll_ode_options.atol' not in args.keys() else args['ll_ode_options.atol']
            guidance_scale = 0.0 if 'pipe.guidance_scale' not in args.keys() else args['pipe.guidance_scale']
            print(f'{tol} & {guidance_scale} && {class_mrr:.4f} & {domain_mrr:.4f} && {class_acc:.4f} & {domain_acc:.4f} \\\\')

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


    sub = launcher.bind({'model':'clip',
                        'n_repeats':20,
                        'dora.dir':'outputs/clip'
                        })


    sub({'+data.name':'pacs',
         'data.path':'/mnt/parscratch/users/acq22mc/data/PACS',
})

    sub({'+data.name':'clevr',
        'data.path':'/mnt/parscratch/users/acq22mc/data/clevr/single_object/images'})
         
