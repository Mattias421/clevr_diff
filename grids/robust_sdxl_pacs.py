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
from . import plot_confusion_pacs as plot_confusion 
    
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

        class_mrr, domain_mrr, class_acc, domain_acc = plot_confusion(df, sheep.xp, use_maximum_likelihood=use_maximum_likelihood)

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
                        'full_determinism':False,
                        'n_repeats':20, # currently small for speed testing
                        'pipe':pipe_options,
                        'data.path':'/mnt/parscratch/users/acq22mc/data/PACS',
                        'dora.dir':'outputs/sdxl_pacs',
                        '+data.name':'pacs',
    })

    with launcher.job_array():
        for guidance_scale, tol, reconstruct in product([0.0, 3.0, 5.0, 7.0], [1e-3], [True, False]):
            ode_options['atol'] = tol
            ode_options['rtol'] = tol

            sub({'pipe':{'guidance_scale':guidance_scale,
                        'reconstruct':reconstruct,
                        'll_guidance_scale':guidance_scale,
                         **pipe_options
                        },
                'ode_options':ode_options,
                'll_ode_options':ode_options,})

            # turbo_sub({'pipe':{'guidance_scale':guidance_scale,
            #             'reconstruct':reconstruct,
            #             'll_guidance_scale':guidance_scale,
            #             },
            #     'ode_options':ode_options,
            #     'll_ode_options':ode_options,})
         
