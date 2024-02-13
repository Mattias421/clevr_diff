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
                tt.group(key, [
                    tt.leaf('ll', '.4f'),
                    tt.leaf('fid', '.4f'),
                    tt.leaf('clip', '.4f'),])
                for key in ['init', 'half', 'final', 'ref']
                ]
    
    def bpd(self, ll):
        return -ll * np.log2(np.exp(1)) / (3 * 1024 * 1024)

    def process_sheep(self, sheep, history: List[dict]) -> dict:
        if history == []:
            return {}

        print(history)
        
        metrics = {}

        keys = ['init', 'half', 'final', 'ref']

        for key in keys:
            df = pd.DataFrame([h[key] for h in history])

            df['ll'] = df['ll'].map(self.bpd)

            metrics[key] = {'ll': df['ll'].mean(),
                            'fid': df['fid'].mean(),
                            'clip': df['clip'].mean()}
        
        return metrics
        


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
                        'full_determinism':False,
                        'n_repeats':20, # currently small for speed testing
                        'pipe':pipe_options,
                        'data.path':'/mnt/parscratch/users/acq22mc/data/PACS',
                        'dora.dir':'outputs/sdxl_pacs',
                        '+data.name':'pacs',
                        '+task':'sample_ll',
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

         
