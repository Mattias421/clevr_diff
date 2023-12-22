import os
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import random
import torch
import pandas as pd
from urllib import request
from PIL import Image

from .sdxl_turbo_ll import LogLikelihoodPipeline as TurboPipe
# from pipeline_stable_diffusion_log_likelihood import LogLikelihoodPipeline as Pipe

from dora import get_xp, hydra_main

import logging
logger = logging.getLogger(__name__)

full_determinism = False

os.environ['PYTHONHASHSEED'] = str(0)
np.random.seed(0)
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(full_determinism)
generator = torch.Generator(device='cuda:0').manual_seed(0)




@hydra_main(config_path='config', config_name='config')
def main(cfg):

    xp = get_xp()

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    pipeline = TurboPipe(device, use_fp16=True)

    ll_ode_options = {
        'num_inference_steps': 4,
        'atol': 1e-3,
        'rtol': 1e-3,
        'method': 'dopri5',
    }
    ode_options = {
        'num_inference_steps': 4,
        'atol': 1e-5,
        'rtol': 1e-5,
        'method': 'euler',
    }

    colours = ['red', 'green', 'blue', 'brown', 'yellow', 'gray', 'purple', 'cyan']
    shapes = ['sphere', 'cube', 'cylinder']


    dataset_path = '/store/store4/data/clevr/single_object'
    dataset = pd.read_csv(f'{dataset_path}/train.csv')

    colour_hits = 0
    shape_hits = 0
    colour_shape_hits = 0
    total = 0

    prompt = 'a picture of a {obj}'

    results = {'colour': [], 'shape': [], 
               'll_red':[], 'll_blue':[], 'll_green':[],
               'll_gray':[], 'll_brown':[], 'll_yellow':[], 'll_purple':[], 'll_cyan':[],
               'll_sphere':[], 'll_cube':[], 'll_cylinder':[],}

    for colour in colours:
        for shape in shapes:
            
            obj = f'{colour} {shape}'

            print(f'obj: {obj}')

            file = f'CLEVR_new_large_{colour}_metal_{shape}000001.png'
            img = Image.open(f'{dataset_path}/images/{file}').convert('RGB')

            # confound colours

            prompts = [prompt.format(obj=f'{c}{shape}') for c in colours]

            colour_ll_results = pipeline(prompts=prompts,
                                    images=img,
                                    height=512, width=512,
                                    reconstruct=False,
                                    guidance_scale=0,
                                    ll_guidance_scale=0,
                                    ll_ode_options=ll_ode_options,
                                    reconstruct_ode_options=ode_options,
                                    num_inference_steps=4,
                                    generator=generator,
                                    return_image=False)
            
            # confound shapes

            prompts = [prompt.format(obj=f'{colour}{s}') for s in shapes]

            shape_ll_results = pipeline(prompts=prompts,
                                    images=img,
                                    height=512, width=512,
                                    reconstruct=False,
                                    guidance_scale=0,
                                    ll_guidance_scale=0,
                                    ll_ode_options=ll_ode_options,
                                    reconstruct_ode_options=ode_options,
                                    num_inference_steps=4,
                                    generator=generator,
                                    return_image=False)
            
            results['colour'].append(colour)
            results['shape'].append(shape)

            colour_ll = colour_ll_results['log_likelihood'].tolist()
            shape_ll = shape_ll_results['log_likelihood'].tolist()

            results['ll_red'].append(colour_ll[0])
            results['ll_blue'].append(colour_ll[1])
            results['ll_green'].append(colour_ll[2])
            results['ll_brown'].append(colour_ll[3])
            results['ll_yellow'].append(colour_ll[4])
            results['ll_gray'].append(colour_ll[5])
            results['ll_purple'].append(colour_ll[6])
            results['ll_cyan'].append(colour_ll[7])

            results['ll_sphere'].append(shape_ll[0])
            results['ll_cube'].append(shape_ll[1])
            results['ll_cylinder'].append(shape_ll[2])

            # calc if colour got hit
            colour_hits += int(np.argmax(colour_ll) == colours.index(colour))
            shape_hits += int(np.argmax(shape_ll) == shapes.index(shape))
            colour_shape_hits += int(np.argmax(colour_ll) == colours.index(colour) and np.argmax(shape_ll) == shapes.index(shape))
            total += 1

            metrics = {'acc_colour': colour_hits / total,
                       'acc_shape': shape_hits / total,
                       'acc_colour_shape': colour_shape_hits / total,
                       'completion': total/(len(colours)*len(shapes)),
                       }

            xp.link.push_metrics(metrics)
            logger.info(metrics)
            

            
    pd.DataFrame(results).to_csv('loglikelihoods.csv')









