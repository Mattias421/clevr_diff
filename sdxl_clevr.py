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
from .pipeline_stable_diffusion_log_likelihood import LogLikelihoodPipeline as Pipe

from dora import get_xp, hydra_main

import logging
logger = logging.getLogger(__name__)


@hydra_main(version_base=None, config_path='config', config_name='config')
def main(cfg):
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

    xp = get_xp()

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    if cfg.model == 'sdxl_turbo':
        pipeline = TurboPipe(device, use_fp16=True)
    elif cfg.model == 'sdxl':
        pipeline = Pipe(device, use_fp16=True)

    ll_ode_options = cfg.ll_ode_options
    ode_options = cfg.ode_options

    colours = ['red', 'green', 'blue', 'brown', 'yellow', 'gray', 'purple', 'cyan']
    shapes = ['sphere', 'cube', 'cylinder']

    dataset_path = cfg.data.path 

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
            img = Image.open(f'{dataset_path}/{file}').convert('RGB')

            # confound colours

            prompts = [prompt.format(obj=f'{c}{shape}') for c in colours]

            colour_ll_results = pipeline(prompts=prompts,
                                        images=img,
                                        height=cfg.pipe.height, width=cfg.pipe.width,
                                        reconstruct=cfg.pipe.reconstruct,
                                        guidance_scale=cfg.pipe.guidance_scale,
                                        ll_guidance_scale=cfg.pipe.ll_guidance_scale,
                                        ll_ode_options=ll_ode_options,
                                        reconstruct_ode_options=ode_options,
                                        num_inference_steps=cfg.pipe.num_inference_steps,
                                        generator=generator,
                                        return_image=False)
            
            # confound shapes

            prompts = [prompt.format(obj=f'{colour}{s}') for s in shapes]

            shape_ll_results = pipeline(prompts=prompts,
                                        images=img,
                                        height=cfg.pipe.height, width=cfg.pipe.width,
                                        reconstruct=cfg.pipe.reconstruct,
                                        guidance_scale=cfg.pipe.guidance_scale,
                                        ll_guidance_scale=cfg.pipe.ll_guidance_scale,
                                        ll_ode_options=ll_ode_options,
                                        reconstruct_ode_options=ode_options,
                                        num_inference_steps=cfg.pipe.num_inference_steps,
                                        generator=generator,
                                        return_image=False)
            
            results['colour'].append(colour)
            results['shape'].append(shape)

            colour_ll = colour_ll_results['log_likelihood'].tolist()
            shape_ll = shape_ll_results['log_likelihood'].tolist()

            # make new results dict
            results = {'colour':colour, 'shape':shape,
                       'll_red':colour_ll[0], 'll_blue':colour_ll[1], 'll_green':colour_ll[2],
                       'll_gray':colour_ll[5], 'll_brown':colour_ll[3], 'll_yellow':colour_ll[4], 'll_purple':colour_ll[6], 'll_cyan':colour_ll[7],
                       'll_sphere':shape_ll[0], 'll_cube':shape_ll[1], 'll_cylinder':shape_ll[2],}

            xp.link.push_metrics(results)
            logger.info(results.values())
            

            









