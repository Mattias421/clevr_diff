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
    full_determinism = cfg.full_determinism

    if full_determinism:
        logger.info('Running with full determinism')
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = cfg.cublas_workspace_config

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
        pipeline = TurboPipe(device, use_fp16=(not cfg.full_determinism))
    elif cfg.model == 'sdxl':
        pipeline = Pipe(device, use_fp16=(not cfg.full_determinism))

    ll_ode_options = cfg.ll_ode_options
    ode_options = cfg.ode_options

    classes = ['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person']
    domains = ['art_painting', 'cartoon', 'photo', 'sketch']

    dataset_path = cfg.data.path 

    df = pd.read_csv(f'{dataset_path}/manifest.csv')

    prompt = 'a {domain} of a {obj}'


    for i in range(cfg.n_repeats):
        for class_ in classes:
            for domain in domains:
                
                obj = class_

                logger.info(f'obj: {obj}')

                file = df[(df['domain'] == domain) & (df['class'] == class_)]['image'].values[i]
                img = Image.open(file).convert('RGB')

                # confound classes

                domain_name = 'painting' if domain == 'art_painting' else domain

                prompts = [prompt.format(obj=c, domain=domain_name) for c in classes]

                class_ll_results = pipeline(prompts=prompts,
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
                
                # confound domain

                prompts = [prompt.format(obj=class_, domain='painting' if d == 'art_painting' else d) for d in domains]

                domain_ll_results = pipeline(prompts=prompts,
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
                
                class_ll = class_ll_results['log_likelihood'].tolist()
                domain_ll = domain_ll_results['log_likelihood'].tolist()

                # make new results dict
                results = {'class': class_, 'domain': domain}

                for i, c in enumerate(classes):
                    results[f'll_{c}'] = class_ll[i]
                for i, d in enumerate(domains):
                    results[f'll_{d}'] = domain_ll[i] 

                xp.link.push_metrics(results)
                logger.info(results.values())
            

            









