
import os
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import random
import torch
import pandas as pd
from urllib import request
from PIL import Image


from torchmetrics.functional.multimodal import clip_score
from functools import partial
from dora import get_xp, hydra_main

import logging
logger = logging.getLogger(__name__)


clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")

def calculate_clip_score(images, prompts):
    clip_scores = np.zeros(len(prompts))
    images = images[None]
    images_int = (images * 255).astype("uint8")

    for i, p in enumerate(prompts):
        clip_scores[i] = clip_score_fn(torch.from_numpy(images_int).permute(0, 3, 1, 2), [p]).detach()
    return clip_scores


def pacs(cfg, xp):
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
                img = np.array(img) / 255

                # confound classes

                domain_name = 'painting' if domain == 'art_painting' else domain

                prompts = [prompt.format(obj=c, domain=domain_name) for c in classes]

                class_ll = calculate_clip_score(img, prompts)
                
                # confound domain

                prompts = [prompt.format(obj=class_, domain='painting' if d == 'art_painting' else d) for d in domains]
                
                domain_ll = calculate_clip_score(img, prompts)

                # make new results dict
                results = {'class': class_, 'domain': domain}

                for i, c in enumerate(classes):
                    results[f'll_{c}'] = class_ll[i]
                for i, d in enumerate(domains):
                    results[f'll_{d}'] = domain_ll[i] 

                xp.link.push_metrics(results)
                logger.info(results.values())



def clevr(cfg, xp):
    colours = ['red', 'green', 'blue', 'brown', 'yellow', 'gray', 'purple', 'cyan']
    shapes = ['sphere', 'cube', 'cylinder']

    dataset_path = cfg.data.path 

    prompt = 'a picture of a {obj}'

    for i in range(cfg.n_repeats):
        for colour in colours:
            for shape in shapes:
                
                obj = f'{colour} {shape}'

                print(f'obj: {obj}')

                number = str(i+1).zfill(6)

                file = f'CLEVR_new_large_{colour}_metal_{shape}{number}.png'
                img = Image.open(f'{dataset_path}/{file}').convert('RGB')
                img = np.array(img) / 255

                # confound colours

                prompts = [prompt.format(obj=f'{c}{shape}') for c in colours]
                colour_ll = calculate_clip_score(img, prompts)

                
                # confound shapes

                prompts = [prompt.format(obj=f'{colour}{s}') for s in shapes]
                shape_ll = calculate_clip_score(img, prompts)

                

                # make new results dict
                results = {'colour':colour, 'shape':shape,
                        'll_red':colour_ll[0], 'll_blue':colour_ll[1], 'll_green':colour_ll[2],
                        'll_gray':colour_ll[5], 'll_brown':colour_ll[3], 'll_yellow':colour_ll[4], 'll_purple':colour_ll[6], 'll_cyan':colour_ll[7],
                        'll_sphere':shape_ll[0], 'll_cube':shape_ll[1], 'll_cylinder':shape_ll[2],
                        'number':number}

                xp.link.push_metrics(results)
                logger.info(results.values())
                



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
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    generator = torch.Generator(device=device).manual_seed(0)

    xp = get_xp()

    if cfg.data.name == 'pacs':
        pacs(cfg, xp)
    else:
        clevr(cfg, xp)


