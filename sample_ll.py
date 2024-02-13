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

from torchmetrics.functional.multimodal import clip_score
from functools import partial
clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")

from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.transforms.functional import pil_to_tensor

def p2t(img):
    return pil_to_tensor(img).unsqueeze(0)

def calculate_clip_score(images, prompts):
    clip_scores = np.zeros(len(prompts))
    images = p2t(images)
    images_np = images.permute(0, 2, 3, 1).cpu().numpy()
    images_int = (images_np * 255).astype("uint8")

    for i, p in enumerate(prompts):
        clip_scores[i] = clip_score_fn(torch.from_numpy(images_int).permute(0, 3, 1, 2), [p]).detach()
    return clip_scores


import logging
logger = logging.getLogger(__name__)


@hydra_main(version_base=None, config_path='config', config_name='config')
def main(cfg):
    full_determinism = cfg.full_determinism

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

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
    generator = torch.Generator(device=device).manual_seed(0)

    xp = get_xp()
    logger.info(f'Starting XP {xp.sig}')


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

    fid = {}
    os.makedirs('images', exist_ok=True)

    img_dir = xp.folder / 'images'
    img_dir.mkdir()

    for i in range(cfg.n_repeats):
        for class_ in classes:
            for domain in domains:
                
                obj = class_

                logger.info(f'obj: {obj}')

                file = df[(df['domain'] == domain) & (df['class'] == class_)]['image'].values[i]
                img = Image.open(file).convert('RGB')

                # confound classes

                domain_name = 'painting' if domain == 'art_painting' else domain

                prompts = [prompt.format(obj=class_, domain=domain_name)]

                ref_ll = pipeline(prompts=prompts,
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

                ref_ll = ref_ll['log_likelihood'].item()
                clip_ref = calculate_clip_score(img, prompts)[0]

                results = {'ref':{'ll':ref_ll, 'fid':0.0, 'clip':clip_ref}}

                for stage in ['init', 'q1', 'half', 'q3', 'final']:

                    if stage not in fid.keys():
                        fid[stage] = FrechetInceptionDistance()

                    sample = pipeline.generate(prompts=prompts,
                                                height=cfg.pipe.height, width=cfg.pipe.width,
                                                guidance_scale=cfg.pipe.guidance_scale,
                                                reconstruct_ode_options=ode_options,
                                                num_inference_steps=cfg.pipe.num_inference_steps,
                                                generator=generator,
                                                stage=stage,)
                    sample_ll = pipeline(prompts=prompts,
                                                images=sample,
                                                height=cfg.pipe.height, width=cfg.pipe.width,
                                                reconstruct=cfg.pipe.reconstruct,
                                                guidance_scale=cfg.pipe.guidance_scale,
                                                ll_guidance_scale=cfg.pipe.ll_guidance_scale,
                                                ll_ode_options=ll_ode_options,
                                                reconstruct_ode_options=ode_options,
                                                num_inference_steps=cfg.pipe.num_inference_steps,
                                                generator=generator,
                                                return_image=False)

                    sample_ll = sample_ll['log_likelihood'].item()
                    sample = sample[0]

                    fid[stage].update(p2t(img), real=True)
                    fid[stage].update(p2t(sample), real=False)

                    clip = calculate_clip_score(sample, prompts)[0]

                    img_path = img_dir / f'{stage}_{class_}_{domain}_{i}.png'
                    sample.save(img_path)

                    results[stage] = {'ll':sample_ll, 'clip':clip, 'index':i}

                    if i > 1:
                        results[stage]['fid'] = fid[stage].compute()
                    else:
                        results[stage]['fid'] = None
                


                xp.link.push_metrics(results)
                logger.info(results.values())
            

            









