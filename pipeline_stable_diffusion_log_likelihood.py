from diffusers import StableDiffusionXLPipeline
from diffusers.image_processor import VaeImageProcessor
import torch
from torchmetrics.functional.multimodal import clip_score
from functools import partial
from tqdm import tqdm
from PIL import Image
import numpy as np
import random
from torchdiffeq import odeint
from seaborn import heatmap
import matplotlib.pyplot as plt

import hydra
from omegaconf import DictConfig, OmegaConf

import os
from scipy.stats import rankdata

from typing import TYPE_CHECKING, List, Optional, Tuple, Union, Dict

import logging
logger = logging.getLogger(__name__)

class LogLikelihoodPipeline():
    def __init__(self, device, use_fp16=False):
        model_checkpoint ="stabilityai/stable-diffusion-xl-base-1.0"

        if use_fp16:
            self.pipe = StableDiffusionXLPipeline.from_pretrained(model_checkpoint, use_safetensors=True, torch_dtype=torch.float16, variant="fp16")
        else:
            self.pipe = StableDiffusionXLPipeline.from_pretrained(model_checkpoint, use_safetensors=True, variant="fp16")

        self.use_fp16 = use_fp16

        self.device = device
        self.pipe = self.pipe.to(device)

        self.vae_scale_factor = 2 ** (len(self.pipe.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

    def sigma_to_t(self, sigma, log_sigmas):
        # get log sigma
        log_sigma = torch.log(sigma)

        # get distribution
        dists = log_sigma - log_sigmas[:, None]

        # get sigmas range
        low_idx = torch.cumsum((dists >= 0), dim=0).argmax(dim=0).clamp(max=log_sigmas.shape[0] - 2)
        high_idx = low_idx + 1

        low = log_sigmas[low_idx]
        high = log_sigmas[high_idx]

        # interpolate sigmas
        w = (low - log_sigma) / (low - high)
        w = torch.clamp(w, 0, 1)

        # transform interpolation to time range
        t = (1 - w) * low_idx + w * high_idx
        return t[0]

    @torch.no_grad()
    def log_likelihood(self, model, x, sigma_min, sigma_max, log_sigmas, prompt_embeds, added_cond_kwargs, guidance_scale, ode_options):
        v = torch.randint_like(x, 2) * 2 - 1
        do_classifier_free_guidance = guidance_scale > 1.0

        sigma_to_t = self.sigma_to_t

        class ODEfunc(torch.nn.Module):
            def __init__(self):
                super(ODEfunc, self).__init__()

                self.nfev = 0

            def forward(self, sigma, x):
                with torch.enable_grad():
                    x = x[0].detach().requires_grad_()

                    latent_model_input = torch.cat([x] * 2) if do_classifier_free_guidance else x
                    latent_model_input = latent_model_input.to(dtype=model.dtype)
                    latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5)

                    t = int(sigma_to_t(sigma, log_sigmas))

                    # predict the noise residual
                    noise_pred = model(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        added_cond_kwargs=added_cond_kwargs,
                        return_dict=False,
                    )[0]

                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                    noise_pred = noise_pred.to(dtype=torch.float32)

                    d = noise_pred 
                    grad = torch.autograd.grad((d * v).sum(), x)[0]
                    d_ll = (v * grad).flatten(1).sum(1)
                self.nfev += 1

                return d.detach(), d_ll

        x = x.to(dtype=torch.float32)
        x_min = x, x.new_zeros([x.shape[0]])
        t = x.new_tensor([sigma_min, sigma_max])
        ode_func = ODEfunc().cuda()

        method = ode_options['method']
        step_size = abs(sigma_min - sigma_max) / ode_options['num_inference_steps']
        atol = ode_options['atol']
        rtol = ode_options['rtol']

        sol = odeint(ode_func, x_min, t, atol=atol, rtol=rtol, method=method, options={'step_size': step_size})

        latent, delta_ll = sol[0][-1], sol[1][-1]
        ll_prior = torch.distributions.Normal(0, sigma_max).log_prob(latent).flatten(1).sum(1)
        return ll_prior + delta_ll, ll_prior, delta_ll, ode_func.nfev, latent

    @torch.no_grad()
    def ode_sample(self, model, x, sigma_min, sigma_max, log_sigmas, prompt_embeds, added_cond_kwargs, guidance_scale, ode_options):

        do_classifier_free_guidance = guidance_scale > 1.0

        sigma_to_t = self.sigma_to_t
        
        class ODEfunc(torch.nn.Module):
            def __init__(self):
                super(ODEfunc, self).__init__()

            def forward(self, sigma, x):
                zeros = x[1]
                x = x[0] 

                if sigma_max < sigma_min:
                    sigma = sigma_max if sigma_max > sigma else sigma # hacky fix
                latent_model_input = torch.cat([x] * 2) if do_classifier_free_guidance else x
                latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5)

                t = int(sigma_to_t(sigma, log_sigmas))

                latent_model_input = latent_model_input.to(dtype=model.dtype)

                # predict the noise residual
                noise_pred = model(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                d = noise_pred.to(dtype=torch.float32)

                return d.detach(), zeros 

        x = x.to(dtype=torch.float32)
        x_min = x, x.new_zeros([x.shape[0]])
        t = x.new_tensor([sigma_min, sigma_max])

        ode_func = ODEfunc().cuda()

        method = ode_options['method']
        step_size = abs(sigma_min - sigma_max) / ode_options['num_inference_steps']
        atol = ode_options['atol']
        rtol = ode_options['rtol']

        sol = odeint(ode_func, x_min, t, atol=atol, rtol=rtol, method=method, options={'step_size': step_size})

        latent = sol[0][-1]
        return latent



    def get_scheduler_params(self, scheduler, num_inference_steps, device):
        beta_start = scheduler.config.beta_start
        beta_end = scheduler.config.beta_end
        num_train_timesteps = scheduler.config.num_train_timesteps

        betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32) ** 2

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        sigmas = np.array(((1 - alphas_cumprod)  / alphas_cumprod) ** 0.5)
        sigmas = np.concatenate([sigmas[::-1], [0,0]]).astype(np.float32)
        sigmas = torch.from_numpy(sigmas)

        init_sigma = (sigmas.max() ** 2 + 1) ** 0.5

        timesteps = np.linspace(0, num_train_timesteps - 1, num_train_timesteps, dtype=float)[::-1].copy()
        timesteps = torch.from_numpy(timesteps)

        step_ratio = num_train_timesteps // num_inference_steps
        timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.float32)
        timesteps += 1

        sigmas = np.array(((1 - alphas_cumprod) / alphas_cumprod) ** 0.5)
        log_sigmas = np.log(sigmas)
        sigmas = np.interp(timesteps, np.arange(0, len(sigmas)), sigmas)

        sigmas = np.concatenate([sigmas, [0.0]]).astype(np.float32)

        sigmas = torch.from_numpy(sigmas).to(device=device)

        log_sigmas = torch.from_numpy(log_sigmas).to(device=device)

        # scheduler.set_timesteps(num_inference_steps, device=device)
        # sigmas = scheduler.sigmas
        # log_sigmas = torch.log(sigmas)

        # print(sigmas)
        # print(log_sigmas)

        return sigmas[-2], sigmas[0], log_sigmas

    @torch.no_grad()
    def generate(self,
        prompts: List[str],
        height: int,
        width: int,
        guidance_scale: float,
        reconstruct_ode_options: Dict,
        num_inference_steps: int,
        generator,
        ):
        original_size = (height, width)
        target_size = (height, width)

        # define parameters for diffusion
        batch_size = len(prompts)
        guidance_scale = guidance_scale
        do_classifier_free_guidance = guidance_scale > 1.0

        # encode input prompt
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.pipe.encode_prompt(
            prompt=prompts,
            device=self.device,
            do_classifier_free_guidance=do_classifier_free_guidance,
        )

        # prepare timesteps
        self.pipe.scheduler.set_timesteps(num_inference_steps, device=self.device)

        # prepare latent variables
        num_channels_latents = self.pipe.unet.config.in_channels
        latents = self.pipe.prepare_latents(
            batch_size,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            self.device,
            generator
        )


        # prepare added time ids & embeddings
        add_text_embeds = pooled_prompt_embeds
        crops_coords_top_left = (0, 0)
        add_time_ids = self.pipe._get_add_time_ids(
            original_size, crops_coords_top_left, target_size, dtype=prompt_embeds.dtype
        )


        if do_classifier_free_guidance:
                    prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
                    add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
                    add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0)

        prompt_embeds = prompt_embeds.to(self.device)
        add_text_embeds = add_text_embeds.to(self.device)
        add_time_ids = add_time_ids.to(self.device).repeat(batch_size, 1)

        # prepare scheduler params
        sigma_min, sigma_max, log_sigmas = self.get_scheduler_params(self.pipe.scheduler, num_inference_steps, self.device)

        # prepare added cond kwargs
        added_cond_kwargs = {'text_embeds': add_text_embeds, 'time_ids': add_time_ids}

        with torch.no_grad():
            latents = self.ode_sample(
                self.pipe.unet,
                latents,
                sigma_max,
                sigma_min,
                log_sigmas,
                prompt_embeds,
                added_cond_kwargs,
                guidance_scale,
                reconstruct_ode_options,
            )

            # TODO: make sure this works in fp16
            vae = self.pipe.vae.to(dtype=torch.float32)
            latents = latents.to(next(iter(vae.post_quant_conv.parameters())).dtype)
            img = vae.decode(latents / vae.config.scaling_factor).sample
            img = self.pipe.image_processor.postprocess(img, output_type='pil')
            del vae    

        return img


    @torch.no_grad()
    def __call__(self,
        prompts: List[str],
        images: Image.Image,
        height: int,
        width: int,
        reconstruct: bool,
        guidance_scale: float,
        ll_guidance_scale: float,
        ll_ode_options: Dict,
        reconstruct_ode_options: Dict,
        num_inference_steps: int,
        generator,
        return_image=False,
        ):
        original_size = (height, width)
        target_size = (height, width)

        # define parameters for diffusion
        batch_size = len(prompts)
        guidance_scale = guidance_scale
        do_classifier_free_guidance = guidance_scale > 1.0

        # encode input prompt

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.pipe.encode_prompt(
            prompt=prompts,
            device=self.device,
            do_classifier_free_guidance=do_classifier_free_guidance,
        )

        # prepare timesteps
        self.pipe.scheduler.set_timesteps(num_inference_steps, device=self.device)

        with torch.no_grad():
            images = self.image_processor.preprocess(images, height, width)
            images = images.to(device=self.device, dtype=prompt_embeds.dtype)

            if prompt_embeds.dtype == torch.float16:
                # avoid overflow error
                images = images.float()
                self.pipe.vae = self.pipe.vae.to(dtype=torch.float32)

            latents = self.pipe.vae.encode(images).latent_dist.sample(generator)

            if prompt_embeds.dtype == torch.float16:
                self.pipe.vae = self.pipe.vae.to(dtype=torch.float16)
                latents = latents.to(dtype=torch.float16)

            init_latents = self.pipe.vae.config.scaling_factor * latents


        # shape = init_latents.shape
        # noise = randn_tensor(shape, generator=generator, device=self.device, dtype=dtype)

        # # get latents
        # init_latents = self.scheduler.add_noise(init_latents, noise, timestep)

        latents = init_latents

        # prepare added time ids & embeddings
        add_text_embeds = pooled_prompt_embeds
        crops_coords_top_left = (0, 0)
        add_time_ids = self.pipe._get_add_time_ids(
            original_size, crops_coords_top_left, target_size, dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=self.pipe.text_encoder_2.config.projection_dim
        )


        if do_classifier_free_guidance:
                    prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
                    add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
                    add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0)

        prompt_embeds = prompt_embeds.to(self.device)
        add_text_embeds = add_text_embeds.to(self.device)
        add_time_ids = add_time_ids.to(self.device).repeat(batch_size, 1)

        # prepare scheduler params
        sigma_min, sigma_max, log_sigmas = self.get_scheduler_params(self.pipe.scheduler, num_inference_steps, self.device)

        # prepare added cond kwargs
        added_cond_kwargs = {'text_embeds': add_text_embeds, 'time_ids': add_time_ids}

        # calculate log likelihood

        # repeat latents to batch size
        latents = latents[0].unsqueeze(0).expand(batch_size, -1, -1, -1)

        with torch.no_grad():
            if reconstruct:
                logger.info('reconstructing')
                latents = self.ode_sample(
                    self.pipe.unet,
                    latents,
                    sigma_min,
                    sigma_max,
                    log_sigmas,
                    prompt_embeds,
                    added_cond_kwargs,
                    guidance_scale,
                    reconstruct_ode_options,
                )
                latents = self.ode_sample(
                    self.pipe.unet,
                    latents,
                    sigma_max,
                    sigma_min,
                    log_sigmas,
                    prompt_embeds,
                    added_cond_kwargs,
                    guidance_scale,
                    reconstruct_ode_options,
                )

            ll_grid = torch.zeros((4, batch_size)).to(self.device)
            ll_guidance_scale = ll_guidance_scale

            for j in range(batch_size):
                logger.info(f'calculating log likelihood for prompt {j+1}/{batch_size}')
                if do_classifier_free_guidance:
                    prompt_embeds_cat = torch.cat([prompt_embeds[j].unsqueeze(0), prompt_embeds[batch_size+j].unsqueeze(0)], dim=0)
                    added_cond_kwargs = {"text_embeds": torch.cat([add_text_embeds[j].unsqueeze(0), add_text_embeds[batch_size+j].unsqueeze(0)], dim=0), 
                                        "time_ids": torch.cat([add_time_ids[j].unsqueeze(0), add_time_ids[batch_size+j].unsqueeze(0)], dim=0)}
                else:
                    prompt_embeds_cat = prompt_embeds[j].unsqueeze(0)
                    added_cond_kwargs = {"text_embeds": add_text_embeds[j].unsqueeze(0), "time_ids": add_time_ids[j].unsqueeze(0)}
                
                ll, prior, delta, nfev, _ = self.log_likelihood(self.pipe.unet, 
                                latents[j].unsqueeze(0), 
                                sigma_min, sigma_max, 
                                log_sigmas,
                                prompt_embeds_cat,
                                added_cond_kwargs,
                                ll_guidance_scale,
                                ll_ode_options)

                ll_grid[0, j] = ll
                ll_grid[1, j] = prior 
                ll_grid[2, j] = delta 
                ll_grid[3, j] = nfev

            output_data = {'log_likelihood': ll_grid[0], 'prior': ll_grid[1], 'delta': ll_grid[2], 'nfev': ll_grid[3]}

            if return_image:
                img = self.pipe.vae.decode(latents / self.pipe.vae.config.scaling_factor).sample
                img = self.pipe.image_processor.postprocess(img, output_type='pil')

                return img, output_data

            else:
                return output_data


