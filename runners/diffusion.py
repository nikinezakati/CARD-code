import os
import logging
import time
import glob

import numpy as np
import tqdm
import torch
import torch.utils.data as data

from models.diffusion import Model
from datasets import get_dataset, data_transform, inverse_data_transform
from functions.ckpt_util import get_ckpt_path, download
from functions.denoising import efficient_generalized_steps
from functions.svd_replacement import H_functions
from whitened_functions.WhitenedH_denoise import *
from whitened_functions.WhitenedH_deblur import *
from whitened_functions.WhitenedH_sr import *


import torchvision.utils as tvu

from guided_diffusion.unet import UNetModel
from guided_diffusion.script_util import create_model, create_classifier, classifier_defaults, args_to_dict
import random
import torchvision.utils as tvu
import lpips
import piq
import matplotlib.pyplot as plt

patch_size = 8
dim = patch_size * patch_size


def gpu_mem(tag="", device=None, reset_peak=False, sync=True):
    if not torch.cuda.is_available():
        return
    if device is None:
        device = torch.device("cuda")
    if sync:
        torch.cuda.synchronize(device)
    if reset_peak:
        torch.cuda.reset_peak_memory_stats(device)

    alloc = torch.cuda.memory_allocated(device) / (1024**2)
    reserv = torch.cuda.memory_reserved(device) / (1024**2)
    peak  = torch.cuda.max_memory_allocated(device) / (1024**2)

    print(f"[GPU MEM] {tag:>28s} | alloc={alloc:8.1f} MB | reserved={reserv:8.1f} MB | peak={peak:8.1f} MB")



def perturb_cov_matrix(Sigma, noise_scale=0.05, eps=1e-9, seed=None):
    if seed is not None:
        np.random.seed(seed)
    S = 0.5 * (Sigma + Sigma.T)
    E = np.random.randn(*S.shape)
    N = 0.5 * (E + E.T)                         # symmetric noise
    S_tilde = S + noise_scale * N

    # Project to SPD by shifting spectrum if needed
    w, U = np.linalg.eigh(0.5 * (S_tilde + S_tilde.T))
    w = np.clip(w, a_min=eps - w.min() + eps, a_max=None)  # shift if any < eps
    Sigma_tilde = U @ np.diag(w) @ U.T

    # Whitener
    ew, Uw = np.linalg.eigh(Sigma_tilde)
    W_tilde = Uw @ np.diag(1.0 / np.sqrt(np.clip(ew, eps, None))) @ Uw.T
    return Sigma_tilde, W_tilde


def build_banded_covariance(
    size=64,
    var=2e-2,
    band_weights=None,   
    strength=0.99,       
    eps=1e-6,
    verbose=True,
    normalize=True,      
    auto_clip=True,     
    margin=1e-3,        
):
    if band_weights is None:
        band_weights = {1: 0.6, 8: 0.6}

    I = np.eye(size)
    B = np.zeros((size, size), dtype=float)

    for off, w in band_weights.items():
        if off <= 0 or off >= size or w == 0.0:
            continue
        B += w * (np.eye(size, k=off) + np.eye(size, k=-off))

    if np.allclose(B, 0):
        rho = 0.0
    else:
        eig = np.linalg.eigvalsh(B)
        rho = max(abs(eig[0]), abs(eig[-1]))

    if normalize:
        B_norm = B / (rho + 1e-12) if rho > 0 else B
        B_eff = strength * B_norm
        mode = "spectral_norm (pattern-only)"
    else:
        alpha_raw = strength
        if rho > 0:
            alpha_max = (1.0 - margin) / rho  
        else:
            alpha_max = np.inf
        if auto_clip and alpha_raw > alpha_max:
            alpha = alpha_max
            clipped = True
        else:
            alpha = alpha_raw
            clipped = (alpha_raw > alpha_max)
        B_eff = alpha * B
        mode = f"raw (weights amplify); alpha_used={alpha:.4g}" + (" [CLIPPED]" if clipped else "")

    Sigma = var * (I + B_eff) + eps * I

    if verbose:
        print(f"[Σ] var={var}, strength={strength}, normalize={normalize}, mode={mode}")
        print(f"    bands={list(band_weights.keys())}, weights={list(band_weights.values())}, rho(B)={rho:.4g}")

    return Sigma


band_weights = {1: 0.6, 2: 0.6, 8: 0.6, 16: 0.6}

strength=0.99
var=0.1


Sigma = build_banded_covariance(size=64, var=var, band_weights=band_weights,
                                strength=strength, normalize=False, auto_clip=True, verbose=True)

L = np.linalg.cholesky(Sigma)
Sigma_inv_sqrt = np.linalg.inv(L)


def add_correlated_noise_patchwise(img, L_torch, scale=1.0):
    B, C, H, W = img.shape
    out = img.clone()
    for b in range(B):
        for ch in range(C):
            for i in range(0, H, patch_size):
                for j in range(0, W, patch_size):
                    ph = min(patch_size, H - i)
                    pw = min(patch_size, W - j)
                    patch = img[b, ch, i:i+ph, j:j+pw]
                    padded = torch.zeros(patch_size, patch_size, device=img.device)
                    padded[:ph, :pw] = patch
                    flat = padded.reshape(-1)
                    noise = torch.randn(dim, device=img.device)
                    corr = scale * (L_torch @ noise)
                    flat_noisy = flat + corr
                    patch_noisy = flat_noisy.reshape(patch_size, patch_size)[:ph, :pw]
                    out[b, ch, i:i+ph, j:j+pw] = patch_noisy
    return out

def add_lr_correlated_noise_patchwise(img, L_lr, q, scale=1.0):
    """
    Add n ~ N(0, Σ_y) per LR q×q patch using a linear map L_lr such that
    L_lr L_lr^T = Σ_y (e.g., L_lr = W_patch^{-1}).
    """
    B, C, H, W = img.shape
    q2 = q * q
    unfold = torch.nn.Unfold(kernel_size=q, stride=q)
    fold   = torch.nn.Fold(output_size=(H, W), kernel_size=q, stride=q)
    with torch.no_grad():
        w = torch.clamp(fold(unfold(torch.ones_like(img))), min=1e-8)

    patches = unfold(img).view(B * C, q2, -1)                        # (BC, q2, N)
    z       = torch.randn_like(patches)                               # (BC, q2, N)
    corr    = torch.einsum('pq,bqn->bpn', L_lr.to(patches), z)        # (BC, q2, N)
    noisy   = patches + scale * corr
    out = fold(noisy.view(B, C * q2, -1)) / w
    return out



def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device
        self.L_torch = torch.from_numpy(L).float().to(self.device)
        self.Sigma_inv_sqrt_torch = torch.from_numpy(Sigma_inv_sqrt).float().to(self.device)
        self.lpips_fn = lpips.LPIPS(net='vgg').to(self.device)
        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        self.alphas_cumprod_prev = alphas_cumprod_prev
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
            # torch.cat(
            # [posterior_variance[1:2], betas[1:]], dim=0).log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

    def sample(self):
        cls_fn = None
        if self.config.model.type == 'simple':    
            model = Model(self.config)
            # This used the pretrained DDPM model, see https://github.com/pesser/pytorch_diffusion
            if self.config.data.dataset == "CIFAR10":
                name = "cifar10"
            elif self.config.data.dataset == "LSUN":
                name = f"lsun_{self.config.data.category}"
            elif self.config.data.dataset == 'CelebA_HQ':
                name = 'celeba_hq'
            else:
                raise ValueError
            if name != 'celeba_hq':
                ckpt = get_ckpt_path(f"ema_{name}", prefix=self.args.exp)
                print("Loading checkpoint {}".format(ckpt))
            elif name == 'celeba_hq':
                #ckpt = '~/.cache/diffusion_models_converted/celeba_hq.ckpt'
                ckpt = os.path.join(self.args.exp, "logs/celeba/celeba_hq.ckpt")
                if not os.path.exists(ckpt):
                    download('https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/celeba_hq.ckpt', ckpt)
            else:
                raise ValueError
            model.load_state_dict(torch.load(ckpt, map_location=self.device))
            model.to(self.device)
            model = torch.nn.DataParallel(model)

        elif self.config.model.type == 'openai':
            config_dict = vars(self.config.model)
            model = create_model(**config_dict)
            if self.config.model.use_fp16:
                model.convert_to_fp16()
            if self.config.model.class_cond:
                ckpt = os.path.join(self.args.exp, 'logs/imagenet/%dx%d_diffusion.pt' % (self.config.data.image_size, self.config.data.image_size))
                if not os.path.exists(ckpt):
                    download('https://openaipublic.blob.core.windows.net/diffusion/jul-2021/%dx%d_diffusion_uncond.pt' % (self.config.data.image_size, self.config.data.image_size), ckpt)
            else:
                ckpt = os.path.join(self.args.exp, "logs/imagenet/256x256_diffusion_uncond.pt")
                if not os.path.exists(ckpt):
                    download('https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt', ckpt)
                
            
            model.load_state_dict(torch.load(ckpt, map_location=self.device))
            model.to(self.device)
            model.eval()
            model = torch.nn.DataParallel(model)
            gpu_mem("after model load", self.device, reset_peak=True)


            if self.config.model.class_cond:
                ckpt = os.path.join(self.args.exp, 'logs/imagenet/%dx%d_classifier.pt' % (self.config.data.image_size, self.config.data.image_size))
                if not os.path.exists(ckpt):
                    image_size = self.config.data.image_size
                    download('https://openaipublic.blob.core.windows.net/diffusion/jul-2021/%dx%d_classifier.pt' % image_size, ckpt)
                classifier = create_classifier(**args_to_dict(self.config.classifier, classifier_defaults().keys()))
                classifier.load_state_dict(torch.load(ckpt, map_location=self.device))
                classifier.to(self.device)
                if self.config.classifier.classifier_use_fp16:
                    classifier.convert_to_fp16()
                classifier.eval()
                classifier = torch.nn.DataParallel(classifier)

                import torch.nn.functional as F
                def cond_fn(x, t, y):
                    with torch.enable_grad():
                        x_in = x.detach().requires_grad_(True)
                        logits = classifier(x_in, t)
                        log_probs = F.log_softmax(logits, dim=-1)
                        selected = log_probs[range(len(logits)), y.view(-1)]
                        return torch.autograd.grad(selected.sum(), x_in)[0] * self.config.classifier.classifier_scale
                cls_fn = cond_fn

        self.sample_sequence(model, cls_fn)

    def sample_sequence(self, model, cls_fn=None):
        start_time = time.time()
        args, config = self.args, self.config

        #get original images and corrupted y_0
        dataset, test_dataset = get_dataset(args, config)
        
        device_count = torch.cuda.device_count()
        
        if args.subset_start >= 0 and args.subset_end > 0:
            assert args.subset_end > args.subset_start
            test_dataset = torch.utils.data.Subset(test_dataset, range(args.subset_start, args.subset_end))
        else:
            args.subset_start = 0
            args.subset_end = len(test_dataset)

        print(f'Dataset has size {len(test_dataset)}')    
        
        def seed_worker(worker_id):
            worker_seed = args.seed % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        g = torch.Generator()
        g.manual_seed(args.seed)
        val_loader = data.DataLoader(
            test_dataset,
            batch_size=config.sampling.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
            worker_init_fn=seed_worker,
            generator=g,
        )

        if torch.cuda.is_available():
            gpu_mem("START sample_sequence (before reset)", self.device)
            gpu_mem("RESET PEAK", self.device, reset_peak=True)

        

        ## get degradation matrix ##
        deg = args.deg
        H_funcs = None
        if deg[:2] == 'cs':
            compress_by = int(deg[2:])
            from functions.svd_replacement import WalshHadamardCS
            H_funcs = WalshHadamardCS(config.data.channels, self.config.data.image_size, compress_by, torch.randperm(self.config.data.image_size**2, device=self.device), self.device)
        elif deg[:3] == 'inp':
            from functions.svd_replacement import Inpainting
            if deg == 'inp_lolcat':
                loaded = np.load("inp_masks/lolcat_extra.npy")
                mask = torch.from_numpy(loaded).to(self.device).reshape(-1)
                missing_r = torch.nonzero(mask == 0).long().reshape(-1) * 3
            elif deg == 'inp_lorem':
                loaded = np.load("inp_masks/lorem3.npy")
                mask = torch.from_numpy(loaded).to(self.device).reshape(-1)
                missing_r = torch.nonzero(mask == 0).long().reshape(-1) * 3
            else:
                missing_r = torch.randperm(config.data.image_size**2)[:config.data.image_size**2 // 2].to(self.device).long() * 3
            missing_g = missing_r + 1
            missing_b = missing_g + 1
            missing = torch.cat([missing_r, missing_g, missing_b], dim=0)
            H_funcs = Inpainting(config.data.channels, config.data.image_size, missing, self.device)
        elif deg == 'deno':
            from functions.svd_replacement import Denoising
            H_funcs = Denoising(config.data.channels, self.config.data.image_size, self.device)
        elif deg[:10] == 'sr_bicubic':
            factor = int(deg[10:])
            from functions.svd_replacement import SRConv
            def bicubic_kernel(x, a=-0.5):
                if abs(x) <= 1:
                    return (a + 2)*abs(x)**3 - (a + 3)*abs(x)**2 + 1
                elif 1 < abs(x) and abs(x) < 2:
                    return a*abs(x)**3 - 5*a*abs(x)**2 + 8*a*abs(x) - 4*a
                else:
                    return 0
            k = np.zeros((factor * 4))
            for i in range(factor * 4):
                x = (1/factor)*(i- np.floor(factor*4/2) +0.5)
                k[i] = bicubic_kernel(x)
            k = k / np.sum(k)
            kernel = torch.from_numpy(k).float().to(self.device)
            H_funcs = SRConv(kernel / kernel.sum(), \
                             config.data.channels, self.config.data.image_size, self.device, stride = factor)
        elif deg == 'deblur_uni':
            from functions.svd_replacement import Deblurring
            H_funcs = Deblurring(torch.Tensor([1/9] * 9).to(self.device), config.data.channels, self.config.data.image_size, self.device)
        elif deg == 'deblur_gauss':
            from functions.svd_replacement import Deblurring
            sigma = 10
            pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x/sigma)**2]))
            kernel = torch.Tensor([pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2)]).to(self.device)
            H_funcs = Deblurring(kernel / kernel.sum(), config.data.channels, self.config.data.image_size, self.device)
        elif deg == 'deblur_aniso':
            from functions.svd_replacement import Deblurring2D
            sigma = 20
            pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x/sigma)**2]))
            kernel2 = torch.Tensor([pdf(-4), pdf(-3), pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2), pdf(3), pdf(4)]).to(self.device)
            sigma = 1
            pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x/sigma)**2]))
            kernel1 = torch.Tensor([pdf(-4), pdf(-3), pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2), pdf(3), pdf(4)]).to(self.device)
            H_funcs = Deblurring2D(kernel1 / kernel1.sum(), kernel2 / kernel2.sum(), config.data.channels, self.config.data.image_size, self.device)
        elif deg[:2] == 'sr':
            blur_by = int(deg[2:])
            from functions.svd_replacement import SuperResolution
            H_funcs = SuperResolution(config.data.channels, config.data.image_size, blur_by, self.device)
        elif deg == 'color':
            from functions.svd_replacement import Colorization
            H_funcs = Colorization(config.data.image_size, self.device)
        else:
            print("ERROR: degradation type not supported")
            quit()
        args.sigma_0 = 2 * args.sigma_0 #to account for scaling to [-1,1]
        sigma_0 = args.sigma_0
        
        print(f'Start from {args.subset_start}')
        idx_init = args.subset_start
        idx_so_far = args.subset_start
        avg_psnr = 0.0
        avg_ssim = 0.0
        avg_lpips = 0.0
        pbar = tqdm.tqdm(val_loader)

        orig_H = H_funcs
        if deg == 'deblur_aniso' or deg == 'deblur_gauss' or deg == 'deblur_uni':
            H_Whitened = WhitenedH_deblur(
                        orig_H, self.Sigma_inv_sqrt_torch,
                        channels=config.data.channels,
                        height=config.data.image_size,
                        width=config.data.image_size,
                        patch_size=patch_size,
                        stride=patch_size,
                        device=self.device,
                        mc_samples=64, mc_batch=16
                    )
        elif deg[:2] == 'sr':
            H_Whitened = WhitenedH_sr(orig_H, self.Sigma_inv_sqrt_torch,
                    channels=config.data.channels,
                    height=config.data.image_size,
                    width=config.data.image_size,
                    patch_size=patch_size,
                    stride=patch_size,           
                    device=self.device
                )
        else:
            H_Whitened = WhitenedH_deno(orig_H, self.Sigma_inv_sqrt_torch,
                    channels=config.data.channels,
                    height=config.data.image_size,
                    width=config.data.image_size,
                    patch_size=patch_size,
                    stride=patch_size,           
                    device=self.device
                )
        if args.noise_model == 'corr_ddrm':
            H_funcs = H_Whitened
        else:
            H_funcs = orig_H

        for x_orig, classes in pbar:
            x_orig = x_orig.to(self.device)
            x_orig = data_transform(self.config, x_orig)
            
            if deg[:2] == 'sr' or deg == 'deblur_aniso' or deg == 'deblur_gauss' or deg == 'deblur_uni':
                B = x_orig.shape[0]
                C = config.data.channels
                H = config.data.image_size
                W = H
                if hasattr(H_funcs, "hL") and hasattr(H_funcs, "wL"):
                    yH, yW = int(H_funcs.hL), int(H_funcs.wL)
                else:
                    xlen  = C * H * W
                    probe = torch.zeros(1, xlen, device=self.device, dtype=x_orig.dtype)
                    ylen  = int(orig_H.H(probe).shape[1])  
                    if ylen == xlen:
                        yH, yW = H, W
                    else:
                        per_ch = ylen // C
                        side   = int(round(per_ch ** 0.5))
                        yH, yW = side, side

            if args.noise_setting == 'correlated' and args.noise_model == 'corr_ddrm':
                if deg[:2] == 'sr':
                    y_clean = orig_H.H(x_orig)                                   
                    y_clean_img = y_clean.view(B, C, yH, yW)
                    y_corr_img = add_lr_correlated_noise_patchwise(
                        y_clean_img, H_funcs.Wp_patch, H_funcs.p_out, scale=0.0
                    )
                    y_0 = H_funcs.H(y_corr_img)
                 
                elif deg == 'deblur_aniso' or deg == 'deblur_gauss' or deg == 'deblur_uni':
                    y_clean = orig_H.H(x_orig)                                   
                    y_clean_img = y_clean.view(B, C, H, W)
                    y_corr_img = add_correlated_noise_patchwise(
                        y_clean_img, H_funcs.Wp_patch, scale=0.0
                    )
                    y_0_img = H_funcs.whiten(y_corr_img)
                    y_0 = y_0_img.view(B, -1)

                else:
                    y_0 = orig_H.H(x_orig) 
                    y_0_img = y_0.view(x_orig.shape) 
                    y_corr_img = add_correlated_noise_patchwise(y_0_img, self.L_torch_added, scale=0.2) 
                    y_corr_img = y_corr_img.view(x_orig.shape[0], -1)
                    y_0 = H_funcs.H(y_corr_img)

            elif args.noise_setting == 'correlated' and args.noise_model == 'ddrm':
                if deg[:2] == 'sr':
                    y_clean = orig_H.H(x_orig)                                   
                    y_clean_img = y_clean.view(B, C, yH, yW)
                    y_corr_img = add_lr_correlated_noise_patchwise(
                        y_clean_img, H_Whitened.Wp_patch, H_Whitened.p_out, scale=0.0
                    )
                    y_0 = y_corr_img.view(x_orig.shape[0], -1)
                elif deg == 'deblur_aniso' or deg == 'deblur_gauss' or deg == 'deblur_uni':
                    y_clean = orig_H.H(x_orig)                                   
                    y_clean_img = y_clean.view(B, C, H, W)
                    y_corr_img = add_correlated_noise_patchwise(
                        y_clean_img, H_Whitened.Wp_patch, scale=0.0
                    )
                    y_0 = y_corr_img.view(B, -1)
                   
                else:
                    y_0 = orig_H.H(x_orig)
                    y_0_img = y_0.view_as(x_orig)            
                    y_corr_img = add_correlated_noise_patchwise(y_0_img, self.L_torch, scale=sigma_0)   
                    y_0 = y_corr_img.view(x_orig.shape[0], -1)

            else:
                y_0 = H_funcs.H(x_orig)
                y_0 = y_0 + sigma_0 * torch.randn_like(y_0)

            pinv_y_0 = H_funcs.H_pinv(y_0).view(y_0.shape[0], config.data.channels, self.config.data.image_size, self.config.data.image_size)
            if deg[:6] == 'deblur':
                if args.noise_model == 'corr_ddrm':
                    pinv_y_0 = y_corr_img
                else:
                    pinv_y_0 = y_0.view(y_0.shape[0], config.data.channels, self.config.data.image_size, self.config.data.image_size)
            elif deg == 'color': pinv_y_0 = y_0.view(y_0.shape[0], 1, self.config.data.image_size, self.config.data.image_size).repeat(1, 3, 1, 1)
            elif deg[:3] == 'inp': pinv_y_0 += H_funcs.H_pinv(H_funcs.H(torch.ones_like(pinv_y_0))).reshape(*pinv_y_0.shape) - 1

            for i in range(len(pinv_y_0)):
                tvu.save_image(
                    inverse_data_transform(config, pinv_y_0[i]), os.path.join(self.args.image_folder, f"y0_{idx_so_far + i}.png")
                )
                tvu.save_image(
                    inverse_data_transform(config, x_orig[i]), os.path.join(self.args.image_folder, f"orig_{idx_so_far + i}.png")
                )

            ##Begin DDIM
            x = torch.randn(
                y_0.shape[0],
                config.data.channels,
                config.data.image_size,
                config.data.image_size,
                device=self.device,
            )

            with torch.no_grad():
                x, _ = self.sample_image(x, model, H_funcs, y_0, sigma_0, last=False, cls_fn=cls_fn, classes=classes)

            x = [inverse_data_transform(config, y) for y in x]

            for i in [-1]: 
                for j in range(x[i].size(0)):
                    tvu.save_image(
                        x[i][j], os.path.join(self.args.image_folder, f"{idx_so_far + j}_{i}.png")
                    )
                    if i == len(x)-1 or i == -1:
                        orig = inverse_data_transform(config, x_orig[j])
                        mse = torch.mean((x[i][j].to(self.device) - orig) ** 2)
                        psnr = 10 * torch.log10(1 / mse)
                        avg_psnr += psnr

                        # Prepare tensors: shape [1,C,H,W], normalized to [0,1] for SSIM/LPIPS
                        recon = x[i][j].unsqueeze(0).to(self.device)
                        target = orig.unsqueeze(0).to(self.device)
                        recon_01 = (recon + 1) / 2  # [-1,1] to [0,1]
                        target_01 = (target + 1) / 2

                        # SSIM (expect values in [0,1])
                        ssim_val = piq.ssim(recon_01, target_01, data_range=1.).item()
                        avg_ssim += ssim_val

                        # LPIPS (lower is better, usually in [0,1])
                        lpips_val = self.lpips_fn(recon_01, target_01)
                        avg_lpips += lpips_val.item()


            idx_so_far += y_0.shape[0]

            pbar.set_description("PSNR: %.2f" % (avg_psnr / (idx_so_far - idx_init)))

        avg_psnr = avg_psnr / (idx_so_far - idx_init)
        print("Total Average PSNR: %.2f" % avg_psnr)
        print("Number of samples: %d" % (idx_so_far - idx_init))

        n_samples = (idx_so_far - idx_init)
        avg_ssim = avg_ssim / n_samples
        avg_lpips = avg_lpips / n_samples

        print("Total Average SSIM: %.4f" % avg_ssim)
        print("Total Average LPIPS: %.4f" % avg_lpips)
        print("Number of samples: %d" % n_samples)

        end_time = time.time()
        total_sec = end_time - start_time
        total_min = total_sec / 60
        total_hr  = total_sec / 3600

        print("\n==================== Timing Summary ====================")
        print(f"Total time: {total_sec:.2f} seconds")
        print(f"Total time: {total_min:.2f} minutes")
        print(f"Total time: {total_hr:.2f} hours")
        print("=======================================================\n")


        if torch.cuda.is_available():
            gpu_mem("END sample_sequence", self.device)


    def sample_image(self, x, model, H_funcs, y_0, sigma_0, last=True, cls_fn=None, classes=None):
        skip = self.num_timesteps // self.args.timesteps
        seq = range(0, self.num_timesteps, skip)
        
        x = efficient_generalized_steps(x, seq, model, self.betas, H_funcs, y_0, sigma_0, \
            etaB=self.args.etaB, etaA=self.args.eta, etaC=self.args.eta, cls_fn=cls_fn, classes=classes, deg=self.args.deg)
        if last:
            x = x[0][-1]
        return x