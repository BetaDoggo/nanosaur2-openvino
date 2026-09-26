"""Sampling for Nanosaur2 rectified-flow: ComfyUI euler + simple scheduler, shift 3.0.

ComfyUI math (comfy/model_sampling.py + comfy/samplers.py):
  - base sigmas: sigma(t) = shift * t / (1 + (shift - 1) * t), t = 1..1000 / 1000
  - "simple" scheduler: pick steps evenly from the END of the base table, append 0
  - euler step: x += v * (sigma_next - sigma), v = model velocity output
  - CFG: v = v_uncond + (v_cond - v_uncond) * guidance
"""

import numpy as np


def base_sigmas(shift=3.0, timesteps=1000):
    t = np.arange(1, timesteps + 1, dtype=np.float64) / timesteps
    return shift * t / (1.0 + (shift - 1.0) * t)


def simple_scheduler_sigmas(steps=50, shift=3.0):
    base = base_sigmas(shift)
    ss = len(base) / steps
    sigs = [base[-(1 + int(x * ss))] for x in range(steps)]
    return np.array(sigs + [0.0], dtype=np.float64)


def initial_noise(shape, seed, dtype=np.float16):
    """ComfyUI-style CPU generator noise; sigma_max = 1.0 so x0 = noise.

    The leading batch dim of 2 (cond/uncond) is filled with the SAME noise:
    CFG requires both rows to denoise the identical latent.
    """
    import torch
    g = torch.Generator("cpu").manual_seed(seed)
    single = (1,) + tuple(shape[1:])
    n = torch.randn(single, generator=g, dtype=torch.float32).numpy().astype(dtype)
    return np.repeat(n, shape[0], axis=0)


def euler_cfg_sample(dit_call, shape, seed, steps=50, guidance=4.0, dtype=np.float16, callback=None):
    """dit_call(x, sigma) -> (v_cond, v_uncond) already CFG-batched model output.

    x shape (2, C, H, W): row 0 = cond, row 1 = uncond.
    """
    sigmas = simple_scheduler_sigmas(steps)
    x = initial_noise(shape, seed, dtype)
    for i in range(len(sigmas) - 1):
        sigma = sigmas[i]
        v = dit_call(x.astype(dtype), sigma)
        v_cond, v_uncond = v[0], v[1]
        v_cfg = v_uncond + (v_cond - v_uncond) * guidance
        dt = sigmas[i + 1] - sigma
        x = x + v_cfg.astype(np.float32) * dt
        if callback is not None:
            callback(i, len(sigmas) - 1, float(sigma))
    return x
