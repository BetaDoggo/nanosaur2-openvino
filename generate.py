"""Nanosaur2-670M text-to-image on OpenVINO.

Runs the full pipeline — Gemma3-270M text encoding, 670M DiT rectified-flow
sampling (euler / simple scheduler / CFG), and the Semantic VAE decoder —
on CPU, GPU (Intel Arc iGPU), or NPU.

Usage:
  python generate.py --prompt "newest, masterpiece, 1girl, solo, forest" \
      [--negative "..."] [--width 832 --height 1216] [--steps 50] [--cfg 4] \
      [--seed 42] [--device GPU] [--output out.png]

Notes:
  - Resolution buckets should keep total pixels near 1024^2 (e.g. 832x1216,
    1024x1024, 1216x832); width/height must be multiples of 16.
  - Emphasis syntax "(text:1.3)" is supported like ComfyUI.
  - Guidance uses plain CFG on the negative prompt (the loader's "cfg" mode).
"""

import argparse
import time
from pathlib import Path

import numpy as np
import openvino as ov
from PIL import Image

from nanosaur_ov.sampling import euler_cfg_sample, simple_scheduler_sigmas
from nanosaur_ov.tokenizer import Nanosaur2Tokenizer

DEFAULT_NEGATIVE = ("oldest, low quality, lowres, blurry, out of focus, jpeg artifacts, "
                    "watermark, signature, text, bad anatomy, deformed, extra limbs, "
                    "missing fingers, cropped")


def pick_device(core, requested):
    avail = core.available_devices
    if requested != "AUTO":
        return requested
    for dev in ("GPU", "NPU", "CPU"):
        if any(d.startswith(dev) for d in avail):
            return dev
    return "CPU"


class Nanosaur2Pipeline:
    def __init__(self, model_dir="models_ov", tokenizer_path=None, device="AUTO"):
        self.core = ov.Core()
        self.device = pick_device(self.core, device)
        model_dir = Path(model_dir)
        print(f"[pipeline] device: {self.device} (available: {self.core.available_devices})")

        for f in ("text_encoder.xml", "diffusion_model.xml", "vae_decoder.xml"):
            if not (model_dir / f).exists():
                raise FileNotFoundError(
                    f"{model_dir / f} not found. Download the OpenVINO IR (or run export.py "
                    f"on the original safetensors) — see the README.")

        tokenizer_path = tokenizer_path or model_dir / "tokenizer.model"
        self.te = self.core.compile_model(str(model_dir / "text_encoder.xml"), self.device)
        self.dit = self.core.compile_model(str(model_dir / "diffusion_model.xml"), self.device)
        self.vae = self.core.compile_model(str(model_dir / "vae_decoder.xml"), self.device)
        self.tokenizer = Nanosaur2Tokenizer(tokenizer_path)

    def encode_prompts(self, positive, negative):
        ids, mask, weights = self.tokenizer.encode_batch([positive, negative])
        out = self.te({"token_ids": ids.numpy(), "attn_mask": mask.numpy()})
        context = np.asarray(next(iter(out.values())))  # (2, 256, 640)
        return context.astype(np.float16), weights.numpy()

    @staticmethod
    def rope_np(head_dim, h, w):
        axis_dim = head_dim // 2
        inv = 1.0 / (10000.0 ** (np.arange(0, axis_dim, 2, dtype=np.float64) / axis_dim))
        y = np.arange(h, dtype=np.float64) - (h - 1) / 2
        x = np.arange(w, dtype=np.float64) - (w - 1) / 2
        yg, xg = np.meshgrid(y, x, indexing="ij")
        angles = np.concatenate([np.outer(xg.ravel(), inv), np.outer(yg.ravel(), inv)], axis=-1)
        return np.cos(angles).astype(np.float16), np.sin(angles).astype(np.float16)

    def generate(self, positive, negative=DEFAULT_NEGATIVE, width=832, height=1216,
                 steps=50, guidance=4.0, seed=42, verbose=True):
        assert width % 16 == 0 and height % 16 == 0, "width/height must be multiples of 16"
        context, token_weights = self.encode_prompts(positive, negative)

        lh, lw = height // 16, width // 16
        cos, sin = self.rope_np(96, lh, lw)

        t_start = time.time()
        x = euler_cfg_sample(
            lambda x, sigma: self._dit_call(x, sigma, context, token_weights, cos, sin),
            shape=(2, 64, lh, lw), seed=seed, steps=steps, guidance=guidance,
            callback=(lambda i, n, s: verbose and self._progress(i, n, t_start)),
        )
        if verbose:
            print(f"[sampler] {steps} steps in {time.time() - t_start:.1f}s")
        return self.decode(x[0])

    def _progress(self, i, n, t_start):
        done = (i + 1) / n
        rate = (i + 1) / (time.time() - t_start)
        print(f"\r[sampler] step {i + 1}/{n} ({rate:.2f} it/s)", end="", flush=True)
        if i + 1 == n:
            print()

    def _dit_call(self, x, sigma, context, token_weights, cos, sin):
        t = np.full((2,), sigma, dtype=np.float16)
        out = self.dit({
            "x": x, "timestep": t, "context": context,
            "token_weights": token_weights, "rope_cos": cos, "rope_sin": sin,
        })
        return np.asarray(next(iter(out.values())))

    def decode(self, latent):
        t0 = time.time()
        out = self.vae({"z": latent[None].astype(np.float16)})
        img = np.asarray(next(iter(out.values())))[0]  # (3, H, W) in [-1, 1]
        print(f"[vae] decoded in {time.time() - t0:.1f}s")
        img = ((img.astype(np.float32) + 1.0) * 0.5).clip(0.0, 1.0)
        return (img.transpose(1, 2, 0) * 255).round().astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(description="Nanosaur2-670M text-to-image (OpenVINO)")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--negative", default=DEFAULT_NEGATIVE)
    ap.add_argument("--width", type=int, default=832)
    ap.add_argument("--height", type=int, default=1216)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--cfg", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="AUTO", help="AUTO, CPU, GPU, or NPU")
    ap.add_argument("--models", default="models_ov")
    ap.add_argument("--tokenizer", default=None,
                    help="path to tokenizer.model or the TE safetensors (default: <models>/tokenizer.model)")
    ap.add_argument("--output", default="output.png")
    args = ap.parse_args()

    pipe = Nanosaur2Pipeline(args.models, args.tokenizer, args.device)
    img = pipe.generate(args.prompt, args.negative, args.width, args.height,
                        args.steps, args.cfg, args.seed)
    Image.fromarray(img).save(args.output)
    print(f"[done] saved {args.output} ({img.shape[1]}x{img.shape[0]})")


if __name__ == "__main__":
    main()
