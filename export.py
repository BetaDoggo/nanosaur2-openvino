"""Export the Nanosaur2-670M pipeline to OpenVINO IR (fp16).

Loads the three safetensors checkpoints, remaps them onto the standalone
PyTorch modules in nanosaur_ov/models.py, sanity-checks each forward pass,
then exports via torch.export -> ov.convert_model with dynamic spatial
shapes, and validates OV output against the PyTorch fp16 reference.

Usage:  python export.py [--outdir models_ov] [--downloads downloads]
"""

import argparse
import gc
import time
from pathlib import Path

import numpy as np
import torch
import openvino as ov
from torch.export import Dim
from safetensors.torch import load_file

from nanosaur_ov import models
from nanosaur_ov.tokenizer import extract_spiece_model


# ---------------------------------------------------------------------------
# Weight remapping: checkpoint key -> nanosaur_ov.models key
# ---------------------------------------------------------------------------

def remap_dit(sd):
    out = {}
    for k, v in sd.items():
        if k.startswith("shared_encoder_adaLN.0."):
            k = "shared_encoder_adaLN." + k[len("shared_encoder_adaLN.0."):]
        elif ".adaLN_modulation.0." in k:  # refine blocks use nn.Sequential(Linear)
            k = k.replace(".adaLN_modulation.0.", ".adaLN_modulation.")
        elif k == "y_embedder.norm.weight":
            k = "y_embedder.norm_weight"
        elif ".norm1.weight" in k or ".norm2.weight" in k:
            k = k[:-len(".weight")]
        elif ".attn.q_norm.weight" in k or ".attn.k_norm.weight" in k:
            k = k[:-len(".weight")]
        out[k] = v
    return out


def remap_te(sd):
    sd = {k: v for k, v in sd.items() if k != "spiece_model"}
    out = {}
    for k, v in sd.items():
        k = k[len("model."):] if k.startswith("model.") else k
        if k == "norm.weight":
            k = "norm"
        elif k.endswith("layernorm.weight"):
            k = k[:-len(".weight")]
        elif k.endswith("_norm.weight"):  # self_attn q_norm/k_norm
            k = k[:-len(".weight")]
        out[k] = v
    return out


def remap_vae(sd):
    # only the decoder (+ latent stats) is needed for text-to-image; the
    # DINOv2 encoder half of the checkpoint is intentionally dropped
    out = {"latent_mean": sd["latent_mean"].float(), "latent_std": sd["latent_std"].float()}
    for k, v in sd.items():
        if not k.startswith("decoder."):
            continue
        k = k[len("decoder."):]
        if k.startswith("mid.block_1."):
            k = "mid_block_1." + k[len("mid.block_1."):]
        elif k.startswith("mid.attn_1."):
            k = "mid_attn_1." + k[len("mid.attn_1."):]
        elif k.startswith("mid.block_2."):
            k = "mid_block_2." + k[len("mid.block_2."):]
        elif k.startswith("up."):
            rest = k[len("up."):]
            level, tail = rest.split(".", 1)
            if tail.startswith("block."):
                tail = "blocks." + tail[len("block."):]
            elif tail.startswith("upsample."):
                tail = "up." + tail[len("upsample."):]
            k = f"levels.{4 - int(level)}.{tail}"
        out[k] = v
    return out


def to_fp16(module):
    for p in module.parameters():
        p.data = p.data.to(torch.float16)
        assert torch.isfinite(p.data).all(), "weight overflow in bf16 -> fp16 conversion"
    return module.eval()


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def export_and_check(name, module, args, dynamic_shapes, out_path, inputs_np, ref_out):
    print(f"[{name}] exporting...")
    t0 = time.time()
    with torch.no_grad():
        ep = torch.export.export(module, args=args, dynamic_shapes=dynamic_shapes)
    ovm = ov.convert_model(ep)
    print(f"[{name}] converted in {time.time() - t0:.1f}s")
    core = ov.Core()
    compiled = core.compile_model(ovm, "CPU")
    got = compiled(inputs_np)
    got = np.concatenate([np.asarray(got[o]) for o in compiled.outputs], axis=0)
    got = got.reshape(ref_out.shape)
    ref = ref_out.numpy().astype(np.float32)
    diff = np.abs(got.astype(np.float32) - ref)
    denom = np.maximum(np.abs(ref), 1e-3)
    rel_l2 = np.linalg.norm(diff) / max(np.linalg.norm(ref), 1e-9)
    print(f"[{name}] parity vs torch ref: max_abs={diff.max():.3e} "
          f"max_rel={(diff / denom).max():.3e} rel_l2={rel_l2:.3e}")
    assert np.isfinite(got).all(), f"{name}: non-finite OV output"
    assert rel_l2 < 0.02, f"{name}: OV output deviates from torch reference (rel_l2={rel_l2:.3e})"
    ov.serialize(ovm, str(out_path))
    print(f"[{name}] saved {out_path} ({out_path.stat().st_size / 1e6:.0f} MB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--downloads", default="downloads")
    ap.add_argument("--outdir", default="models_ov")
    ap.add_argument("--skip-dit", action="store_true")
    ap.add_argument("--skip-te", action="store_true")
    ap.add_argument("--skip-vae", action="store_true")
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)
    dl = Path(args.downloads)
    torch.manual_seed(0)

    # standalone tokenizer file so inference needs no safetensors access
    extract_spiece_model(dl / "nanosaur2_text_encoder.safetensors", outdir / "tokenizer.model")
    print(f"[tokenizer] wrote {outdir / 'tokenizer.model'}")

    # ---------------- text encoder ----------------
    # NOTE: exported in fp32. Gemma3's residual stream reaches magnitudes ~1e5
    # (ComfyUI runs it in bf16/fp32 for the same reason), which overflows fp16.
    if not args.skip_te:
        te = models.Gemma3TextEncoder(models.GEMMA3_270M_CFG, seq_len=256)
        sd = remap_te(load_file(dl / "nanosaur2_text_encoder.safetensors"))
        missing, unexpected = te.load_state_dict(sd, strict=False, assign=True)
        missing = [m for m in missing if "inv_freq" not in m and "positions" not in m and "causal" not in m]
        assert not missing and not unexpected, f"TE key mismatch: {missing[:5]} / {unexpected[:5]}"
        for p in te.parameters():
            p.data = p.data.to(torch.float32)
        te.eval()
        ids = torch.randint(0, 260000, (2, 256), dtype=torch.long)
        mask = torch.zeros(2, 256)
        mask[0, :30] = 1
        mask[1, :18] = 1
        with torch.no_grad():
            ref = te(ids, mask)
        print(f"[te] torch out {tuple(ref.shape)} finite={torch.isfinite(ref).all().item()} "
              f"absmax={ref.abs().max():.1f}")
        export_and_check(
            "te", te, (ids, mask), None, outdir / "text_encoder.xml",
            {"token_ids": ids.numpy(), "attn_mask": mask.numpy()}, ref,
        )
        del te, sd, ref
        gc.collect()

    # ---------------- diffusion transformer ----------------
    if not args.skip_dit:
        dit = models.Nanosaur2DiT()
        sd = remap_dit(load_file(dl / "nanosaur2_diffusion_model.safetensors"))
        missing, unexpected = dit.load_state_dict(sd, strict=True, assign=True)
        to_fp16(dit)
        H, W = 16, 24  # small example for export; real shapes at runtime
        cos, sin = models.build_rope(dit.head_dim, H, W, "cpu")
        x = torch.randn(2, 64, H, W, dtype=torch.float16)
        t = torch.tensor([0.8, 0.8], dtype=torch.float16)
        ctx = torch.randn(2, 256, 640, dtype=torch.float16) * 0.1
        tw = torch.ones(2, 256)
        tw[0, 40:] = 0
        tw[1, 20:] = 0
        with torch.no_grad():
            ref = dit(x, t, ctx, tw, cos.to(torch.float16), sin.to(torch.float16))
        print(f"[dit] torch out {tuple(ref.shape)} finite={torch.isfinite(ref).all().item()} std={ref.float().std():.3e}")
        assert torch.isfinite(ref).all(), "dit: non-finite torch output (fp16 overflow?)"
        n_dim = Dim("n")
        ds = (
            {2: Dim("h"), 3: Dim("w")},   # x
            None,                          # timestep
            None,                          # context
            None,                          # token_weights
            {0: n_dim},                    # rope_cos
            {0: n_dim},                    # rope_sin
        )
        export_and_check(
            "dit", dit, (x, t, ctx, tw, cos.to(torch.float16), sin.to(torch.float16)), ds,
            outdir / "diffusion_model.xml",
            {"x": x.numpy(), "timestep": t.numpy(), "context": ctx.numpy(), "token_weights": tw.numpy(),
             "rope_cos": cos.to(torch.float16).numpy(), "rope_sin": sin.to(torch.float16).numpy()},
            ref,
        )
        del dit, sd, ref
        gc.collect()

    # ---------------- VAE decoder ----------------
    if not args.skip_vae:
        vae = models.Nanosaur2VaeDecoder()
        sd = remap_vae(load_file(dl / "nanosaur2_vae.safetensors"))
        missing, unexpected = vae.load_state_dict(sd, strict=False, assign=True)
        missing = [m for m in missing if "latent_mean" not in m and "latent_std" not in m]
        assert not missing and not unexpected, f"VAE key mismatch: {missing[:5]} / {unexpected[:5]}"
        for p in vae.parameters():
            p.data = p.data.to(torch.float32)
        vae.eval()
        z = torch.randn(1, 64, 16, 24) * 0.5
        with torch.no_grad():
            # fp32 reference (fp16 CPU convs crawl on machines without fp16 SIMD)
            ref = vae(z)
        print(f"[vae] torch out {tuple(ref.shape)} finite={torch.isfinite(ref).all().item()} "
              f"range=[{ref.min():.3f},{ref.max():.3f}]")
        to_fp16(vae)
        z16 = z.to(torch.float16)
        ds = ({2: Dim("h"), 3: Dim("w")},)
        export_and_check(
            "vae", vae, (z16,), ds, outdir / "vae_decoder.xml",
            {"z": z16.numpy()}, ref,
        )
        del vae, sd, ref
        gc.collect()

    print("done.")


if __name__ == "__main__":
    main()
