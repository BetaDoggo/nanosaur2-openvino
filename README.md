# nanosaur2-openvino

A super sloppy (I woke up and this repo appeared on my computer) conversion of the [Nanosaur2-670M](https://huggingface.co/well9472/Nanosaur2-670M) model into OpenVino for intel IGPUs (maybe arc works idk). I was visiting family and didn't have access to a real gpu. It works but I didn't read shit, so quality is not assured.


![sample](assets/sample.png)

*832×1216, 50 steps, CFG 4, seed 42*

The model's three components are reimplemented standalone based on the reference comfyui node.

| Component | Params | IR precision |
|---|---|---|
| Gemma3-270M text encoder (layer −2 + final norm) | 270M | fp32* |
| Nanosaur2 DiT — adaLN-single, 2D RoPE, QK-norm, SPRINT sparse path, x-prediction | 670M | fp16 |
| Semantic VAE decoder (txt2img) | ~60M | fp16 |

\* Gemma3's residual stream overflows in fp16, for bf16 or fp32 is required.

**Ready-made IR + tokenizer**: [HDiffusion/Nanosaur2-OpenVino](https://huggingface.co/HDiffusion/Nanosaur2-OpenVino)

## Setup

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt   # .venv/bin/pip on Linux/macOS

# option 1: download the pre-converted OpenVINO IR (~2.3 GB)
hf download HDiffusion/Nanosaur2-OpenVino \
  --local-dir models_ov --include "*.xml" --include "*.bin" --include "tokenizer.model"

# option 2: convert from the original safetensors yourself (needs ~5 GB RAM)
.venv/Scripts/pip install -r requirements-export.txt   # conversion extras
mkdir -p downloads
curl -L -o downloads/nanosaur2_diffusion_model.safetensors https://huggingface.co/well9472/Nanosaur2-670M/resolve/main/nanosaur2_diffusion_model.safetensors
curl -L -o downloads/nanosaur2_text_encoder.safetensors    https://huggingface.co/well9472/Nanosaur2-670M/resolve/main/nanosaur2_text_encoder.safetensors
curl -L -o downloads/nanosaur2_vae.safetensors             https://huggingface.co/well9472/Nanosaur2-670M/resolve/main/nanosaur2_vae.safetensors
.venv/Scripts/python export.py
```

## Usage

```
.venv/Scripts/python generate.py --prompt "newest, masterpiece, 1girl, solo, (fennec ears:1.3), long blonde wavy hair, blue eyes, big fluffy tail, smile, forest, sunlight" --width 832 --height 1216 --steps 50 --cfg 4 --seed 42 --device GPU
```

- Emphasis syntax `(text:1.3)` works like ComfyUI — weights become attention bias
  and pooling weights inside the DiT.
- `--device AUTO` (default) picks GPU → CPU.
- Keep total pixels near 1024² (buckets like 832×1216, 1024×1024, 1216×832);
  width/height must be multiples of 16.
- Defaults match the model card / reference workflow: Euler, `simple` scheduler,
  shift 3.0, 50 steps, CFG 4.

## Sampler (ComfyUI parity)

- Base sigmas: `sigma(t) = 3t / (1 + 2t)` (flow shift 3.0), t = k/1000
- `simple` scheduler: 50 evenly back-indexed sigmas from 1.0 → 0.0577, plus 0
- Euler: `x += v · (σ_next − σ)` with `v = (x − x0)/σ` from the DiT's x-prediction head
- CFG: `v = v_neg + (v_pos − v_neg) · 4`, both rows denoise the **same** latent
- Text is padded to 256 tokens; padding is masked (−∞ attention bias), so any
  prompt length runs through one compiled graph shape without recompiles

## Performance

Measured on a Dell Latitude 9440 (Core Ultra, Iris Xe iGPU):

| Resolution | Steps | Time |
|---|---|---|
| 512×512 | 50 | ~1.8 min (~2 s/step) |
| 832×1216 | 50 | ~10.5 min (~13 s/step) |

VAE decode adds 1–5 s. The first call per shape triggers one-time OpenVINO kernel
compilation (cached afterwards).

## Notes / limitations

- Text-to-image only: the VAE *encoder* (DINOv2-based, for img2img) is not ported.
- The loader's `alternate`/`path_drop` guidance modes (skipping the SPRINT sparse
  middle blocks on the uncond pass — a speed optimization) are not implemented;
  this port always runs the full model on both CFG halves (`cfg` mode).

## License

MIT. The converted weights inherit the base model's MIT license — see the
[HuggingFace repo](https://huggingface.co/HDiffusion/Nanosaur2-OpenVino).
