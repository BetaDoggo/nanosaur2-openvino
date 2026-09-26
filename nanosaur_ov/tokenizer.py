"""Tokenizer for Nanosaur2: Gemma3 sentencepiece + ComfyUI-style emphasis parsing.

The serialized sentencepiece model lives in the `spiece_model` U8 blob inside
the original text encoder safetensors. Export it once to a standalone
`tokenizer.model` file (export.py does this automatically) for a self-contained
OpenVINO-only setup.
"""

import re
from pathlib import Path

import sentencepiece as spm
import torch


def extract_spiece_model(safetensors_path, out_path):
    """Pull the spiece_model blob out of the TE safetensors and save it."""
    from safetensors.torch import load_file
    sd = load_file(safetensors_path, device="cpu")
    proto = sd.pop("spiece_model").numpy().tobytes()
    Path(out_path).write_bytes(proto)
    return out_path


def load_spiece_model(path):
    path = Path(path)
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file
        sd = load_file(str(path), device="cpu")
        return spm.SentencePieceProcessor(model_proto=sd["spiece_model"].numpy().tobytes())
    return spm.SentencePieceProcessor(model_file=str(path))


def parse_prompt_emphasis(caption):
    """Strip "(text:weight)" groups; return plain text and (start, end, weight) spans.

    Direct port of nanosaur2_support/text_encoder.py.
    """
    weight_pattern = re.compile(r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)$")
    spans = []
    parts = []
    cursor = 0
    output_len = 0
    idx = 0
    while idx < len(caption):
        if caption[idx] != "(":
            idx += 1
            continue

        depth = 1
        end = idx + 1
        while end < len(caption) and depth > 0:
            if caption[end] == "(":
                depth += 1
            elif caption[end] == ")":
                depth -= 1
            end += 1

        if depth != 0:
            idx += 1
            continue

        inner = caption[idx + 1:end - 1]
        inner_depth = 0
        colon_idx = -1
        for inner_idx, char in enumerate(inner):
            if char == "(":
                inner_depth += 1
            elif char == ")":
                inner_depth -= 1
            elif char == ":" and inner_depth == 0:
                colon_idx = inner_idx

        emphasized_text = inner[:colon_idx]
        weight_text = inner[colon_idx + 1:].strip()
        if colon_idx == -1 or not emphasized_text or not weight_pattern.fullmatch(weight_text):
            idx += 1
            continue

        parts.append(caption[cursor:idx])
        output_len += idx - cursor
        parts.append(emphasized_text)
        spans.append((output_len, output_len + len(emphasized_text), float(weight_text)))
        output_len += len(emphasized_text)
        cursor = end
        idx = end

    parts.append(caption[cursor:])
    return "".join(parts), spans


class Nanosaur2Tokenizer:
    """Produces fixed-length (256) padded token ids, weights, and an attention mask."""

    def __init__(self, safetensors_path, max_length=256):
        self.spm = load_spiece_model(safetensors_path)
        self.max_length = max_length
        self.bos = 2  # Gemma <s>

    def tokenize(self, text):
        """Returns (ids, weights) lists trimmed to max_length (BOS included)."""
        text, spans = parse_prompt_emphasis(text)
        encoded = self.spm.encode(text, add_bos=False, add_eos=False, out_type="proto")
        tokens = [(piece.id, (piece.begin, piece.end)) for piece in encoded.pieces]
        ids = [self.bos]
        weights = [1.0]
        for token, (token_begin, token_end) in tokens:
            weight = 1.0
            for begin, end, span_weight in spans:
                if token_begin < end and token_end > begin:
                    weight *= span_weight
            ids.append(token)
            weights.append(weight)
            if len(ids) >= self.max_length:
                break
        return ids[:self.max_length], weights[:self.max_length]

    def encode_batch(self, texts):
        """Pad to (2, max_length): ids int64, mask float32, weights float32 (0 on padding)."""
        tokenized = [self.tokenize(t) for t in texts]
        n = self.max_length
        ids = torch.zeros(len(texts), n, dtype=torch.long)
        mask = torch.zeros(len(texts), n, dtype=torch.float32)
        weights = torch.zeros(len(texts), n, dtype=torch.float32)
        for i, (tid, tw) in enumerate(tokenized):
            ids[i, :len(tid)] = torch.tensor(tid, dtype=torch.long)
            mask[i, :len(tid)] = 1.0
            weights[i, :len(tw)] = torch.tensor(tw, dtype=torch.float32)
        return ids, mask, weights
