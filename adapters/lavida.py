"""Adapter for LaViDa-LLaDa (KonstantinosKK/lavida-llada-v1.0-instruct-hf-transformers).

LaViDa is a masked diffusion VLM. The HF-transformers-compatible checkpoint exposes
a custom AutoModelForCausalLM whose .generate() runs the LLaDa-style diffusion
sampler (llada_generate). It does NOT have a standard HF processor; we replicate
the small subset of llava/mm_utils utilities inline (anyres tiling + the
<image>-token splitting trick that maps to IMAGE_TOKEN_INDEX = -200).

Prompt format: the bundled tokenizer chat template is Llama-3 style. We insert
the literal string "<image>" into the user content; the helper
`tokenizer_image_token` then splits on it and stitches IMAGE_TOKEN_INDEX = -200
into input_ids so the model's prepare_inputs_labels_for_multimodal can splice in
image embeddings at the right offset.
"""
import math
from typing import List

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"


# --- llava/mm_utils helpers, inlined ---------------------------------------
def _select_best_resolution(original_size, possible_resolutions):
    ow, oh = original_size
    best_fit = None
    max_eff = 0
    min_waste = float("inf")
    for w, h in possible_resolutions:
        scale = min(w / ow, h / oh)
        dw, dh = int(ow * scale), int(oh * scale)
        eff = min(dw * dh, ow * oh)
        waste = (w * h) - eff
        if eff > max_eff or (eff == max_eff and waste < min_waste):
            max_eff = eff
            min_waste = waste
            best_fit = (w, h)
    return best_fit


def _resize_and_pad(image, target_resolution):
    ow, oh = image.size
    tw, th = target_resolution
    sw, sh = tw / ow, th / oh
    if sw < sh:
        nw = tw
        nh = min(math.ceil(oh * sw), th)
    else:
        nh = th
        nw = min(math.ceil(ow * sh), tw)
    resized = image.resize((nw, nh))
    out = Image.new("RGB", (tw, th), (0, 0, 0))
    out.paste(resized, ((tw - nw) // 2, (th - nh) // 2))
    return out


def _divide_to_patches(image, patch_size):
    patches = []
    w, h = image.size
    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            patches.append(image.crop((j, i, j + patch_size, i + patch_size)))
    return patches


def _proc_short_edge(image_processor):
    # SigLipImageProcessor exposes .size as a tuple (H, W) and .crop_size as a
    # dict {'height': H, 'width': W}; older Llava processors used shortest_edge.
    sz = image_processor.size
    if isinstance(sz, dict):
        return sz.get("shortest_edge") or sz.get("height")
    return sz[0]


def _process_anyres(image, image_processor, grid_pinpoints):
    # Pick best multi-tile resolution, pad+resize, divide into tiles, and
    # prepend a globally-resized thumbnail (matches llava process_anyres_image).
    possible = grid_pinpoints if isinstance(grid_pinpoints, list) else list(grid_pinpoints)
    best = _select_best_resolution(image.size, possible)
    padded = _resize_and_pad(image, best)
    cs = image_processor.crop_size if isinstance(image_processor.crop_size, dict) \
        else {"height": image_processor.crop_size[0]}
    tile = cs["height"]
    tiles = _divide_to_patches(padded, tile)
    shortest = _proc_short_edge(image_processor)
    thumb = image.resize((shortest, shortest))
    pieces = [thumb] + tiles
    proc = [image_processor.preprocess(p, return_tensors="pt")["pixel_values"][0] for p in pieces]
    return torch.stack(proc, dim=0)


def _tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX):
    chunks = [tokenizer(chunk).input_ids for chunk in prompt.split(DEFAULT_IMAGE_TOKEN)]

    def insert_sep(X, sep):
        return [e for pair in zip(X, [sep] * len(X)) for e in pair][:-1]

    input_ids = []
    offset = 0
    if chunks and chunks[0] and chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(chunks[0][0])
    for x in insert_sep(chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])
    return torch.tensor(input_ids, dtype=torch.long)
# ---------------------------------------------------------------------------


class Adapter:
    name = "LaViDa-LLaDa"
    model_id = "KonstantinosKK/lavida-llada-v1.0-instruct-hf-transformers"

    # Diffusion sampler knobs — kept short for MCQ (one letter answer).
    # gen_length and block_length must be equal (single block) and gen_length
    # must be divisible by num_blocks; steps must be a multiple of num_blocks.
    GEN_LENGTH = 16
    BLOCK_LENGTH = 16
    STEPS = 8

    def setup(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        ).to("cuda:0")
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.tie_weights()
        self.model.eval()
        self.image_processor = self.model.get_vision_tower().image_processor
        self.grid_pinpoints = self.model.config.image_grid_pinpoints
        self.device = next(self.model.parameters()).device

    @torch.no_grad()
    def infer(self, image_paths: List[str], question: str, choices: str) -> str:
        image = Image.open(image_paths[0]).convert("RGB")
        image_tensor = _process_anyres(image, self.image_processor, self.grid_pinpoints)
        image_tensor = image_tensor.to(dtype=torch.bfloat16, device=self.device)
        image_sizes = [image.size]

        text_blob = f"{question}\n{choices}\nAnswer with the letter only."
        user_content = f"{DEFAULT_IMAGE_TOKEN}\n{text_blob}"
        prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": user_content}],
            tokenize=False,
            add_generation_prompt=True,
        )
        input_ids = _tokenizer_image_token(prompt, self.tokenizer).unsqueeze(0).to(self.device)

        cont = self.model.generate(
            input_ids,
            images=[image_tensor],
            image_sizes=image_sizes,
            do_sample=False,
            temperature=0.0,
            max_new_tokens=self.GEN_LENGTH,
            block_length=self.BLOCK_LENGTH,
            steps=self.STEPS,
            tokenizer=self.tokenizer,
            prefix_lm=False,
            schedule="shift",
        )
        # llada_generate may return a tensor or (tensor, history)
        if isinstance(cont, tuple):
            cont = cont[0]
        text = self.tokenizer.batch_decode(cont, skip_special_tokens=True)[0]
        return text.lstrip("!").strip()
