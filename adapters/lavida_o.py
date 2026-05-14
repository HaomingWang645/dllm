"""Adapter for LaViDa-O (jacklishufan/LaViDa-O-v1.0).

LaViDa-O is an 11B unified masked-diffusion VLM (Adobe Research) that supports
both multi-modal understanding and image generation/editing. For
ViewSpatial-Bench we only need the understanding path (text answer to an
image + question); for VSI-Bench we feed N uniformly-sampled video frames
through the same LLaVA-NeXT-style vision tower using the model's video
modality path (single `<image>` placeholder, `(T,3,H,W)` pixel tensor,
`modalities=["video"]`).

This adapter follows the inference recipe shown in the official Demo.ipynb
"Image Understanding" cell: build_model -> process_images -> conversation
template ("llada") -> tokenizer_image_token -> model.generate(...) with the
LLaDa-style masked-diffusion sampler (step_ratio=1, prefix_lm=True). The
video path uses the same conv template but bypasses anyres tiling (datasets.py
in the LaViDa-O repo treats videos the same way: a single batched preprocess
across all frames, one <image> token in the prompt, modalities=["video"], and
the model handles 2D spatial pooling internally).

Requires the local editable install of the llava package under
repos/LaVida-O/. We add that path to sys.path so the adapter works regardless
of the CWD that eval_vsb.py / eval_vsi.py is launched from.
"""
import copy
import os
import sys
from pathlib import Path
from typing import List, Optional

# These env flags match the official Demo.ipynb. They control attention
# padding behavior inside the model code; without them generate() may
# choke on prefix-LM inputs.
os.environ.setdefault("DEBUG_FIX_PADDING", "1")
os.environ.setdefault("NOT_ALWASY_DO_2DPOOL", "1")

# Editable install lives here; make sure llava can be imported from anywhere.
_LAVIDA_O_REPO = Path("/home/haoming/dllm/repos/LaVida-O")
if str(_LAVIDA_O_REPO) not in sys.path:
    sys.path.insert(0, str(_LAVIDA_O_REPO))

import math
import types

import torch
import torch.nn as nn
from PIL import Image

from llava.constants import IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.eval.predict_t2i_edit import build_model
from llava.mm_utils import process_images, resize_and_center_crop, tokenizer_image_token

from adapters.video_utils import sample_frames


def _patched_get_2dPool(self, image_feature, stride=2):
    """Replacement for llava_arch.LlavaMetaForCausalLM.get_2dPool.

    The official implementation reads `num_patches_per_side` from the vision
    tower config (27 for siglip-so400m-patch14-384). But LaViDa-O ships with a
    SpatialPool vision resampler (stride=2, conv mode) that already pools
    27x27 -> 13x13 = 169 tokens before this function is called. So the
    hard-coded `height = width = 27` reshape blows up. We just derive H, W
    from the actual token count instead and skip the (now redundant) 2D pool
    when stride == config.mm_spatial_pool_stride (resampler already did it).
    """
    num_frames, num_tokens, num_dim = image_feature.shape
    height = width = int(math.sqrt(num_tokens))
    assert height * width == num_tokens, (
        f"Non-square token grid: {num_tokens} tokens"
    )
    # The SpatialPool resampler in this checkpoint has already applied the
    # configured stride, so a second pool here would over-shrink the feature
    # map. Pass it through; add_token_per_grid handles the rest.
    image_feature = image_feature.view(num_frames, height, width, -1)
    image_feature = image_feature.permute(0, 3, 1, 2).contiguous()
    mode = getattr(self.config, "mm_spatial_pool_mode", "bilinear")
    if mode == "average":
        image_feature = nn.functional.avg_pool2d(image_feature, stride) if stride > 1 else image_feature
    elif mode == "max":
        image_feature = nn.functional.max_pool2d(image_feature, stride) if stride > 1 else image_feature
    elif mode == "bilinear":
        if stride > 1:
            h, w = image_feature.shape[2:]
            scaled = [math.ceil(h / stride), math.ceil(w / stride)]
            image_feature = nn.functional.interpolate(image_feature, size=scaled, mode="bilinear")
    elif mode == "conv":
        # Resampler already did a conv-pool; treat this call as identity.
        pass
    else:
        raise ValueError(f"Unexpected mm_spatial_pool_mode: {mode}")
    image_feature = image_feature.permute(0, 2, 3, 1)
    image_feature = image_feature.view(num_frames, -1, num_dim)
    return image_feature


class Adapter:
    name = "LaViDa-O"
    model_id = "jacklishufan/LaViDa-O-v1.0"

    # Generation knobs for short MCQ answers. The LLaDa diffusion sampler
    # decodes max_new_tokens in blocks of `block_length`; step_ratio=1 means
    # one denoising step per token (cheapest setting). prefix_lm=True is
    # what the official VQA recipe uses.
    MAX_NEW_TOKENS = 16
    BLOCK_LENGTH = 16
    STEP_RATIO = 1
    IMAGE_RESOLUTION = 1024  # the Demo notebook uses 1024-px crops for VQA

    # Video knobs. eval_vsi.py overrides `num_frames` from CLI; default 32
    # matches LLaVA-NeXT-Video convention. This is the 11B model and has the
    # heaviest VRAM footprint of any adapter, so 16 / 8 are sensible fallbacks
    # if 32 OOMs.
    num_frames = 32

    def setup(self):
        # build_model handles vision-tower setup and casts to bf16.
        self.tokenizer, self.model, self.image_processor = build_model(
            pretrained=self.model_id,
            model_name="llava_llada",
            device="cuda",
        )
        self.model.eval()
        self.model.requires_grad_(False)
        self.device = self.model.device

        # Patch the (broken-on-video) get_2dPool: the SpatialPool vision
        # resampler in this checkpoint already pools 27x27 -> 13x13, but the
        # default get_2dPool implementation tries to reshape features as
        # 27x27, which crashes on the video path. Our patch derives the grid
        # size from the actual token count and treats the redundant pool as a
        # no-op for "conv" mode (the configured resampler mode).
        self.model.get_2dPool = types.MethodType(_patched_get_2dPool, self.model)

    @torch.no_grad()
    def infer(self, image_paths: List[str], question: str, choices: str) -> str:
        # Match the Demo.ipynb "Image Understanding" recipe.
        image = Image.open(image_paths[0]).convert("RGB")
        image = resize_and_center_crop(image, self.IMAGE_RESOLUTION)

        image_tensor = process_images([image], self.image_processor, self.model.config)
        image_tensor = [t.to(dtype=torch.bfloat16, device=self.device) for t in image_tensor]
        image_sizes = [image.size]

        text_blob = f"{question}\n{choices}\nAnswer with the letter only."
        # The "<image>" placeholder gets split out and replaced with
        # IMAGE_TOKEN_INDEX (-200) by tokenizer_image_token below.
        question_text = f"<image>\n {text_blob}"
        conv = copy.deepcopy(conv_templates["llada"])
        conv.append_message(conv.roles[0], question_text)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()

        input_ids = tokenizer_image_token(
            prompt_question,
            self.tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors="pt",
        ).unsqueeze(0).to(self.device)

        res = self.model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=image_sizes,
            do_sample=False,
            temperature=0,
            max_new_tokens=self.MAX_NEW_TOKENS,
            block_length=self.BLOCK_LENGTH,
            step_ratio=self.STEP_RATIO,
            tokenizer=self.tokenizer,
            prefix_lm=True,
            verbose=False,
        )
        # llada_generate returns (out_ids, history) – we want out_ids only.
        out_ids = res[0] if isinstance(res, (tuple, list)) else res
        text = self.tokenizer.batch_decode(out_ids, skip_special_tokens=True)[0]
        return text.strip()

    # --------------------------------------------------------------- video
    def _build_video_prompt(self, question: str, options: Optional[List[str]]) -> str:
        if options:
            opts_blob = "\n".join(options)
            text_blob = f"{question}\n{opts_blob}\nAnswer with the letter only."
        else:
            text_blob = f"{question}\nAnswer with a single number."
        question_text = f"<image>\n {text_blob}"
        conv = copy.deepcopy(conv_templates["llada"])
        conv.append_message(conv.roles[0], question_text)
        conv.append_message(conv.roles[1], None)
        return conv.get_prompt()

    def _frames_to_video_tensor(self, frames: List[Image.Image]) -> torch.Tensor:
        # The LaViDa-O training pipeline (llava/train/data/datasets.py) feeds
        # video frames through the image_processor as one batched preprocess
        # call rather than per-frame anyres tiling: it produces a
        # (T, 3, H, W) tensor that the model then 2D-pools via the
        # video_idx_in_batch path. We mirror that here.
        pixel_values = self.image_processor.preprocess(
            frames, return_tensors="pt"
        )["pixel_values"]   # (T, 3, H, W)
        return pixel_values.to(dtype=torch.bfloat16, device=self.device)

    @torch.no_grad()
    def infer_video(self, video_path: str, question: str, options) -> str:
        frames = sample_frames(video_path, num_frames=self.num_frames)
        if not frames:
            return ""
        video_tensor = self._frames_to_video_tensor(frames)   # (T, 3, H, W)
        # image_sizes uses the (W, H) of the first frame; on the video path
        # the model branches on modalities=="video" and does not need anyres
        # patch geometry, but llava_arch indexes image_sizes by batch element.
        image_sizes = [frames[0].size]

        prompt = self._build_video_prompt(question, options)
        input_ids = tokenizer_image_token(
            prompt,
            self.tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors="pt",
        ).unsqueeze(0).to(self.device)

        res = self.model.generate(
            input_ids,
            images=[video_tensor],          # one (T, 3, H, W) tensor
            modalities=["video"],           # critical: triggers 2D pooling
            image_sizes=image_sizes,
            do_sample=False,
            temperature=0,
            max_new_tokens=self.MAX_NEW_TOKENS,
            block_length=self.BLOCK_LENGTH,
            step_ratio=self.STEP_RATIO,
            tokenizer=self.tokenizer,
            prefix_lm=True,
            verbose=False,
        )
        out_ids = res[0] if isinstance(res, (tuple, list)) else res
        text = self.tokenizer.batch_decode(out_ids, skip_special_tokens=True)[0]
        return text.strip()
