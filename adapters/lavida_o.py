"""Adapter for LaViDa-O (jacklishufan/LaViDa-O-v1.0).

LaViDa-O is an 11B unified masked-diffusion VLM (Adobe Research) that supports
both multi-modal understanding and image generation/editing. For
ViewSpatial-Bench we only need the understanding path (text answer to an
image + question).

This adapter follows the inference recipe shown in the official Demo.ipynb
"Image Understanding" cell: build_model -> process_images -> conversation
template ("llada") -> tokenizer_image_token -> model.generate(...) with the
LLaDa-style masked-diffusion sampler (step_ratio=1, prefix_lm=True).

Requires the local editable install of the llava package under
repos/LaVida-O/. We add that path to sys.path so the adapter works regardless
of the CWD that eval_vsb.py is launched from.
"""
import copy
import os
import sys
from pathlib import Path
from typing import List

# These env flags match the official Demo.ipynb. They control attention
# padding behavior inside the model code; without them generate() may
# choke on prefix-LM inputs.
os.environ.setdefault("DEBUG_FIX_PADDING", "1")
os.environ.setdefault("NOT_ALWASY_DO_2DPOOL", "1")

# Editable install lives here; make sure llava can be imported from anywhere.
_LAVIDA_O_REPO = Path("/home/haoming/dllm/repos/LaVida-O")
if str(_LAVIDA_O_REPO) not in sys.path:
    sys.path.insert(0, str(_LAVIDA_O_REPO))

import torch
from PIL import Image

from llava.constants import IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.eval.predict_t2i_edit import build_model
from llava.mm_utils import process_images, resize_and_center_crop, tokenizer_image_token


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
