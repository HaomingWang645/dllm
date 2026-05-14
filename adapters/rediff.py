"""Adapter for jiyatai/ReDiff (refining-enhanced vision-language diffusion model built on LLaDA-V).

ReDiff is trained for detailed image captioning. For MCQ tasks (ViewSpatial-Bench)
we ask it to produce a short answer and parse the letter out via eval_vsb.extract_letter.
"""
import copy
import os
import sys
import warnings
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoTokenizer

# Make the ReDiff `train/` directory importable so we get the local `llava` package.
_REDIFF_TRAIN = "/home/haoming/dllm/repos/ReDiff/train"
if _REDIFF_TRAIN not in sys.path:
    sys.path.insert(0, _REDIFF_TRAIN)

warnings.filterwarnings("ignore")

from llava.model.builder import load_pretrained_model  # noqa: E402
from llava.mm_utils import process_images, tokenizer_image_token  # noqa: E402
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN  # noqa: E402
from llava.conversation import conv_templates  # noqa: E402

# Local repo path (sibling of adapters/) — add so `adapters.video_utils` resolves
# whether or not the caller already configured sys.path.
_DLLM_ROOT = "/home/haoming/dllm"
if _DLLM_ROOT not in sys.path:
    sys.path.insert(0, _DLLM_ROOT)

from adapters.video_utils import sample_frames  # noqa: E402


class Adapter:
    name = "ReDiff"
    model_id = "jiyatai/ReDiff"
    # ReDiff config points mm_vision_tower at a non-existent local path; substitute the
    # public siglip2 model it was trained against.
    vision_tower_id = "google/siglip2-so400m-patch14-384"
    conv_template = "llava_llada"

    # Short generation tuned for MCQ (gen_length <= 16, steps <= 8, both divisible).
    gen_length = 8
    steps = 4
    block_length = 8

    # Number of frames to feed into the model for video tasks (VSI-Bench).
    # Fall back to 16 or 8 if 32 is too slow or OOMs on a given GPU.
    num_frames = 32

    def setup(self):
        device = "cuda:0"
        self.device = device

        # Patch the conv template's tokenizer up front so the hardcoded local path in
        # llava/conversation.py (.../LLaDA-V) is never touched.
        tok = AutoTokenizer.from_pretrained(self.model_id, use_fast=False, trust_remote_code=True)
        conv_templates[self.conv_template].tokenizer = tok

        # Force a usable vision_tower path so build_vision_tower doesn't try to read the
        # bundled placeholder path. Loader passes this through `overwrite_config`.
        overwrite_config = {"mm_vision_tower": self.vision_tower_id}

        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
            self.model_id,
            None,
            "llava_llada",  # selects the LLaDA branch in builder.py
            attn_implementation="sdpa",
            device_map=device,
            overwrite_config=overwrite_config,
        )
        self.model.eval()

    @torch.no_grad()
    def infer(self, image_paths, question, choices):
        image = Image.open(image_paths[0]).convert("RGB")
        image_tensor = process_images([image], self.image_processor, self.model.config)
        image_tensor = [t.to(dtype=torch.float16, device=self.device) for t in image_tensor]

        prompt_text = (
            f"{DEFAULT_IMAGE_TOKEN}\n{question}\n{choices}\n"
            "Answer with the letter only."
        )
        conv = copy.deepcopy(conv_templates[self.conv_template])
        conv.append_message(conv.roles[0], prompt_text)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(
            prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(self.device)

        out = self.model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=[image.size],
            revise=True,
            fake_ans=None,
            steps=self.steps,
            gen_length=self.gen_length,
            block_length=self.block_length,
            tokenizer=self.tokenizer,
            stopping_criteria=["<|eot_id|>"],
        )
        # `generate_with_embeds` returns ONLY the generated portion (already sliced
        # to gen_length, optionally truncated at the stop token), so decode directly.
        text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        return text

    @torch.no_grad()
    def infer_video(self, video_path, question, options):
        """Video MCQ / numeric inference for VSI-Bench.

        - Uniformly samples `self.num_frames` frames via shared `sample_frames`.
        - Feeds them through `process_images` as a batch and prefixes the prompt
          with N copies of DEFAULT_IMAGE_TOKEN (the LLaDA-V / LLaVA-NeXT pattern).
        - `options` is None for free-form numeric tasks, or a list of "A. ..."
          strings for MCQ.
        """
        frames = sample_frames(video_path, num_frames=self.num_frames)
        n = len(frames)

        image_tensor = process_images(frames, self.image_processor, self.model.config)
        if isinstance(image_tensor, torch.Tensor):
            image_tensor = [t.to(dtype=torch.float16, device=self.device) for t in image_tensor]
        else:
            image_tensor = [t.to(dtype=torch.float16, device=self.device) for t in image_tensor]

        image_tokens = (DEFAULT_IMAGE_TOKEN + "\n") * n

        if options:
            options_block = "\n".join(options)
            prompt_text = (
                f"{image_tokens}{question}\n{options_block}\n"
                "Answer with the letter only."
            )
        else:
            prompt_text = (
                f"{image_tokens}{question}\n"
                "Answer with a single number."
            )

        conv = copy.deepcopy(conv_templates[self.conv_template])
        conv.append_message(conv.roles[0], prompt_text)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(
            prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(self.device)

        image_sizes = [f.size for f in frames]

        out = self.model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=image_sizes,
            revise=True,
            fake_ans=None,
            steps=self.steps,
            gen_length=self.gen_length,
            block_length=self.block_length,
            tokenizer=self.tokenizer,
            stopping_criteria=["<|eot_id|>"],
        )
        text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        return text
