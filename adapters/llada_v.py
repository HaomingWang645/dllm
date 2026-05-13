"""Adapter for GSAI-ML/LLaDA-V (LLaDA-V multimodal diffusion VLM).

The LLaDA-V code lives in a vendored llava-NeXT-style repo at
/home/haoming/dllm/repos/LLaDA-V/train. We add that path to sys.path so the
`llava` package resolves to the LLaDA-V fork (not any other llava install).
"""
import contextlib
import copy
import io
import os
import sys

import torch
from PIL import Image

_LLADA_V_TRAIN_DIR = "/home/haoming/dllm/repos/LLaDA-V/train"
if _LLADA_V_TRAIN_DIR not in sys.path:
    sys.path.insert(0, _LLADA_V_TRAIN_DIR)


class Adapter:
    name = "LLaDA-V"
    model_id = "GSAI-ML/LLaDA-V"
    # Fast inference knobs. gen_length must be divisible by block_length, and
    # (steps // num_blocks) must be an int. We use a single block of 8 tokens
    # with 4 diffusion steps to get a 1-2 token letter answer quickly.
    gen_length = 8
    steps = 4
    block_length = 8

    def setup(self):
        # Silence the chatty repo (`print("Testing ...")` in builder, etc.)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            from llava.model.builder import load_pretrained_model
            from llava.cache import dLLMCache, dLLMCacheConfig
            from dataclasses import asdict

            # Initialize the cache singleton with no-op defaults so the
            # `feature_cache = dLLMCache()` call inside generate has a valid
            # configured instance. Without any cache hooks registered this
            # remains a pure no-op for correctness.
            dLLMCache.new_instance(
                **asdict(dLLMCacheConfig(
                    prompt_interval_steps=1,
                    gen_interval_steps=1,
                    transfer_ratio=0.0,
                ))
            )

            device_map = "cuda:0"
            tokenizer, model, image_processor, max_length = load_pretrained_model(
                self.model_id, None, "llava_llada",
                attn_implementation="sdpa", device_map=device_map,
            )
            model.eval()

        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.device = "cuda:0"

    @torch.no_grad()
    def infer(self, image_paths, question, choices):
        from llava.mm_utils import process_images, tokenizer_image_token
        from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
        from llava.conversation import conv_templates

        image = Image.open(image_paths[0]).convert("RGB")
        image_tensor = process_images([image], self.image_processor, self.model.config)
        image_tensor = [t.to(dtype=torch.float16, device=self.device) for t in image_tensor]

        text_blob = f"{question}\n{choices}\nAnswer with the letter only."
        prompt_with_image = DEFAULT_IMAGE_TOKEN + "\n" + text_blob

        conv = copy.deepcopy(conv_templates["llava_llada"])
        conv.append_message(conv.roles[0], prompt_with_image)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()

        input_ids = tokenizer_image_token(
            prompt_question, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(self.device)
        image_sizes = [image.size]

        # Suppress any prints emitted from inside generate (e.g. cache hooks).
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            cont = self.model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=image_sizes,
                steps=self.steps,
                gen_length=self.gen_length,
                block_length=self.block_length,
                tokenizer=self.tokenizer,
                stopping_criteria=["<|eot_id|>"],
                temperature=0.0,
            )
        text = self.tokenizer.batch_decode(cont, skip_special_tokens=True)[0]
        return text
