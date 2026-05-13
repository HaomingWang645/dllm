"""Adapter for Alpha-VLLM/Lumina-DiMOO (omni discrete-diffusion MLLM).

Mirrors the calls in repos/Lumina-DiMOO/inference/inference_mmu.py:
  - LLaDAForMultiModalGeneration as the LM
  - VQModel (vqvae/) as the image tokenizer
  - generate_text_understanding() as the discrete-diffusion sampler

Special tokens / offsets are pulled from repos/Lumina-DiMOO/config.py.
"""
import os
import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoTokenizer

# Make the cloned repo importable
_REPO = Path("/home/haoming/dllm/repos/Lumina-DiMOO")
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from config import SPECIAL_TOKENS  # noqa: E402
from model import LLaDAForMultiModalGeneration  # noqa: E402
from generators.text_understanding_generator import generate_text_understanding  # noqa: E402
from utils.image_utils import (  # noqa: E402
    add_break_line,
    calculate_vq_params,
    encode_img_with_breaks,
    generate_crop_size_list,
    var_center_crop,
)
from utils.prompt_utils import generate_multimodal_understanding_prompt  # noqa: E402


class Adapter:
    name = "Lumina-DiMOO"
    model_id = "Alpha-VLLM/Lumina-DiMOO"
    vae_id = "Alpha-VLLM/Lumina-DiMOO"  # vqvae lives in the same repo under subfolder "vqvae"

    # Short generation for MCQ ("A"/"B"/"C"/"D"). Must satisfy:
    #   gen_length % block_length == 0  and  steps % (gen_length/block_length) == 0
    gen_length = 8
    block_length = 8
    steps = 4
    temperature = 0.0
    cfg_scale = 0.0
    remasking = "low_confidence"

    def setup(self):
        # Use cuda:0 (caller pins GPU 7 via CUDA_VISIBLE_DEVICES)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        self.model = LLaDAForMultiModalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map={"": self.device},
        ).eval()

        # VQ-VAE for image tokenization (subfolder "vqvae" in same HF repo)
        from diffusers import VQModel
        self.vqvae = VQModel.from_pretrained(self.vae_id, subfolder="vqvae").to(self.device)
        self.vae_scale = 2 ** (len(self.vqvae.config.block_out_channels) - 1)

        # Special tokens
        self.MASK = SPECIAL_TOKENS["mask_token"]
        self.NEWLINE = SPECIAL_TOKENS["newline_token"]
        self.BOA = SPECIAL_TOKENS["answer_start"]
        self.EOA = SPECIAL_TOKENS["answer_end"]

        # Pre-compute crop sizes the same way inference_mmu.py does (1024px ref).
        self._crop_size_list = generate_crop_size_list((1024 // 32) ** 2, 32)

    @torch.no_grad()
    def infer(self, image_paths, question, choices):
        text_blob = f"{question}\n{choices}\nAnswer with the letter only."
        input_prompt = generate_multimodal_understanding_prompt(text_blob)
        input_ids = self.tokenizer(input_prompt)["input_ids"]

        # Image -> VQ tokens (with row newline markers + BOI/EOI), per inference_mmu.py
        img = Image.open(image_paths[0]).convert("RGB")
        image = var_center_crop(img, crop_size_list=self._crop_size_list)
        image_width, image_height = image.size
        _seq_len, _newline_every, gh, gw = calculate_vq_params(
            image_height, image_width, self.vae_scale
        )
        input_img_token = encode_img_with_breaks(image, vqvae=self.vqvae)
        input_img_token = add_break_line(input_img_token, gh, gw, new_number=self.NEWLINE)

        # Splice image tokens into the prompt, then append masked answer slot.
        input_token = input_ids[:-1] + input_img_token + input_ids[-1:]
        code_start = len(input_token) + 1  # text starts after BOA
        input_token = input_token + [self.BOA] + self.gen_length * [self.MASK] + [self.EOA]

        x = torch.tensor(input_token, device=self.device).unsqueeze(0)

        out = generate_text_understanding(
            self.model, x,
            steps=self.steps,
            gen_length=self.gen_length,
            block_length=self.block_length,
            temperature=self.temperature,
            cfg_scale=self.cfg_scale,
            remasking=self.remasking,
            mask_id=self.MASK,
            code_start=code_start,
        )

        # The model writes its answer tokens between BOA and EOA.
        text = self.tokenizer.batch_decode(
            out[:, code_start:-1], skip_special_tokens=True
        )[0]
        return text
