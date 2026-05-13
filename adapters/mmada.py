"""Adapter for Gen-Verse/MMaDA-8B-MixCoT (multimodal understanding).

Lifts the multimodal-understanding pipeline from MMaDA's `inference_mmu.py`:
  - Loads MAGVITv2 VQ tokenizer (image -> discrete tokens).
  - Loads MMadaModelLM (a diffusion LM that consumes interleaved text+image tokens).
  - Builds the canonical MMaDA prompt:
        [<|mmu|>] [<|soi|>] <image_tokens> [<|eoi|>] <chat-templated text tokens>
  - Runs `model.mmu_generate(...)` which performs masked-diffusion denoising
    in blocks.
"""
import os
import sys

import torch
from PIL import Image

# Make MMaDA's `models` / `training` packages importable.
_MMADA_REPO = "/home/haoming/dllm/repos/MMaDA"
if _MMADA_REPO not in sys.path:
    sys.path.insert(0, _MMADA_REPO)

from models import MAGVITv2, MMadaModelLM  # noqa: E402
from training.prompting_utils import UniversalPrompting  # noqa: E402
from training.utils import image_transform  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402


class Adapter:
    name = "MMaDA-8B-MixCoT"
    model_id = "Gen-Verse/MMaDA-8B-MixCoT"
    vq_model_id = "showlab/magvitv2"
    resolution = 512  # matches configs/mmada_demo.yaml

    # Short generation (per task spec). 8 must be divisible by block_length;
    # steps must be divisible by num_blocks = gen_length / block_length.
    gen_length = 8
    block_length = 2          # -> num_blocks = 4
    steps = 4                 # -> steps_per_block = 1 (4 // 4)
    max_text_len = 512        # matches mmada_demo.yaml dataset.preprocessing.max_seq_length

    def setup(self):
        self.device = torch.device("cuda:0")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, padding_side="left", trust_remote_code=True
        )
        # use_reserved_token=True replicates inference_mmu.py
        self.uni_prompting = UniversalPrompting(
            self.tokenizer,
            max_text_len=self.max_text_len,
            special_tokens=(
                "<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>",
                "<|t2i|>", "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>",
            ),
            ignore_id=-100,
            cond_dropout_prob=0.1,
            use_reserved_token=True,
        )

        self.vq_model = MAGVITv2.from_pretrained(self.vq_model_id).to(self.device)
        self.vq_model.eval()
        self.vq_model.requires_grad_(False)

        self.model = MMadaModelLM.from_pretrained(
            self.model_id, trust_remote_code=True, torch_dtype=torch.bfloat16,
        ).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def infer(self, image_paths, question, choices):
        image_ori = Image.open(image_paths[0]).convert("RGB")
        image = image_transform(image_ori, resolution=self.resolution).to(self.device)
        image = image.unsqueeze(0)
        image_tokens = self.vq_model.get_code(image) + len(self.uni_prompting.text_tokenizer)

        text_blob = f"{question}\n{choices}\nAnswer with the letter only."
        messages = [{"role": "user", "content": text_blob}]
        text_token_ids = self.uni_prompting.text_tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt",
        ).to(self.device)

        batch_size = image_tokens.shape[0]
        input_ids = torch.cat([
            (torch.ones(batch_size, 1) * self.uni_prompting.sptids_dict['<|mmu|>']).to(self.device),
            (torch.ones(batch_size, 1) * self.uni_prompting.sptids_dict['<|soi|>']).to(self.device),
            image_tokens,
            (torch.ones(batch_size, 1) * self.uni_prompting.sptids_dict['<|eoi|>']).to(self.device),
            text_token_ids,
        ], dim=1).long()

        with torch.autocast("cuda", dtype=torch.bfloat16):
            output_ids = self.model.mmu_generate(
                input_ids,
                max_new_tokens=self.gen_length,
                steps=self.steps,
                block_length=self.block_length,
            )

        generated_ids = output_ids[:, input_ids.shape[1]:]
        response_text = self.uni_prompting.text_tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )[0]
        return response_text
