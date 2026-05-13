"""Adapter for MeissonFlow/Muddit (1B unified discrete-diffusion model).

The Muddit pipeline is custom (not a `transformers` model); we lift the calls from
`repos/Muddit/inference_i2t.py` directly. Transformer weights come from
`MeissonFlow/Muddit` (subfolder `1024/transformer`), while tokenizer / text encoder /
VQ-VAE / scheduler come from `MeissonFlow/Meissonic`. The mask-token embedding for VQA
is loaded from `MeissonFlow/Muddit/1024/mask_token_embedding.pth`.
"""
import os
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

# Make the Muddit src/ importable
MUDDIT_REPO = Path("/home/haoming/dllm/repos/Muddit")
if str(MUDDIT_REPO) not in sys.path:
    sys.path.insert(0, str(MUDDIT_REPO))

from src.transformer import SymmetricTransformer2DModel  # noqa: E402
from src.pipeline import UnifiedPipeline  # noqa: E402
from src.scheduler import Scheduler  # noqa: E402

from transformers import CLIPTextModelWithProjection, CLIPTokenizer  # noqa: E402
from diffusers import VQModel  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402


class Adapter:
    name = "Muddit-1B"
    base_repo = "MeissonFlow/Meissonic"  # tokenizer / text_encoder / vqvae / scheduler
    muddit_repo = "MeissonFlow/Muddit"  # transformer + mask_token_embedding
    # Use the 1024 variant — it ships `mask_token_embedding.pth` (the 512 variant does not).
    variant = "1024"
    resolution = 1024
    steps = 32
    cfg = 1.0  # i2t uses cfg=1.0 per inference_i2t.sh

    def setup(self):
        # Materialize the Muddit weights locally so we can point at a real directory.
        muddit_root = snapshot_download(
            repo_id=self.muddit_repo,
            allow_patterns=[f"{self.variant}/*"],
        )
        self.transformer_dir = os.path.join(muddit_root, self.variant)

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = device

        transformer = SymmetricTransformer2DModel.from_pretrained(
            self.transformer_dir, subfolder="transformer",
        )
        vq_model = VQModel.from_pretrained(self.base_repo, subfolder="vqvae")
        text_encoder = CLIPTextModelWithProjection.from_pretrained(
            self.base_repo, subfolder="text_encoder",
        )
        tokenizer = CLIPTokenizer.from_pretrained(self.base_repo, subfolder="tokenizer")
        scheduler = Scheduler.from_pretrained(self.base_repo, subfolder="scheduler")

        self.pipe = UnifiedPipeline(
            vqvae=vq_model,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            transformer=transformer,
            scheduler=scheduler,
        )
        self.pipe.to(device)

        # Disable the inner per-step tqdm so the eval log stays clean.
        try:
            self.pipe.set_progress_bar_config(disable=True)
        except Exception:
            pass

        self._image_tf = transforms.Compose([
            transforms.Resize((self.resolution, self.resolution)),
            transforms.ToTensor(),
        ])

    def _load_image(self, image_path):
        img = Image.open(image_path).convert("RGB")
        t = self._image_tf(img).unsqueeze(0)  # [1, 3, H, W] in [0, 1]
        return t

    @torch.no_grad()
    def infer(self, image_paths, question, choices):
        image = self._load_image(image_paths[0])
        prompt = f"{question}\n{choices}\nAnswer with the letter only."

        output = self.pipe(
            prompt=prompt,
            image=image,
            height=self.resolution,
            width=self.resolution,
            guidance_scale=self.cfg,
            num_inference_steps=self.steps,
            mask_token_embedding=self.transformer_dir,
        )
        # `prompts` is a list of decoded strings (one per image).
        return output.prompts[0] if output.prompts else ""
