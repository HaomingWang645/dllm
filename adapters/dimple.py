"""Adapter for rp-yu/Dimple-7B (discrete diffusion MLLM with parallel decoding).

Based on the official inference snippet from the model card:
https://huggingface.co/rp-yu/Dimple-7B

Dimple is Qwen2-VL-based. Its tokenizer / chat template reserves a
`video_token_id=151656` and `<|video_pad|>` slot, but the model's forward pass
explicitly raises NotImplementedError on the `pixel_values_videos` branch
("Video feature projector not implemented"). So for video input we use the
multi-image path: each sampled frame is sent as an `<|image_pad|>` block, and
the standard `img_projector` handles them. This matches how Dimple was actually
trained on visual inputs.
"""
from typing import List, Optional

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

from adapters.video_utils import sample_frames


class Adapter:
    name = "Dimple-7B"
    model_id = "rp-yu/Dimple-7B"

    # CLI override: `eval_vsi.py` rewrites this from --num-frames.
    num_frames = 32

    def setup(self):
        self.processor = AutoProcessor.from_pretrained(
            self.model_id, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to("cuda:0").eval()
        # Short generation settings for letter-only / single-number answers.
        self.gen_kwargs = dict(
            max_new_tokens=8,
            output_history=False,
            return_dict_in_generate=True,
            steps=8,
            temperature=0.0,
            top_p=0.95,
            alg="origin",
            use_cache=True,
            alg_p_threshold=0.95,
            use_original_confidence=True,
            decoding_pipeline="dim",
        )

    # ------------------------------------------------------------- image API
    @torch.no_grad()
    def infer(self, image_paths, question, choices):
        image = Image.open(image_paths[0]).convert("RGB")
        text_blob = f"{question}\n{choices}\nAnswer with the letter only."
        # Dimple expects a batched (list-of-conversations) messages structure
        # matching the Qwen2.5-VL chat template.
        messages = [[
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_paths[0]},
                    {"type": "text", "text": text_blob},
                ],
            }
        ]]
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            add_vision_id=False,
        )
        inputs = self.processor(
            text=text,
            images=[image],
            videos=None,
            padding="longest",
            return_tensors="pt",
        )
        inputs = {
            k: (v.to(self.model.device) if hasattr(v, "to") else v)
            for k, v in inputs.items()
        }
        input_ids = inputs.pop("input_ids")
        output = self.model.diffusion_generate(
            input_ids,
            **self.gen_kwargs,
            **inputs,
        )
        # Strip the prompt prefix from the generated sequence(s).
        seq = output.sequences[0]
        gen_tokens = seq[input_ids.shape[1]:].cpu().tolist()
        text_out = self.processor.tokenizer.decode(gen_tokens)
        eos = self.processor.tokenizer.eos_token
        if eos and eos in text_out:
            text_out = text_out.split(eos)[0]
        return text_out

    # ------------------------------------------------------------- video API
    def _build_text_blob(self, question: str, options: Optional[List[str]]) -> str:
        if options:
            opts_blob = "\n".join(options)
            return f"{question}\n{opts_blob}\nAnswer with the letter only."
        return (
            f"{question}\n"
            "Answer with a single number — no units, no explanation."
        )

    @torch.no_grad()
    def infer_video(self, video_path: str, question: str, options) -> str:
        frames: List[Image.Image] = sample_frames(
            video_path, num_frames=self.num_frames
        )
        if not frames:
            return ""

        text_blob = self._build_text_blob(question, options)
        # Multi-image path: one `{"type": "image"}` entry per frame. The chat
        # template expands each into `<|vision_start|><|image_pad|><|vision_end|>`,
        # and the processor then rewrites every `<|image_pad|>` based on each
        # frame's `image_grid_thw`. Dimple's video path is intentionally not
        # implemented in modeling_dimple.py (raises NotImplementedError on
        # `pixel_values_videos`), so we route video as a frame stack instead.
        content = [{"type": "image", "image": video_path} for _ in frames]
        content.append({"type": "text", "text": text_blob})
        messages = [[{"role": "user", "content": content}]]
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            add_vision_id=False,
        )
        inputs = self.processor(
            text=text,
            images=frames,
            videos=None,
            padding="longest",
            return_tensors="pt",
        )

        inputs = {
            k: (v.to(self.model.device) if hasattr(v, "to") else v)
            for k, v in inputs.items()
        }
        input_ids = inputs.pop("input_ids")
        output = self.model.diffusion_generate(
            input_ids,
            **self.gen_kwargs,
            **inputs,
        )
        seq = output.sequences[0]
        gen_tokens = seq[input_ids.shape[1]:].cpu().tolist()
        text_out = self.processor.tokenizer.decode(gen_tokens)
        eos = self.processor.tokenizer.eos_token
        if eos and eos in text_out:
            text_out = text_out.split(eos)[0]
        return text_out
