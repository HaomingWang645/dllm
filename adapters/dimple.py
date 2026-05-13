"""Adapter for rp-yu/Dimple-7B (discrete diffusion MLLM with parallel decoding).

Based on the official inference snippet from the model card:
https://huggingface.co/rp-yu/Dimple-7B
"""
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor


class Adapter:
    name = "Dimple-7B"
    model_id = "rp-yu/Dimple-7B"

    def setup(self):
        self.processor = AutoProcessor.from_pretrained(
            self.model_id, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to("cuda:0").eval()
        # Short generation settings for letter-only answers.
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
