"""Adapter for hustvl/DiffusionVL-Qwen2.5VL-7B."""
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor


class Adapter:
    name = "DiffusionVL-Qwen2.5VL-7B"
    model_id = "hustvl/DiffusionVL-Qwen2.5VL-7B"

    def setup(self):
        self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, torch_dtype=torch.bfloat16,
            device_map="auto", trust_remote_code=True,
        ).eval()

    @torch.no_grad()
    def infer(self, image_paths, question, choices):
        image = Image.open(image_paths[0]).convert("RGB")
        text_blob = f"{question}\n{choices}\nAnswer with the letter only."
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": text_blob},
        ]}]
        text = self.processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self.processor(text=[text], images=[image],
                                return_tensors="pt", padding=True)
        inputs = {k: (v.to(self.model.device) if hasattr(v, "to") else v)
                  for k, v in inputs.items()}
        out_ids = self.model.generate(
            inputs=inputs["input_ids"],
            images=inputs.get("pixel_values"),
            image_grid_thws=inputs.get("image_grid_thw"),
            gen_length=8, steps=4, temperature=0.0,
            remasking_strategy="low_confidence_static",
        )
        return self.processor.decode(out_ids[0], skip_special_tokens=True)
