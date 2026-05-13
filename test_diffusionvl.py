"""Minimal smoke test for hustvl/DiffusionVL-Qwen2.5VL-7B."""
import time
import requests
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

MODEL_ID = "hustvl/DiffusionVL-Qwen2.5VL-7B"
IMG_URL = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
PROMPT = "Describe this image."

print(f"[1/4] torch={torch.__version__} cuda={torch.cuda.is_available()} "
      f"device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")

t0 = time.time()
print(f"[2/4] Loading processor & model: {MODEL_ID}")
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
model.eval()
print(f"      load time: {time.time()-t0:.1f}s  "
      f"vram alloc: {torch.cuda.memory_allocated()/1e9:.2f} GB")

print(f"[3/4] Fetching image: {IMG_URL}")
image = Image.open(requests.get(IMG_URL, stream=True, timeout=30).raw).convert("RGB")
print(f"      image size: {image.size}")

messages = [
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": PROMPT},
    ]}
]
text = processor.tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)
inputs = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}

print(f"[4/4] Generating (gen_length=128, steps=8) ...")
t0 = time.time()
with torch.no_grad():
    output_ids = model.generate(
        inputs=inputs["input_ids"],
        images=inputs.get("pixel_values"),
        image_grid_thws=inputs.get("image_grid_thw"),
        gen_length=128,
        steps=8,
        temperature=0.0,
        remasking_strategy="low_confidence_static",
    )
gen_t = time.time() - t0

output_text = processor.decode(output_ids[0], skip_special_tokens=True)
print(f"\n=== OUTPUT (gen took {gen_t:.2f}s) ===\n{output_text}\n=== END ===")
