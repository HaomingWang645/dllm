"""Adapter for LucasJinWang/FUDOKI (discrete-flow-matching multimodal model).

Lifts the multimodal-understanding pipeline from FUDOKI's
`inference_i2t_local.py` and `VLMEvalKit/run.py`:
  - Loads the FUDOKI MultiModalityCausalLM (a Janus-style backbone retrained
    with discrete flow matching).
  - Builds the canonical FUDOKI prompt:
        <image_placeholder>{question}\n{choices}\nAnswer with the letter only.
    wrapped by FUDOKI's chat template (User: ... Assistant: ...).
  - Performs discrete-flow-matching sampling via
    `MixtureDiscreteSoftmaxEulerSolver` to fill the masked text region after
    "Assistant:".

The text region after "Assistant:" is sampled in one shot by the diffusion
solver -- there is no autoregressive generation. We keep `txt_max_length`
(total text capacity including prompt + answer) and `answer_token_num`
(tokens we actually decode) small so that we stay near ~1s/Q.
"""
import os
import sys

import torch
from PIL import Image
from torchvision import transforms

# Make FUDOKI's `fudoki` / `flow_matching` packages importable.
_FUDOKI_REPO = "/home/haoming/dllm/repos/FUDOKI"
if _FUDOKI_REPO not in sys.path:
    sys.path.insert(0, _FUDOKI_REPO)

from fudoki.eval_loop import CFGScaledModel  # noqa: E402
from fudoki.janus.models import VLChatProcessor  # noqa: E402
from fudoki.model import instantiate_model  # noqa: E402
from flow_matching.path import MixtureDiscreteSoftmaxProbPath  # noqa: E402
from flow_matching.solver import MixtureDiscreteSoftmaxEulerSolver  # noqa: E402


VOCABULARY_SIZE_TXT = 102400
VOCABULARY_SIZE_IMG = 16384
IMG_LEN = 576


def _resize_pad(image, image_size=384):
    """Resize to fit image_size while preserving aspect ratio; pad with gray.

    Lifted verbatim (sans Chinese comments) from FUDOKI inference scripts.
    """
    w, h = image.size
    if w <= 0 or h <= 0:
        return image.resize((image_size, image_size), Image.Resampling.BILINEAR)

    resize_scale = image_size / max(w, h)
    new_w = max(1, int(w * resize_scale))
    new_h = max(1, int(h * resize_scale))

    padding_color = (127, 127, 127)
    new_image = Image.new("RGB", (image_size, image_size), padding_color)

    if new_w <= 0 or new_h <= 0:
        return image.resize((image_size, image_size), Image.Resampling.BILINEAR)

    image = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
    paste_x = (image_size - new_w) // 2
    paste_y = (image_size - new_h) // 2
    new_image.paste(image, (paste_x, paste_y))
    return new_image


class Adapter:
    name = "FUDOKI"
    model_path = (
        "/home/haoming/.cache/huggingface/hub/"
        "models--LucasJinWang--FUDOKI/snapshots/"
        "6827760f631d253b9219158a6a36d3fda2b38821"
    )

    # Short generation for multiple-choice letter answers.
    # txt_max_length is the total text-slot capacity (prompt + answer region).
    # The solver fills the masked region (after "Assistant:") in one shot.
    txt_max_length = 128       # plenty for "User: <img>...question...\nA. ...\nAssistant:" + small answer
    answer_token_num = 8       # only decode the first few tokens after "Assistant:"
    discrete_fm_steps = 20     # matches GQA recipe (short answers)
    image_size = 384

    def setup(self):
        self.device = torch.device("cuda:0")

        # Some torch backports rely on get_default_device; provide a shim if
        # this older torch (2.0.1) is used.
        if not hasattr(torch, "get_default_device"):
            torch.get_default_device = lambda: torch.device("cpu")

        # Load the FUDOKI multimodal causal LM.
        self.model = (
            instantiate_model(self.model_path)
            .to(self.device)
            .to(torch.float32)
        )
        self.model.train(False)

        self.vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(
            self.model_path
        )

        # CFG wrapper (we use cfg_scale=0 for understanding).
        self.cfg_model = CFGScaledModel(model=self.model, g_or_u="understanding")

        # Discrete-flow-matching paths (text + image embeddings live in checkpoint dir).
        text_embedding_path = os.path.join(self.model_path, "text_embedding.pt")
        image_embedding_path = os.path.join(self.model_path, "image_embedding.pt")
        self.path_txt = MixtureDiscreteSoftmaxProbPath(
            mode="text", embedding_path=text_embedding_path
        )
        self.path_img = MixtureDiscreteSoftmaxProbPath(
            mode="image", embedding_path=image_embedding_path
        )
        self.solver = MixtureDiscreteSoftmaxEulerSolver(
            model=self.cfg_model,
            path_txt=self.path_txt,
            path_img=None,
            vocabulary_size_txt=VOCABULARY_SIZE_TXT,
            vocabulary_size_img=VOCABULARY_SIZE_IMG,
        )

        self.image_transform = transforms.Compose([
            transforms.Lambda(lambda im: _resize_pad(im, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])

    @torch.no_grad()
    def infer(self, image_paths, question, choices):
        device = self.device
        proc = self.vl_chat_processor

        text_blob = f"{question}\n{choices}\nAnswer with the letter only."
        conversation = [
            {"role": "User", "content": f"<image_placeholder>{text_blob}"},
            {"role": "Assistant", "content": ""},
        ]
        sft_format = proc.apply_sft_template_for_multi_turn_prompts(
            conversations=conversation,
            sft_format=proc.sft_format,
            system_prompt=proc.system_prompt,
        )

        # Tokenize prompt and expand <image_placeholder> tokens.
        input_ids = torch.LongTensor(proc.tokenizer.encode(sft_format))
        image_token_mask = (input_ids == proc.image_id)
        image_indices = image_token_mask.nonzero()
        input_ids, _ = proc.add_image_token(
            image_indices=image_indices, input_ids=input_ids
        )

        original_input_id_len = input_ids.shape[0]
        txt_max_length = self.txt_max_length
        img_len = IMG_LEN

        if original_input_id_len >= txt_max_length + img_len:
            # Prompt is too long for our small budget; fall back gracefully.
            return ""

        # Pad to (txt_max_length + img_len).
        pad_count = txt_max_length + img_len - input_ids.shape[0]
        input_ids = torch.cat(
            [input_ids, torch.LongTensor([proc.pad_id]).repeat(pad_count)], dim=0
        )

        # attention_mask: True everywhere up to (prompt + answer_token_num), False after.
        attention_mask = torch.zeros((input_ids.shape[0],), dtype=torch.bool)
        attention_mask[: original_input_id_len + self.answer_token_num] = True

        # Image token mask: locations of image placeholder tokens (zero them out
        # since image features come via the vision encoder, not these ids).
        image_expanded_token_mask = (input_ids == proc.image_id).to(dtype=torch.int64)
        image_idx = torch.where(image_expanded_token_mask == 1)[0]
        input_ids[image_idx] = 0

        # Text token mask: positions to be sampled by diffusion (after "Assistant:").
        text_expanded_token_mask = torch.zeros_like(image_expanded_token_mask)
        split_token = proc.tokenizer.encode("Assistant:", add_special_tokens=False)
        split_len = len(split_token)
        start_index = -1
        for j in range(len(input_ids) - split_len + 1):
            if input_ids[j : j + split_len].tolist() == split_token:
                start_index = j
                break
        if start_index == -1:
            raise RuntimeError("FUDOKI adapter: 'Assistant:' split token not found in prompt")
        # Only mask the answer region (next `answer_token_num` positions).
        ans_begin = start_index + split_len
        ans_end = min(ans_begin + self.answer_token_num, text_expanded_token_mask.shape[0])
        text_expanded_token_mask[ans_begin:ans_end] = 1

        # Prepare image tensor.
        img = Image.open(image_paths[0]).convert("RGB")
        img = self.image_transform(img).to(device)

        batch_size = 1
        data_info = {
            "text_token_mask": text_expanded_token_mask.unsqueeze(0).to(device),
            "image_token_mask": image_expanded_token_mask.unsqueeze(0).to(device),
            "generation_or_understanding_mask": torch.zeros(
                (1, 1), dtype=torch.long, device=device
            ),
            "attention_mask": attention_mask.unsqueeze(0).to(device),
            "understanding_img": img.unsqueeze(0).to(device),
            "has_understanding_img": torch.ones((1, 1), dtype=torch.long, device=device),
        }

        input_ids = input_ids.unsqueeze(0).to(device)

        # Initial state: random tokens at masked text positions, prompt tokens elsewhere.
        x_0_txt = torch.randint(
            VOCABULARY_SIZE_TXT, input_ids.shape, dtype=torch.long, device=device
        )
        x_init = (
            x_0_txt * data_info["text_token_mask"]
            + input_ids * (1 - data_info["text_token_mask"])
        )

        synthetic_samples = self.solver.sample(
            x_init=x_init,
            step_size=1.0 / self.discrete_fm_steps,
            verbose=False,
            div_free=0.0,
            dtype_categorical=torch.float32,
            datainfo=data_info,
            cfg_scale=0,
        )

        # Decode only the answer region (post-"Assistant:").
        ans_ids = synthetic_samples[0, ans_begin:ans_end]
        text = proc.tokenizer.decode(ans_ids, skip_special_tokens=True)

        # Strip FUDOKI's EOS sentinel if it slipped through.
        for eos in ("<｜end▁of▁sentence｜>", "<|end_of_sentence|>"):
            idx = text.find(eos)
            if idx != -1:
                text = text[:idx]
        return text.strip()
