"""Qwen zero-shot genre explanation."""
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import config

SYSTEM_PROMPT = (
    "You are an expert in Arabic literature, media, and genre classification. "
    "You are skilled at identifying genres across Modern Standard Arabic, "
    "Classical Arabic, and Arabic dialects."
)


def load_qwen():
    print("\nLoading Qwen tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.QWEN_MODEL)
    print(f"Loading Qwen model ({config.QWEN_DEVICE})...")
    model = AutoModelForCausalLM.from_pretrained(config.QWEN_MODEL, torch_dtype=torch.float16)
    model = model.to(config.QWEN_DEVICE)
    model.eval()
    return tokenizer, model


def build_prediction_prompt(text_en, candidates):
    """Candidates are shown as hints only; the embedding snap enforces validity."""
    genre_hints = "\n".join(
        f"- {c['specific_genre']} ({c['broad_genre']}): {c['specific_genre_definition']}"
        for c in candidates
    )
    return f"""You are a genre classification expert specialising in Arabic literature and media.

Known genre categories (use these as reference):
{genre_hints}

Read the text below carefully and describe its genre in detail.
Focus on: writing style, formality level, topic, register, and structural features.
The text may be Modern Standard Arabic, Classical Arabic, or a dialect — factor this into your reasoning.

Text:
{text_en}

Respond in this exact format:
Broad genre: <your answer>
Specific genre: <your answer>
Explanation: <detailed reasoning about style, register, and content>
"""


def _generate(messages, tokenizer, model):
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True)
    inputs = {k: v.to(config.QWEN_DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=config.MAX_NEW_TOKENS,
            do_sample=False,
            temperature=config.TEMPERATURE,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True,
    )
    return generated.strip()


def call_qwen(prompt, tokenizer, model):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    return _generate(messages, tokenizer, model)


def summarize_text(text_en, tokenizer, model):
    """Optional helper (disabled in the pipeline): summarise a text with Qwen."""
    messages = [{
        "role": "user",
        "content": f"Summarize the text delimited by triple backticks and only give the final answer. ```{text_en}```",
    }]
    return _generate(messages, tokenizer, model)


def extract_explanation(qwen_output):
    """Return only the Explanation section (richer embedding signal)."""
    match = re.search(r"Explanation:(.*)", qwen_output, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else qwen_output
