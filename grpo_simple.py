#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Minimal GRPO training script with Unsloth and an external judge API.

Principles
- No prompt text in Python: load learner and judge templates from files.
- Train on all XML files in data/train, evaluate on data/test.
- Reward uses an OpenAI-compatible API (e.g., GPT) as judge.

Templates
- Learner template: must contain {rules} and {text} placeholders.
- Judge template: must contain {rules}, {text}, and {candidate}.

Example
  OPENAI_API_KEY=... \
  python grpo_simple.py \
    --base_model mistralai/Mistral-7B-Instruct-v0.2 \
    --learner_prompt prompts/learner.txt \
    --judge_prompt prompts/judge.txt \
    --rules_path data/rules_2.md \
    --output_dir out_grpo \
    --judge_model gpt-4o-mini
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import Dataset

from unsloth import FastLanguageModel, PatchFastRL
from trl import GRPOConfig, GRPOTrainer


# -----------------------------
# Utilities
# -----------------------------

def read_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def list_xml_files(root: str) -> List[str]:
    files: List[str] = []
    for base, _, names in os.walk(root):
        for n in names:
            if n.lower().endswith(".xml"):
                files.append(os.path.join(base, n))
    return sorted(files)


def extract_relevant_section(xml_text: str) -> str:
    # Prefer <NonStructure>, else <TXD>, else full file.
    m = re.search(r"<NonStructure>[\s\S]*?</NonStructure>", xml_text)
    if m:
        return m.group(0)
    m = re.search(r"<TXD>[\s\S]*?</TXD>", xml_text)
    if m:
        return m.group(0)
    return xml_text


def chunk_text(text: str, max_chars: int) -> List[str]:
    if max_chars <= 0 or len(text) <= max_chars:
        return [text]
    # Try to respect paragraph-like tags if present
    parts = re.findall(r"<(?:AL|P)>[\s\S]*?</(?:AL|P)>", text)
    if not parts:
        return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]
    chunks: List[str] = []
    buf = ""
    for p in parts:
        if len(buf) + len(p) > max_chars and buf:
            chunks.append(buf)
            buf = p
        else:
            buf += p
    if buf:
        chunks.append(buf)
    return chunks


def _fit_prompt_to_token_limit(
    learner_template: str,
    rules_text: str,
    text: str,
    tokenizer,
    token_limit: int,
) -> Optional[Tuple[str, str]]:
    """Return (prompt, trimmed_text) so that tokenized prompt <= token_limit.
    Uses binary search on the text length.
    """
    # Quick check: can empty text fit? If not, skip.
    base_prompt = learner_template.format(rules=rules_text, text="")
    base_tokens = tokenizer(base_prompt, add_special_tokens=False).input_ids
    if len(base_tokens) > token_limit:
        return None

    lo, hi = 0, len(text)
    best_len = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        cand_text = text[:mid]
        cand_prompt = learner_template.format(rules=rules_text, text=cand_text)
        n_tok = len(tokenizer(cand_prompt, add_special_tokens=False).input_ids)
        if n_tok <= token_limit:
            best_len = mid
            lo = mid + 1
        else:
            hi = mid - 1
    trimmed = text[:best_len]
    final_prompt = learner_template.format(rules=rules_text, text=trimmed)
    return final_prompt, trimmed


def build_dataset(
    xml_dir: str,
    learner_template: str,
    rules_text: str,
    max_chars_per_sample: int,
    tokenizer,
    prompt_token_limit: int,
    max_samples: Optional[int] = None,
) -> Tuple[Dataset, Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    prompt_to_text: Dict[str, str] = {}
    for fp in list_xml_files(xml_dir):
        try:
            xml = read_file(fp)
        except Exception:
            continue
        section = extract_relevant_section(xml)
        for chunk in chunk_text(section, max_chars_per_sample):
            fitted = _fit_prompt_to_token_limit(learner_template, rules_text, chunk, tokenizer, prompt_token_limit)
            if not fitted:
                continue
            prompt, trimmed = fitted
            rows.append({"prompt": prompt})
            prompt_to_text[prompt] = trimmed
            if max_samples is not None and len(rows) >= max_samples:
                return Dataset.from_list(rows), prompt_to_text
    return Dataset.from_list(rows), prompt_to_text


# -----------------------------
# Judge
# -----------------------------

class JudgeClient:
    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 60.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self._client = None
        try:
            from openai import OpenAI  # type: ignore
            self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        except Exception:
            self._client = None

    def _post_chat_requests(self, content: str, max_tokens: int = 64) -> str:
        import requests
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0.0,
            "max_tokens": max_tokens,
        }
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
            r.raise_for_status()
            data = r.json()
            return data.get("choices", [{}])[0].get("message", {}).get("content", "")
        except Exception as e:
            print(f"[judge] error(requests): {e}")
            return ""

    def _post_openai_responses(self, content: str) -> str:
        try:
            # Use Responses API for GPT-5-like models
            resp = self._client.responses.create(
                model=self.model,
                input=content,
                reasoning={"effort": "medium"},
                text={"verbosity": "low"},
            )
            return getattr(resp, "output_text", None) or ""
        except Exception as e:
            print(f"[judge] error(responses): {e}")
            return ""

    def _post_openai_chat(self, content: str, max_tokens: int = 64) -> str:
        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": content}],
                temperature=0.0,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            print(f"[judge] error(chat): {e}")
            return ""

    @staticmethod
    def parse_score(text: str) -> float:
        t = text.strip()
        # Prefer JSON object with a 'score' field
        try:
            obj = json.loads(t)
            if isinstance(obj, dict) and "score" in obj:
                return float(obj["score"])
        except Exception:
            pass
        # Fallback: first number found in the string
        m = re.search(r"[-+]?\d*\.?\d+", t)
        if not m:
            return 0.0
        val = float(m.group(0))
        # normalize common ranges
        if val > 10:
            return val / 100.0
        if val > 1:
            return val / 10.0
        return val

    def score_batch(self, prompts: List[str]) -> List[float]:
        scores: List[float] = []
        for p in prompts:
            content = ""
            if self._client is not None:
                # Heuristic: use Responses API for GPT-5 family, else chat.completions
                if self.model.startswith("gpt-5"):
                    content = self._post_openai_responses(p)
                    if not content:
                        content = self._post_openai_chat(p)
                else:
                    content = self._post_openai_chat(p)
            else:
                content = self._post_chat_requests(p)
            scores.append(self.parse_score(content))
        return scores


def make_reward_fn(
    judge: JudgeClient,
    judge_template: str,
    prompt_to_text: Dict[str, str],
    rules_text: str,
):
    def reward_fn(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
        batch: List[str] = []
        for p, c in zip(prompts, completions):
            txt = prompt_to_text.get(p, "")
            jp = judge_template.format(rules=rules_text, text=txt, candidate=c)
            batch.append(jp)
        backoff = 1.0
        for _ in range(3):
            scores = judge.score_batch(batch)
            if len(scores) == len(batch):
                return [float(np.clip(s, 0.0, 1.0)) for s in scores]
            time.sleep(backoff)
            backoff *= 2
        return [0.0] * len(batch)

    return reward_fn


# -----------------------------
# Model + Trainer
# -----------------------------

def set_seed(seed: int = 3407) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def build_model_tokenizer(
    base_model: str,
    max_seq_length: int,
    lora_rank: int,
    load_in_4bit: bool,
):
    PatchFastRL("GRPO", FastLanguageModel)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        fast_inference=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=lora_rank,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    return model, tokenizer


def build_trainer(
    model,
    tokenizer,
    dataset: Dataset,
    reward_fn,
    output_dir: str,
    epochs: int,
    num_generations: int,
    max_prompt_len: int,
    max_completion_len: int,
    learning_rate: float,
    
):
    args = GRPOConfig(
        learning_rate=learning_rate,
        adam_beta1 = 0.9,
        adam_beta2 = 0.99,
        weight_decay = 0.1,
        warmup_ratio = 0.1,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        logging_steps=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        num_generations=num_generations,
        max_prompt_length=max_prompt_len,
        max_completion_length=max_completion_len,
        num_train_epochs=epochs,
        report_to="none",
        output_dir=output_dir,
        use_vllm=True,
        loss_type="grpo",
    )
    return GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_fn],
        args=args,
        train_dataset=dataset,
    )


# -----------------------------
# Evaluation
# -----------------------------

@torch.no_grad()
def evaluate(
    model,
    tokenizer,
    prompts: List[str],
    judge: JudgeClient,
    judge_template: str,
    prompt_to_text: Dict[str, str],
    rules_text: str,
    max_new_tokens: int,
    prompt_token_limit: int,
) -> float:
    model.eval()
    scores: List[float] = []
    for p in prompts:
        inputs = tokenizer(p, return_tensors="pt", truncation=True, max_length=prompt_token_limit)
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        completion = tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
        txt = prompt_to_text.get(p, "")
        jp = judge_template.format(rules=rules_text, text=txt, candidate=completion)
        s = judge.score_batch([jp])[0]
        print(f"Completion: {completion}, Score: {s:.4f}")
        scores.append(float(np.clip(s, 0.0, 1.0)))
    return float(np.mean(scores)) if scores else 0.0


# -----------------------------
# CLI
# -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal GRPO with Unsloth + external judge")
    # Data
    parser.add_argument("--train_dir", type=str, default=str(Path("data") / "train"))
    parser.add_argument("--test_dir", type=str, default=str(Path("data") / "test"))
    parser.add_argument("--rules_path", type=str, default=str(Path("data") / "rules.md"))
    parser.add_argument("--learner_prompt", type=str, default=str(Path("data") / "learner_prompt.md"))
    parser.add_argument("--judge_prompt", type=str, default=str(Path("data") / "judge_prompt.md"))
    parser.add_argument("--max_chars_per_sample", type=int, default=20000)
    parser.add_argument("--max_samples", type=int, default=None)

    # Model
    parser.add_argument("--base_model", type=str, default="unsloth/mistral-7b-instruct-v0.3-bnb-4bit")
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--max_prompt_length", type=int, default=4096)
    parser.add_argument("--max_completion_length", type=int, default=4096)
    parser.add_argument("--max_new_tokens", type=int, default=4096)

    # Training
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-6)

    # Judge
    parser.add_argument("--judge_model", type=str, default="gpt-5-nano")
    parser.add_argument("--judge_base_url", type=str, default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--judge_api_key", type=str, default=os.environ.get("OPENAI_API_KEY", ""))

    args = parser.parse_args()

    if not args.judge_api_key:
        raise ValueError("Judge API key is required (set --judge_api_key or OPENAI_API_KEY)")

    set_seed(3407)

    # Templates and rules
    rules_text = read_file(args.rules_path)
    learner_template = read_file(args.learner_prompt)
    judge_template = read_file(args.judge_prompt)

    # Model first to get tokenizer for token limits
    print(f"Loading model: {args.base_model}")
    model, tokenizer = build_model_tokenizer(
        base_model=args.base_model,
        max_seq_length=args.max_seq_length,
        lora_rank=args.lora_rank,
        load_in_4bit=bool(args.load_in_4bit),
    )
    print(f"Parameters: {model.num_parameters()}")

    # Prompt token limit with safety margin
    # Keep some headroom under model/tokenizer max to avoid runtime errors (e.g., bos, system tokens)
    safety_margin = 128
    prompt_token_limit = max(256, min(args.max_prompt_length, getattr(tokenizer, "model_max_length", 2048) - safety_margin))
    print(f"Using prompt token limit: {prompt_token_limit}")

    # Quick diagnostic: does base prompt (template + rules, empty text) already exceed the limit?
    base_prompt = learner_template.format(rules=rules_text, text="")
    base_len = len(tokenizer(base_prompt, add_special_tokens=False).input_ids)
    if base_len > prompt_token_limit:
        print(f"[warn] Base prompt length (rules + template, no text) = {base_len} tokens exceeds the limit {prompt_token_limit}.")
        print("       Consider a shorter rules file, a longer-context base model, or increasing --max_seq_length.")

    # Build train dataset
    train_ds, train_prompt_to_text = build_dataset(
        args.train_dir,
        learner_template,
        rules_text,
        args.max_chars_per_sample,
        tokenizer,
        prompt_token_limit,
        args.max_samples,
    )
    if len(train_ds) == 0:
        raise RuntimeError(f"No training samples found in {args.train_dir}")

    # Judge
    judge = JudgeClient(
        base_url=args.judge_base_url,
        api_key=args.judge_api_key,
        model=args.judge_model,
    )

    # Reward
    reward_fn = make_reward_fn(judge, judge_template, train_prompt_to_text, rules_text)

    # Trainer
    trainer = build_trainer(
        model=model,
        tokenizer=tokenizer,
        dataset=train_ds,
        reward_fn=reward_fn,
        output_dir=args.output_dir,
        epochs=args.epochs,
        num_generations=args.num_generations,
        max_prompt_len=prompt_token_limit,
        max_completion_len=args.max_completion_length,
        learning_rate=args.learning_rate,
    )

    # Train
    print(f"Training: {len(train_ds)} samples")
    trainer.train()

    # Save model
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model saved to {args.output_dir}")

    # Test set
    test_ds, test_prompt_to_text = build_dataset(
        args.test_dir,
        learner_template,
        rules_text,
        args.max_chars_per_sample,
        tokenizer,
        prompt_token_limit,
        None,
    )
    test_prompts = [row["prompt"] for row in test_ds]
    prompt_to_text = {**train_prompt_to_text, **test_prompt_to_text}
    score = evaluate(
        model,
        tokenizer,
        test_prompts,
        judge,
        judge_template,
        prompt_to_text,
        rules_text,
        max_new_tokens=args.max_new_tokens,
        prompt_token_limit=prompt_token_limit,
    )
    print(f"Test average score: {score:.4f}")


if __name__ == "__main__":
    main()
