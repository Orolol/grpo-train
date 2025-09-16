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

# IMPORTANT: Configure CUDA allocator before importing torch/unsloth/vLLM.
# vLLM + WSL can fail with "torch.cuda.MemPool doesn't currently support
# expandable_segments" if PYTORCH_CUDA_ALLOC_CONF enables expandable_segments.
import os as _os
_alloc_conf = _os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
if "expandable_segments" in _alloc_conf.lower():
    # Strip any user-provided expandable_segments setting to avoid MemPool error.
    _alloc_conf = ";".join(
        p for p in _alloc_conf.split(";")
        if not p.strip().lower().startswith("expandable_segments")
    )
    if _alloc_conf:
        _os.environ["PYTORCH_CUDA_ALLOC_CONF"] = _alloc_conf
    else:
        _os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
else:
    # Explicitly disable to avoid external defaults toggling it on.
    _os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
        (_alloc_conf + ";") if _alloc_conf else ""
    ) + "expandable_segments:False"

# Keep Unsloth in standby mode for vLLM when requested.
_os.environ.setdefault("UNSLOTH_VLLM_STANDBY", "1")

from unsloth import FastLanguageModel, PatchFastRL

import argparse
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch
from datasets import Dataset

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
    print(f"Base prompt length (rules + template, no text) = {len(base_tokens)} tokens")
    if len(base_tokens) > token_limit:
        return None

    # No trimming anymore: we either accept or skip based on tokenized length.
    prompt = learner_template.format(rules=rules_text, text=text)
    n_tok = len(tokenizer(prompt, add_special_tokens=False).input_ids)
    print(f"Full text length = {n_tok} tokens")
    if n_tok <= token_limit:
        return prompt, text
    return None


def build_dataset(
    xml_dir: str,
    learner_template: str,
    rules_text: str,
    tokenizer,
    prompt_token_limit: int,
    max_samples: Optional[int] = None,
) -> Tuple[Dataset, Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    prompt_to_text: Dict[str, str] = {}
    skipped_long = 0
    for fp in list_xml_files(xml_dir):
        try:
            xml = read_file(fp)
        except Exception:
            continue
        section = extract_relevant_section(xml)
        fitted = _fit_prompt_to_token_limit(learner_template, rules_text, section, tokenizer, prompt_token_limit)
        if not fitted:
            skipped_long += 1
            continue
        prompt, kept_text = fitted
        rows.append({"prompt": prompt})
        prompt_to_text[prompt] = kept_text
        if max_samples is not None and len(rows) >= max_samples:
            ds = Dataset.from_list(rows)
            print(f"Built dataset with {len(ds)} samples (skipped {skipped_long} too-long documents)")
            return ds, prompt_to_text
    ds = Dataset.from_list(rows)
    print(f"Built dataset with {len(ds)} samples (skipped {skipped_long} too-long documents)")
    return ds, prompt_to_text


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
            raise ImportError("The openai package is required for JudgeClient. Please install/openai v1+")

    # Removed requests-based fallbacks; always use OpenAI SDK below.

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
            # Heuristic: use Responses API for GPT-5 family, else chat.completions
            if self.model.startswith("gpt-5"):
                content = self._post_openai_responses(p)
                if not content:
                    content = self._post_openai_chat(p)
            else:
                content = self._post_openai_chat(p)
            scores.append(self.parse_score(content))
        return scores

    def _score_one_openai(self, prompt: str, max_tokens: int = 64) -> float:
        """Thread-worker: call OpenAI Responses/Chat with retries and backoff."""
        prefer_responses = self.model.startswith("gpt-5")
        delay = 0.5
        for attempt in range(3):
            try:
                text = ""
                if prefer_responses:
                    text = self._post_openai_responses(prompt)
                    if not text:
                        text = self._post_openai_chat(prompt, max_tokens=max_tokens)
                else:
                    text = self._post_openai_chat(prompt, max_tokens=max_tokens)
                if text:
                    return self.parse_score(text)
            except Exception as e:
                print(f"[judge] _score_one_openai attempt {attempt+1} error: {e}")
            time.sleep(delay)
            delay *= 2
        return 0.0

    def score_batch_concurrent(self, prompts: List[str], concurrency: int = 8, max_tokens: int = 64) -> List[float]:
        if not prompts:
            return []
        # Use threads to parallelize OpenAI SDK calls.
        results: List[float] = [0.0] * len(prompts)
        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as ex:
            future_to_idx = {
                ex.submit(self._score_one_openai, prompt, max_tokens): i
                for i, prompt in enumerate(prompts)
            }
            for fut in as_completed(future_to_idx):
                idx = future_to_idx[fut]
                try:
                    results[idx] = float(np.clip(fut.result(), 0.0, 1.0))
                except Exception as e:
                    print(f"[judge] worker error at idx {idx}: {e}")
                    results[idx] = 0.0
        return results


def _strip_qwen_thinking(text: str) -> str:
    """Return the substring after the first </think> tag if present.
    If not found, return the original text.
    """
    marker = "</think>"
    idx = text.find(marker)
    if idx == -1:
        return text
    return text[idx + len(marker):].lstrip()


def make_reward_fn(
    judge: JudgeClient,
    judge_template: str,
    prompt_to_text: Dict[str, str],
    rules_text: str,
    concurrency: int,
):
    def reward_fn(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
        # Pre-evaluate completions: empty or invalid JSON -> -1 (penalty)
        print(completions[0])
        final_scores: List[float] = [-1.0] * len(completions)
        judge_batch: List[str] = []
        judge_map: List[int] = []  # indices in original batch that will receive judge scores

        for i, (p, c_raw) in enumerate(zip(prompts, completions)):
            c = _strip_qwen_thinking(c_raw or "")
            if not c.strip():
                # Keep -1.0 penalty
                continue
            try:
                json.loads(c)
            except Exception:
                # Not valid JSON -> keep -1.0
                continue
            # Valid JSON: prepare judge prompt
            txt = prompt_to_text.get(p, "")
            jp = judge_template.format(rules=rules_text, text=txt, candidate=c)
            judge_map.append(i)
            judge_batch.append(jp)

        # If nothing to judge, return penalties directly
        if not judge_batch:
            return final_scores

        backoff = 1.0
        for _ in range(3):
            scores = judge.score_batch_concurrent(judge_batch, concurrency=concurrency)
            if len(scores) == len(judge_batch):
                # Clamp judge-provided scores to [0,1] and write into final list
                for idx, s in zip(judge_map, scores):
                    final_scores[idx] = float(np.clip(s, 0.0, 1.0))
                return final_scores
            time.sleep(backoff)
            backoff *= 2

        # On repeated failure: set 0.0 for judged items (keep -1 for invalid ones)
        for idx in judge_map:
            final_scores[idx] = 0.0
        return final_scores

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
    enable_vllm: bool,
    gpu_memory_utilization: float,
):

    PatchFastRL("GRPO", FastLanguageModel)
    model, tokenizer = FastLanguageModel.from_pretrained(
        dtype=None,
        model_name=base_model,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        fast_inference=bool(enable_vllm),
        gpu_memory_utilization=gpu_memory_utilization,
        # unsloth_vllm_standby=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        # Native checkpointing avoids flex-attention backward failures on GPT-OSS 20B
        use_gradient_checkpointing=True,
        lora_dropout=0,
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
    use_vllm: bool,
    
):
    # Ensure completion length fits in model max when combined with prompt limit
    max_model_len = int(getattr(getattr(model, "config", object()), "max_position_embeddings", getattr(tokenizer, "model_max_length", 2048)))
    if max_completion_len and max_prompt_len and max_model_len:
        headroom = max(1, max_model_len - max_prompt_len - 8)
        effective_completion_len = max(1, min(max_completion_len, headroom))
    else:
        effective_completion_len = max_completion_len or 128

    args = GRPOConfig(
        learning_rate=learning_rate,
        adam_beta1 = 0.9,
        adam_beta2 = 0.99,
        weight_decay = 0.1,
        warmup_ratio = 0.1,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        logging_steps=1,
        auto_find_batch_size=True,
        gradient_accumulation_steps=4,
        num_generations=num_generations,
        max_prompt_length=max_prompt_len,
        max_completion_length=effective_completion_len,
        num_train_epochs=epochs,
        report_to="none",
        output_dir=output_dir,
        use_vllm=bool(use_vllm),
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
    max_completion_length: int,
    prompt_token_limit: int,
    model_max_len: int,
) -> float:
    model.eval()
    scores: List[float] = []
    for p in prompts:
        inputs = tokenizer(p, return_tensors="pt", truncation=True, max_length=prompt_token_limit)
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        # Ensure we do not exceed model context
        prompt_len = int(inputs["input_ids"].shape[-1])
        allow_new = max(1, min(max_completion_length, model_max_len - prompt_len - 8))
        outputs = model.generate(
            **inputs,
            max_new_tokens=allow_new,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        gen_ids = outputs[0][len(inputs["input_ids"][0]):]
        # Prefer token-level parsing of </think> if available
        try:
            end_think_id = tokenizer.encode("</think>", add_special_tokens=False)
            if len(end_think_id) == 1:
                eid = end_think_id[0]
                ids_list = gen_ids.tolist()
                # index of last occurrence of </think>
                idx = len(ids_list) - ids_list[::-1].index(eid)
                thinking_ids = ids_list[:idx]
                content_ids = ids_list[idx:]
                completion = tokenizer.decode(content_ids, skip_special_tokens=True)
            else:
                completion = tokenizer.decode(gen_ids, skip_special_tokens=True)
                completion = _strip_qwen_thinking(completion)
        except Exception:
            completion = tokenizer.decode(gen_ids, skip_special_tokens=True)
            completion = _strip_qwen_thinking(completion)
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
    parser.add_argument("--max_samples", type=int, default=None)

    # Model
    parser.add_argument("--base_model", type=str, default="unsloth/Qwen3-4B-Thinking-2507-unsloth-bnb-4bit")
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--max_seq_length", type=int, default=6000)
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--max_completion_length", type=int, default=2048)

    # Training
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--num_generations", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--enable_vllm", action="store_true", help="Enable vLLM fast inference (higher VRAM usage)")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.80, help="GPU memory fraction for vLLM KV cache")

    # Judge
    parser.add_argument("--judge_model", type=str, default="gpt-5-nano")
    parser.add_argument("--judge_base_url", type=str, default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--judge_api_key", type=str, default=os.environ.get("OPENAI_API_KEY", ""))
    parser.add_argument("--judge_concurrency", type=int, default=8, help="Parallel requests for judge scoring")

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
        enable_vllm=bool(args.enable_vllm),
        gpu_memory_utilization=float(args.gpu_memory_utilization),
    )
    print(f"Parameters: {model.num_parameters()}")

    # Align generation defaults with CLI args to avoid transformers warnings
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.max_length = args.max_seq_length
        model.generation_config.max_new_tokens = args.max_completion_length

    # Prompt token limit with safety margin and reserved completion budget
    safety_margin = 64
    model_ctx = int(min(args.max_seq_length, getattr(tokenizer, "model_max_length", args.max_seq_length)))
    reserved_completion = int(max(1, args.max_completion_length))
    # prompt_token_limit = max(
    #     16,
    #     model_ctx - reserved_completion - safety_margin
    # )

    prompt_token_limit = model_ctx
    print(
        f"Using prompt token limit: {prompt_token_limit} (ctx={model_ctx}, reserve={reserved_completion}, margin={safety_margin})"
    )

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
    reward_fn = make_reward_fn(judge, judge_template, train_prompt_to_text, rules_text, args.judge_concurrency)

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
        use_vllm=bool(args.enable_vllm),
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
        max_completion_length=args.max_completion_length,
        prompt_token_limit=prompt_token_limit,
        model_max_len=args.max_seq_length,
    )
    print(f"Test average score: {score:.4f}")


if __name__ == "__main__":
    main()
