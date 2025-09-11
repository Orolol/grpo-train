#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Minimal GRPO script using Unsloth and an external API judge.

Design principles:
- No prompt text hardcoded in Python. Both the learner prompt and judge prompt
  are loaded from files and formatted with {rules}, {text}, and {candidate}.
- Train on all XML files in data/train, then evaluate on data/test.
- Judge model is called via an OpenAI-compatible API during reward computation.

Usage (example):
  OPENAI_API_KEY=sk-... \
  python grpo.py \
    --base_model mistralai/Mistral-7B-Instruct-v0.2 \
    --prompt_path prompts/learner_prompt.txt \
    --judge_prompt_path prompts/judge_prompt.txt \
    --rules_path data/rules_2.md \
    --output_dir out_grpo \
    --judge_model gpt-4o-mini

Prompt templates must use Python format placeholders:
- Learner: expects {rules} and {text}
- Judge: expects {rules}, {text}, and {candidate}
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
# Helpers: IO and preprocessing
# -----------------------------

def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def list_xml_files(root: str) -> List[str]:
    files: List[str] = []
    for base, _, names in os.walk(root):
        for n in names:
            if n.lower().endswith(".xml"):
                files.append(os.path.join(base, n))
    return sorted(files)


def extract_relevant_xml_section(xml: str) -> str:
    # Prefer <NonStructure>, else <TXD>, else full file.
    m = re.search(r"<NonStructure>[\s\S]*?</NonStructure>", xml)
    if m:
        return m.group(0)
    m = re.search(r"<TXD>[\s\S]*?</TXD>", xml)
    if m:
        return m.group(0)
    return xml


def chunk_text(text: str, max_chars: int) -> List[str]:
    if max_chars <= 0 or len(text) <= max_chars:
        return [text]
    # Try to respect paragraph tags if present
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


def build_dataset_from_xml_dir(
    xml_dir: str,
    learner_template: str,
    rules_text: str,
    max_chars_per_sample: int,
    max_samples: Optional[int] = None,
) -> Tuple[Dataset, Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    prompt_to_text: Dict[str, str] = {}
    for fp in list_xml_files(xml_dir):
        try:
            xml = read_text(fp)
        except Exception:
            continue
        section = extract_relevant_xml_section(xml)
        for chunk in chunk_text(section, max_chars_per_sample):
            prompt = learner_template.format(rules=rules_text, text=chunk)
            rows.append({"prompt": prompt})
            prompt_to_text[prompt] = chunk
            if max_samples is not None and len(rows) >= max_samples:
                ds = Dataset.from_list(rows)
                return ds, prompt_to_text
    ds = Dataset.from_list(rows)
    return ds, prompt_to_text


# -----------------------------
# Judge client and reward utils
# -----------------------------

class JudgeClient:
    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 60.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def _post_chat(self, messages: List[Dict[str, str]], max_tokens: int = 64) -> str:
        import requests  # local import to keep optional

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": max_tokens,
        }
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
            r.raise_for_status()
            data = r.json()
            return data.get("choices", [{}])[0].get("message", {}).get("content", "")
        except Exception as e:
            print(f"[judge] error: {e}")
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
        m = re.search(r"[-+]?[0-9]*\.?[0-9]+", t)
        if not m:
            return 0.0
        val = float(m.group(0))
        # normalize common ranges
        if val > 10:
            return val / 100.0
        if val > 1:
            return val / 10.0
        return val

    def score_batch(self, judge_prompts: List[str]) -> List[float]:
        scores: List[float] = []
        for p in judge_prompts:
            # Single-message chat to avoid hardcoding a system prompt
            content = self._post_chat(messages=[{"role": "user", "content": p}])
            scores.append(self.parse_score(content))
        return scores


def make_reward_fn(
    judge: JudgeClient,
    judge_template: str,
    prompt_to_text: Dict[str, str],
    rules_text: str,
):
    def reward_fn(prompts: List[str], completions: List[str], **_: dict) -> List[float]:
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
# Model and training utilities
# -----------------------------

def set_seed(seed: int = 3407) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def build_model_and_tokenizer(
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
        fast_inference=False,
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
    num_generations: int,
    num_steps: int,
    max_prompt_len: int,
    max_completion_len: int,
    learning_rate: float,
):
    args = GRPOConfig(
        learning_rate=learning_rate,
        weight_decay=0.0,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        logging_steps=5,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        num_generations=num_generations,
        max_prompt_length=max_prompt_len,
        max_completion_length=max_completion_len,
        max_steps=num_steps,
        report_to="none",
        output_dir=output_dir,
        use_vllm=False,
        loss_type="grpo",
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_fn],
        args=args,
        train_dataset=dataset,
    )
    return trainer


# -----------------------------
# Evaluation
# -----------------------------

@torch.no_grad()
def evaluate_on_test(
    model,
    tokenizer,
    test_prompts: List[str],
    judge: JudgeClient,
    judge_template: str,
    prompt_to_text: Dict[str, str],
    rules_text: str,
    max_new_tokens: int,
) -> float:
    model.eval()
    scores: List[float] = []
    for p in test_prompts:
        inputs = tokenizer(p, return_tensors="pt", truncation=True, max_length=tokenizer.model_max_length)
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
        scores.append(float(np.clip(s, 0.0, 1.0)))
    return float(np.mean(scores)) if scores else 0.0


# -----------------------------
# CLI
# -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal GRPO with Unsloth and external judge")
    # Data
    parser.add_argument("--train_dir", type=str, default=str(Path("data") / "train"))
    parser.add_argument("--test_dir", type=str, default=str(Path("data") / "test"))
    parser.add_argument("--rules_path", type=str, required=True)
    parser.add_argument("--prompt_path", type=str, required=True, help="Learner prompt template file with {rules} and {text}")
    parser.add_argument("--judge_prompt_path", type=str, required=True, help="Judge prompt template with {rules}, {text}, {candidate}")
    parser.add_argument("--max_chars_per_sample", type=int, default=20000)
    parser.add_argument("--max_samples", type=int, default=None)

    # Model/training
    parser.add_argument("--base_model", type=str, default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--max_prompt_length", type=int, default=2048)
    parser.add_argument("--max_completion_length", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--num_steps", type=int, default=None)
    parser.add_argument("--num_epochs", type=int, default=1)

    # Judge API
    parser.add_argument("--judge_model", type=str, required=True)
    parser.add_argument("--judge_base_url", type=str, default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--judge_api_key", type=str, default=os.environ.get("OPENAI_API_KEY", ""))

    args = parser.parse_args()

    if not args.judge_api_key:
        raise ValueError("Judge API key is required (set --judge_api_key or OPENAI_API_KEY)")

    set_seed(3407)

    # Load templates and rules
    rules_text = read_text(args.rules_path)
    learner_template = read_text(args.prompt_path)
    judge_template = read_text(args.judge_prompt_path)

    # Build training set
    train_ds, train_prompt_to_text = build_dataset_from_xml_dir(
        args.train_dir,
        learner_template,
        rules_text,
        args.max_chars_per_sample,
        args.max_samples,
    )
    if len(train_ds) == 0:
        raise RuntimeError(f"No training samples found in {args.train_dir}")

    # Steps = full pass over train set × epochs if not provided
    num_steps = args.num_steps or (len(train_ds) * max(1, args.num_epochs))

    # Model
    print(f"Loading model: {args.base_model}")
    model, tokenizer = build_model_and_tokenizer(
        base_model=args.base_model,
        max_seq_length=args.max_seq_length,
        lora_rank=args.lora_rank,
        load_in_4bit=bool(args.load_in_4bit),
    )
    print(f"Parameters: {model.num_parameters()}")

    # Judge
    judge = JudgeClient(
        base_url=args.judge_base_url,
        api_key=args.judge_api_key,
        model=args.judge_model,
    )

    # Reward fn
    reward_fn = make_reward_fn(judge, judge_template, train_prompt_to_text, rules_text)

    # Trainer
    trainer = build_trainer(
        model=model,
        tokenizer=tokenizer,
        dataset=train_ds,
        reward_fn=reward_fn,
        output_dir=args.output_dir,
        num_generations=args.num_generations,
        num_steps=num_steps,
        max_prompt_len=args.max_prompt_length,
        max_completion_len=args.max_completion_length,
        learning_rate=args.learning_rate,
    )

    # Train full pass on train set
    print(f"Training: {len(train_ds)} samples, steps={num_steps}")
    trainer.train()

    # Save model
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model saved to {args.output_dir}")

    # Build test prompts and evaluate
    test_ds, test_prompt_to_text = build_dataset_from_xml_dir(
        args.test_dir,
        learner_template,
        rules_text,
        args.max_chars_per_sample,
        None,
    )
    test_prompts = [row["prompt"] for row in test_ds]
    # Merge mappings (train + test) so prompt lookup always works
    prompt_to_text = {**train_prompt_to_text, **test_prompt_to_text}
    test_score = evaluate_on_test(
        model,
        tokenizer,
        test_prompts,
        judge,
        judge_template,
        prompt_to_text,
        rules_text,
        max_new_tokens=args.max_new_tokens,
    )
    print(f"Test average score: {test_score:.4f}")


if __name__ == "__main__":
    main()



def build_xml_anonymization_dataset(xml_dir: str, rules_text: str, max_samples: Optional[int], max_chars_per_sample: int,
                                    use_teacher_reference: bool, teacher_call_kwargs: Optional[Dict[str, str]]) -> Dataset:
    rows: List[Dict[str, str]] = []
    xml_files = list_xml_files(xml_dir)

    teacher_enabled = bool(use_teacher_reference and teacher_call_kwargs)
    teacher_model = teacher_call_kwargs.get("model") if teacher_call_kwargs else None

    # Optional teacher via anonymize.call_openai_for_diff
    call_teacher = None
    if teacher_enabled:
        try:
            from anonymize import call_openai_for_diff  # type: ignore
            call_teacher = call_openai_for_diff
        except Exception:
            call_teacher = None

    for fp in xml_files:
        print(f"Processing {fp}")
        try:
            with open(fp, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception:
            continue

        sections, section_tag = ([], None)
        if extract_sections_to_anonymize is not None:
            try:
                sections, section_tag = extract_sections_to_anonymize(content)
            except Exception:
                sections, section_tag = ([], None)
        if not sections:
            # Fallback 1: extract <NonStructure>
            section_tag = "NonStructure"
            m = re.search(r'(<NonStructure>.*?</NonStructure>)', content, re.DOTALL)
            if m:
                sections = [("juri", m.group(1), m.start(), m.end(), "")]
            else:
                # Fallback 2: use entire content to avoid empty dataset
                section_tag = "ROOT"
                sections = [("root", content, 0, len(content), "")]

        for (_sid, section_content, _s, _e, _ctx) in sections:
            inner = extract_section_inner_text(section_tag, section_content)
            chunks = split_by_paragraphs(inner, max_chars=max_chars_per_sample)
            for ch in chunks:
                prompt = PROMPT_TEMPLATE.format(rules=rules_text, text=ch)
                ref = ""
                if teacher_enabled and call_teacher is not None:
                    try:
                        diff_text, _err = call_teacher(rules_text, ch, model=teacher_model)
                        if isinstance(diff_text, str):
                            ref = diff_text
                    except Exception:
                        ref = ""
                rows.append({"prompt": prompt, "reference": ref})
                if max_samples is not None and len(rows) >= max_samples:
                    return Dataset.from_list(rows)
    print(f"Built dataset with {len(rows)} samples")
    return Dataset.from_list(rows)


def try_import_openai():
    try:
        from openai import OpenAI  # type: ignore
        return OpenAI
    except Exception:
        return None


class MetricsLogger:
    def __init__(self, output_dir: str, console: Optional[Console] = None):
        self.output_dir = Path(output_dir)
        self.console = console or Console()
        self.csv_path = self.output_dir / "metrics.csv"
        self.logs_dir = self.output_dir / "logs"
        
        # Create directories
        self.output_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        
        # Initialize CSV
        self.initialize_csv()
        
    def initialize_csv(self):
        if not self.csv_path.exists():
            headers = [
                "timestamp", "epoch", "step", "train_loss", "train_reward", 
                "test_score", "learning_rate", "completion_length", "kl_divergence", 
                "grad_norm", "num_tokens"
            ]
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
    
    def log_metrics(self, metrics: Dict):
        # Add to CSV
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            row = [
                datetime.now().isoformat(),
                metrics.get('epoch', 0),
                metrics.get('step', 0),
                metrics.get('loss', 0),
                metrics.get('reward', 0),
                metrics.get('test_score', 0),
                metrics.get('learning_rate', 0),
                metrics.get('completion_length', 0),
                metrics.get('kl', 0),
                metrics.get('grad_norm', 0),
                metrics.get('num_tokens', 0),
            ]
            writer.writerow(row)
    
    def display_epoch_summary(self, epoch: int, metrics: Dict, test_score: Optional[float] = None):
        if not RICH_AVAILABLE:
            print(f"Epoch {epoch} | Loss: {metrics.get('loss', 0):.4f} | Reward: {metrics.get('reward', 0):.4f}")
            if test_score is not None:
                print(f"Test Score: {test_score:.4f}")
            return
            
        table = Table(title=f"Epoch {epoch} Summary", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan", width=20)
        table.add_column("Value", style="green", width=15)
        
        table.add_row("Train Loss", f"{metrics.get('loss', 0):.6f}")
        table.add_row("Train Reward", f"{metrics.get('reward', 0):.4f}")
        table.add_row("Learning Rate", f"{metrics.get('learning_rate', 0):.2e}")
        table.add_row("Completion Length", f"{metrics.get('completion_length', 0):.1f}")
        table.add_row("KL Divergence", f"{metrics.get('kl', 0):.6f}")
        table.add_row("Grad Norm", f"{metrics.get('grad_norm', 0):.4f}")
        
        if test_score is not None:
            color = "green" if test_score > 0.3 else "yellow" if test_score > 0.15 else "red"
            table.add_row("Test Score", f"{test_score:.4f}", style=color)
        
        self.console.print(table)
        self.console.print()


class TestEvaluator:
    def __init__(self, test_dir: str, rules_text: str, judge: 'JudgeClient', 
                 tokenizer, max_chars_per_sample: int = 12000, console: Optional[Console] = None):
        self.test_dir = Path(test_dir)
        self.rules_text = rules_text
        self.judge = judge
        self.tokenizer = tokenizer
        self.max_chars_per_sample = max_chars_per_sample
        self.console = console or Console()
        
        # Build test dataset once
        self.test_dataset = self._build_test_dataset()
        
    def _build_test_dataset(self) -> List[str]:
        if not self.test_dir.exists():
            print(f"Warning: Test directory {self.test_dir} not found")
            return []
            
        test_files = list(self.test_dir.glob("*.xml"))
        if not test_files:
            print(f"Warning: No XML files found in {self.test_dir}")
            return []
            
        prompts = []
        for xml_file in test_files:
            try:
                with open(xml_file, "r", encoding="utf-8") as f:
                    content = f.read()
                
                # Use same extraction logic as training
                sections, section_tag = ([], None)
                if extract_sections_to_anonymize is not None:
                    try:
                        sections, section_tag = extract_sections_to_anonymize(content)
                    except Exception:
                        sections, section_tag = ([], None)
                        
                if not sections:
                    section_tag = "NonStructure"
                    m = re.search(r'(<NonStructure>.*?</NonStructure>)', content, re.DOTALL)
                    if m:
                        sections = [("juri", m.group(1), m.start(), m.end(), "")]
                    else:
                        section_tag = "ROOT"
                        sections = [("root", content, 0, len(content), "")]
                
                for (_sid, section_content, _s, _e, _ctx) in sections:
                    inner = extract_section_inner_text(section_tag, section_content)
                    chunks = split_by_paragraphs(inner, max_chars=self.max_chars_per_sample)
                    for chunk in chunks[:1]:  # Take only first chunk per file for test
                        prompt = PROMPT_TEMPLATE.format(rules=self.rules_text, text=chunk)
                        prompts.append(prompt)
                        
            except Exception as e:
                print(f"Error processing {xml_file}: {e}")
                
        return prompts
    
    def evaluate(self, model) -> float:
        if not self.test_dataset:
            return 0.0
            
        if RICH_AVAILABLE:
            self.console.print("[yellow]Running test evaluation...[/yellow]")
            
        scores = []
        
        for prompt in self.test_dataset:
            try:
                # Generate single response
                inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=8192)
                
                # Move inputs to same device as model
                device = next(model.parameters()).device
                inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
                
                with torch.no_grad():
                    # Check if model has vLLM engine
                    if hasattr(model, 'vllm_engine'):
                        # Use vLLM for generation
                        prompt_text = self.tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
                        try:
                            from vllm import SamplingParams
                            sampling_params = SamplingParams(
                                temperature=0.7,
                                max_tokens=512,
                                stop_token_ids=[self.tokenizer.eos_token_id]
                            )
                            outputs = model.vllm_engine.generate([prompt_text], sampling_params)
                            completion = outputs[0].outputs[0].text
                        except Exception:
                            # Fallback to standard generation
                            outputs = model.generate(
                                **inputs,
                                max_new_tokens=512,
                                temperature=0.7,
                                do_sample=True,
                                pad_token_id=self.tokenizer.eos_token_id
                            )
                            completion = self.tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
                    else:
                        # Standard generation
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=512,
                            temperature=0.7,
                            do_sample=True,
                            pad_token_id=self.tokenizer.eos_token_id
                        )
                        completion = self.tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
                
                # Score with judge
                text, rules = extract_rules_and_text_from_prompt(prompt)
                judge_prompt = json.dumps({"rules": rules, "text": text}, ensure_ascii=False)
                score = self.judge._score_single(judge_prompt, completion, None)
                scores.append(score)
                
            except Exception as e:
                print(f"Error evaluating test sample: {e}")
                scores.append(0.0)
                
        avg_score = np.mean(scores) if scores else 0.0
        
        if RICH_AVAILABLE:
            color = "green" if avg_score > 0.3 else "yellow" if avg_score > 0.15 else "red"
            self.console.print(f"[{color}]Test Score: {avg_score:.4f}[/{color}] ({len(scores)} samples)")
        else:
            print(f"Test Score: {avg_score:.4f} ({len(scores)} samples)")
            
        return avg_score


class CheckpointManager:
    def __init__(self, output_dir: str, keep_n_checkpoints: int = 3):
        self.output_dir = Path(output_dir)
        self.checkpoints_dir = self.output_dir / "checkpoints"
        self.keep_n_checkpoints = keep_n_checkpoints
        self.best_score = -1.0
        self.best_checkpoint_path = None
        
        self.checkpoints_dir.mkdir(exist_ok=True)
        
    def save_checkpoint(self, model, tokenizer, epoch: int, metrics: Dict, test_score: float):
        checkpoint_dir = self.checkpoints_dir / f"epoch_{epoch}"
        checkpoint_dir.mkdir(exist_ok=True)
        
        # Save model
        model.save_pretrained(str(checkpoint_dir / "model"))
        tokenizer.save_pretrained(str(checkpoint_dir / "tokenizer"))
        
        # Save metrics
        metrics_file = checkpoint_dir / "metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump({**metrics, 'test_score': test_score, 'epoch': epoch}, f, indent=2)
        
        # Update best model
        if test_score > self.best_score:
            self.best_score = test_score
            self.best_checkpoint_path = checkpoint_dir
            
            best_dir = self.output_dir / "best_model"
            if best_dir.exists():
                import shutil
                shutil.rmtree(best_dir)
            
            import shutil
            shutil.copytree(checkpoint_dir, best_dir)
            
            print(f"🏆 New best model saved! Test score: {test_score:.4f}")
        
        # Clean old checkpoints
        self._cleanup_checkpoints()
        
    def _cleanup_checkpoints(self):
        checkpoints = sorted(self.checkpoints_dir.glob("epoch_*"), 
                           key=lambda x: int(x.name.split("_")[1]))
        
        while len(checkpoints) > self.keep_n_checkpoints:
            oldest = checkpoints.pop(0)
            if oldest != self.best_checkpoint_path:
                import shutil
                shutil.rmtree(oldest)


class TrainingCallback:
    def __init__(self, test_evaluator: TestEvaluator, metrics_logger: MetricsLogger, 
                 checkpoint_manager: CheckpointManager, console: Optional[Console] = None):
        self.test_evaluator = test_evaluator
        self.metrics_logger = metrics_logger
        self.checkpoint_manager = checkpoint_manager
        self.console = console or Console()
        self.epoch_count = 0
        self.last_step = -1
        
    def on_epoch_end(self, model, tokenizer, metrics: Dict):
        # Calculate epoch from step (assuming fixed steps per epoch)
        current_step = metrics.get('step', 0)
        if current_step <= self.last_step:
            return  # Same step, skip
            
        self.epoch_count += 1
        self.last_step = current_step
        
        # Run test evaluation
        test_score = self.test_evaluator.evaluate(model)
        
        # Log metrics
        metrics_with_test = {**metrics, 'test_score': test_score, 'epoch': self.epoch_count}
        self.metrics_logger.log_metrics(metrics_with_test)
        
        # Display summary
        self.metrics_logger.display_epoch_summary(self.epoch_count, metrics, test_score)
        
        # Save checkpoint
        self.checkpoint_manager.save_checkpoint(model, tokenizer, self.epoch_count, metrics, test_score)


class JudgeClient:
    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 60.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._timeout = timeout
        OpenAI = try_import_openai()
        self._client = OpenAI(base_url=self._base_url, api_key=self._api_key) if OpenAI else None

    @property
    def model(self) -> str:
        return self._model

    def score(self, items: List[Tuple[str, str, Optional[str]]]) -> List[float]:
        scores: List[float] = []
        for prompt, completion, reference in items:
            scores.append(self._score_single(prompt, completion, reference))
        return scores

    def _score_single(self, prompt: str, completion: str, reference: Optional[str]) -> float:
        system = (
            "Tu es un évaluateur strict d'anonymisation. Retourne UNIQUEMENT un objet JSON avec les clés 'score' (0..1) et 'explanation'. "
            "Attribue une meilleure note si le JSON candidat respecte strictement le schéma du diff, applique correctement les règles et anonymiserait correctement."
        )
        user = {
            "task": "evaluer_diff_anonymisation",
            "context": prompt,
            "candidate_json": completion,
            "teacher_json": reference or "",
            "rubric": (
                "1) La sortie doit être un JSON valide {diff: [{search: [..], replace: str, motif: str}]}. "
                "2) Les chaînes de 'search' doivent exister dans TEXT et viser uniquement le contenu textuel des balises P/AL. "
                "3) 'replace' doit respecter les règles (initiales pour les noms; [Adresse]/[Localité]/[Téléphone]/[Courriel]/etc.). "
                "4) Aucune sur-anonymisation des éléments exclus (composition de la cour, institutions publiques, montants). "
                "5) Si 'teacher_json' est fourni, préférer des diffs sémantiquement équivalents. "
                "Retourner un score dans [0,1]."
            ),
        }
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
        ]

        payload = {
            "model": self._model,
            "messages": messages,
            "max_completion_tokens": 1024,
        }

        # Prefer OpenAI SDK if available; else fallback to requests
        try:
            if self._client is not None:
                # Use new responses.create API only for GPT-5 (not GPT-5-mini)
                # GPT-5-mini still uses the traditional chat completions API
                if self._model == "gpt-5" or self._model == "gpt-5-mini" or self._model == "gpt-5-nano":
                    # Combine system and user messages into single input for GPT-5
                    combined_input = (
                        f"{system}\n\n"
                        f"Task: {json.dumps(user, ensure_ascii=False)}"
                    )
                    
                    resp = self._client.responses.create(
                        model=self._model,
                        input=combined_input,
                        reasoning={"effort": "medium"},
                        text={"verbosity": "low"}
                    )
                    # print(f"OOOOOOO GPT-5 Response: {resp}")
                    content = resp.output_text or ""
                else:
                    # Use traditional chat completions for other models
                    resp = self._client.chat.completions.create(**payload)
                    # print(f"OOOOOOO Chat Response: {resp}")
                    content = resp.choices[0].message.content or ""
            else:
                import requests  # type: ignore
                headers = {
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                }
                url = f"{self._base_url}/chat/completions"
                r = requests.post(url, headers=headers, json=payload, timeout=self._timeout)
                r.raise_for_status()
                json_response = r.json()
                # print(f"OOOOOOO JSON Response: {json_response}")
                content = r.json()["choices"][0]["message"]["content"]
        except Exception as e:
            # Conservative fallback
            print(f"Error scoring: {e}")
            return 0.0
        # print(f"OOOOOOO Content: {content}")
        if not content.strip():
            print(f"WARNING: Empty response from judge model {self._model}. Using fallback score 0.0")
        score = parse_score_from_response(content)
        # print(f"OOOOOOO Completion: {completion}")
        # print(f"OOOOOOO Score: {score}")
        return float(np.clip(score, 0.0, 1.0))


def parse_score_from_response(text: str) -> float:
    text = text.strip()
    # Try JSON first
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "score" in obj:
            val = float(obj["score"])
            return val
    except Exception:
        pass
    # Fallback: number in text, allow 0..100 or 0..10 or 0..1
    m = re.search(r"[-+]?[0-9]*\.?[0-9]+", text)
    if m:
        val = float(m.group(0))
        # Heuristic normalization
        if val > 10:
            return val / 100.0
        if val > 1:
            return val / 10.0
        return val
    return 0.0


def make_reward_fn(judge: JudgeClient, prompt_to_reference: Dict[str, str]):
    def reward_fn(prompts: List[str], completions: List[str], **_: dict) -> List[float]:
        # print(f"Making reward fn with {len(prompts)} prompts and {len(completions)} completions")
        items: List[Tuple[str, str, Optional[str]]] = []
        for p, c in zip(prompts, completions):
            ref = prompt_to_reference.get(p, "")
            # Extract TEXT and RULES to provide compact context to the judge
            text, rules = extract_rules_and_text_from_prompt(p)
            judge_prompt = json.dumps({
                "rules": rules,
                "text": text,
            }, ensure_ascii=False)
            items.append((judge_prompt, c, ref))

        # Local validity check to short-circuit obviously invalid JSON
        local_boost: List[float] = []
        for p, c, _ in items:
            text = json.loads(p).get("text", "")
            valid, coverage = validate_diff_against_text(c, text)
            if not valid:
                local_boost.append(0.0)
            else:
                local_boost.append(coverage)

        backoff = 1.0
        scores: List[float] = [0.0] * len(items)
        for attempt in range(3):
            try:
                # print(f"Judging {len(items)} items")
                judge_scores = judge.score(items)
                if len(judge_scores) == len(items):
                    scores = [float(0.7 * js + 0.3 * lb) for js, lb in zip(judge_scores, local_boost)]
                    break
            except Exception:
                pass
            time.sleep(backoff)
            backoff *= 2
        return scores
    return reward_fn


def extract_rules_and_text_from_prompt(prompt: str) -> Tuple[str, str]:
    rules = ""
    text = ""
    m_rules = re.search(r"<<RULES>>\n([\s\S]*?)\n<<END_RULES>>", prompt)
    if m_rules:
        rules = m_rules.group(1).strip()
    m_text = re.search(r"<<TEXT>>\n([\s\S]*?)\n<<END_TEXT>>", prompt)
    if m_text:
        text = m_text.group(1)
    return text, rules


def validate_diff_against_text(candidate: str, text: str) -> Tuple[bool, float]:
    try:
        obj = json.loads(candidate)
    except Exception:
        return False, 0.0
    if not isinstance(obj, dict) or "diff" not in obj or not isinstance(obj["diff"], list):
        return False, 0.0
    total = 0
    hit = 0
    for item in obj["diff"]:
        if not isinstance(item, dict):
            continue
        searches = item.get("search")
        if isinstance(searches, list):
            for s in searches:
                if not isinstance(s, str):
                    continue
                total += 1
                if s in text:
                    hit += 1
    if total == 0:
        return True, 0.0
    return True, float(hit) / float(total)


def build_model_and_tokenizer(
    base_model: str,
    max_seq_length: int,
    lora_rank: int,
    load_in_4bit: bool,
    fast_inference: bool,
    gpu_memory_utilization: float,
):
    PatchFastRL("GRPO", FastLanguageModel)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        fast_inference=fast_inference,
        max_lora_rank=lora_rank,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
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
    num_generations: int,
    num_steps: int,
    max_prompt_len: int,
    max_completion_len: int,
    learning_rate: float,
    loss_type: str = "grpo",
    use_vllm: bool = True,
):
    # Gracefully disable vLLM if the model does not expose a colocated vLLM engine
    if use_vllm and not hasattr(model, "vllm_engine"):
        print("[warn] --use_vllm requested but model has no vllm_engine; disabling vLLM for sampling.")
        use_vllm = False
    args = GRPOConfig(
        learning_rate=learning_rate,
        weight_decay=0.0,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        logging_steps=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        num_generations=num_generations,
        max_prompt_length=max_prompt_len,
        max_completion_length=max_completion_len,
        max_steps=num_steps,
        save_steps=1000,
        report_to="none",
        output_dir=output_dir,
        use_vllm=use_vllm,
        loss_type=loss_type,
        vllm_sampling_params={
            "temperature": 1.0,
            "top_p": 1.0,
            "min_p": 0.1,
            "n": num_generations,
            "max_tokens": max_completion_len,
            "seed": 3407,
        },
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_fn],
        args=args,
        train_dataset=dataset,
    )
    return trainer


def main() -> None:
    parser = argparse.ArgumentParser(description="GRPO with Unsloth, vLLM and LLM-as-a-Judge reward")
    parser.add_argument("--base_model", type=str, default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--dataset", type=str, default="gsm8k", help="'gsm8k' or path to JSONL/JSON (ignored when --xml_dir is set)")
    parser.add_argument("--xml_dir", type=str, default=os.path.join(_PROJECT_ROOT, "data", "train"), help="Root directory containing legal XML files")
    parser.add_argument("--rules_path", type=str, default=os.path.join(_PROJECT_ROOT, "data", "rules.md"), help="Path to rules_2.md")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_chars_per_sample", type=int, default=1200000)
    parser.add_argument("--use_teacher_reference", action="store_true", help="Call anonymize.call_openai_for_diff to create references")
    parser.add_argument("--teacher_model", type=str, default=os.environ.get("TEACHER_MODEL", "gpt-5-mini"), help="OpenAI-compatible teacher model id (e.g., gpt-5)")
    parser.add_argument("--prompt_field", type=str, default="prompt")
    parser.add_argument("--reference_field", type=str, default="reference")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--max_prompt_length", type=int, default=2048)
    parser.add_argument("--max_completion_length", type=int, default=2048)
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--no_load_in_4bit", action="store_true")
    parser.add_argument("--fast_inference", action="store_true")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.6)
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--num_steps", type=int, default=None, help="Number of training steps (auto-calculated if not provided)")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs to train (used for auto-calculating steps)")
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--loss_type", type=str, default="grpo", choices=["grpo", "bnpo", "dr_grpo", "dapo"]) 
    parser.add_argument("--use_vllm", action="store_true")
    parser.add_argument("--judge_model", type=str, required=True)
    parser.add_argument("--judge_base_url", type=str, default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--judge_api_key", type=str, default=os.environ.get("OPENAI_API_KEY", ""))
    
    # New production features
    parser.add_argument("--test_dir", type=str, default=os.path.join(_PROJECT_ROOT, "data", "test"), help="Directory with test XML files")
    parser.add_argument("--keep_n_checkpoints", type=int, default=3, help="Number of checkpoints to keep")
    parser.add_argument("--eval_every_n_steps", type=int, default=50, help="Run test evaluation every N steps")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience (epochs)")
    parser.add_argument("--min_improvement", type=float, default=0.001, help="Minimum improvement for early stopping")
    args = parser.parse_args()

    set_reproducibility(3407)

    load_in_4bit = True
    if args.no_load_in_4bit:
        load_in_4bit = False
    elif args.load_in_4bit:
        load_in_4bit = True
        
    print(f"Loading model {args.base_model} with load_in_4bit={load_in_4bit} and fast_inference={args.fast_inference or args.use_vllm}")

    model, tokenizer = build_model_and_tokenizer(
        base_model=args.base_model,
        max_seq_length=args.max_seq_length,
        lora_rank=args.lora_rank,
        load_in_4bit=load_in_4bit,
        fast_inference=args.fast_inference or args.use_vllm,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    print(f"Model loaded with {model.num_parameters()} parameters")
    
    print(f"Building dataset with {args.xml_dir} and {args.rules_path}")
    
    # Build dataset
    if args.xml_dir:
        if not args.rules_path:
            raise ValueError("--rules_path is required when --xml_dir is provided")
        rules_text = read_rules_text(args.rules_path)
        teacher_kwargs = {"model": args.teacher_model}
        ds = build_xml_anonymization_dataset(
            xml_dir=args.xml_dir,
            rules_text=rules_text,
            max_samples=args.max_samples,
            max_chars_per_sample=args.max_chars_per_sample,
            use_teacher_reference=args.use_teacher_reference,
            teacher_call_kwargs=teacher_kwargs,
        )
    else:
        if args.dataset.lower() == "gsm8k":
            ds = build_gsm8k(args.split)
        else:
            # Expect a JSON/JSONL dataset with fields provided
            ds = load_custom_text_dataset(args.dataset, args.prompt_field, args.reference_field)
    print(f"Dataset loaded with {len(ds)} samples") 
    
    # Auto-calculate num_steps if not provided
    auto_calculated = False
    if args.num_steps is None:
        samples_per_epoch = len(ds)
        if samples_per_epoch == 0:
            raise ValueError("Dataset is empty! Cannot calculate num_steps. Please check your data paths.")
        
        args.num_steps = samples_per_epoch * args.num_epochs
        auto_calculated = True
        print(f"✨ Auto-calculated num_steps: {args.num_steps} ({samples_per_epoch} samples × {args.num_epochs} epochs)")
    else:
        print(f"📋 Using specified num_steps: {args.num_steps}")
    
    print(f"Building lookup for reference answers")
    # Build lookup for reference answers
    prompt_to_ref: Dict[str, str] = {}
    for row in ds:
        prompt_to_ref[row["prompt"]] = row.get("reference", "")

    if not args.judge_api_key:
        raise ValueError("Judge API key is required (set --judge_api_key or OPENAI_API_KEY)")

    # Initialize console for beautiful output
    console = Console() if RICH_AVAILABLE else None
    
    if RICH_AVAILABLE:
        steps_info = f"Steps: {args.num_steps}"
        if auto_calculated:
            steps_info += f" (auto: {len(ds)} × {args.num_epochs})"
            
        console.print(Panel(
            f"🚀 Starting GRPO Training\n"
            f"Model: {args.base_model}\n"
            f"Dataset: {len(ds)} training samples\n"
            f"Judge: {args.judge_model}\n"
            f"{steps_info}\n"
            f"Epochs: {args.num_epochs}\n"
            f"Output: {args.output_dir}",
            title="Training Configuration",
            style="bold blue"
        ))
    
    judge = JudgeClient(
        base_url=args.judge_base_url,
        api_key=args.judge_api_key,
        model=args.judge_model,
    )

    # Initialize production components
    metrics_logger = MetricsLogger(args.output_dir, console)
    checkpoint_manager = CheckpointManager(args.output_dir, args.keep_n_checkpoints)
    
    # Setup test evaluator
    test_evaluator = TestEvaluator(
        test_dir=args.test_dir,
        rules_text=rules_text if args.xml_dir else "",
        judge=judge,
        tokenizer=tokenizer,
        console=console
    )
    
    # Initialize training callback
    training_callback = TrainingCallback(
        test_evaluator=test_evaluator,
        metrics_logger=metrics_logger,
        checkpoint_manager=checkpoint_manager,
        console=console
    )
    
    reward_fn = make_reward_fn(judge, prompt_to_ref)

    trainer = build_trainer(
        model=model,
        tokenizer=tokenizer,
        dataset=ds,
        reward_fn=reward_fn,
        output_dir=args.output_dir,
        num_generations=args.num_generations,
        num_steps=args.num_steps,
        max_prompt_len=args.max_prompt_length,
        max_completion_len=args.max_completion_length,
        learning_rate=args.learning_rate,
        loss_type=args.loss_type,
        use_vllm=args.use_vllm,
    )
    print(f"Trainer built with {model.num_parameters()} parameters")
    
    # Enhanced training with monitoring
    if RICH_AVAILABLE:
        console.print("[bold green]🎯 Starting training with enhanced monitoring...[/bold green]")
    
    # Initialize early stopping
    best_test_score = -1.0
    patience_counter = 0
    
    # Training loop with callbacks
    try:
        # We'll need to implement a custom training loop to integrate callbacks
        # For now, let's use the standard trainer with post-processing
        
        step_count = 0
        last_logged_step = -1
        
        # Monkey patch the trainer to capture metrics
        original_log = trainer._maybe_log_save_evaluate if hasattr(trainer, '_maybe_log_save_evaluate') else None
        
        def enhanced_log_save_evaluate(_self_trainer, _state, _control, _metrics, _tr_loss_or_something, _another_arg, _yet_another_arg, **kwargs):
            nonlocal step_count, last_logged_step, best_test_score, patience_counter
            
            # Call original if exists
            if original_log:
                original_log(_self_trainer, _state, _control, _metrics, _tr_loss_or_something, _another_arg, _yet_another_arg, **kwargs)
            
            if _metrics is None:
                _metrics = {}
            current_step = _metrics.get('step', 0)
            
            # Run evaluation every N steps
            if current_step > last_logged_step and (current_step % args.eval_every_n_steps == 0 or current_step == args.num_steps):
                last_logged_step = current_step
                
                # Run callback
                training_callback.on_epoch_end(model, tokenizer, _metrics)
                
                # Early stopping check
                test_score = _metrics.get('test_score', 0)
                if test_score > best_test_score + args.min_improvement:
                    best_test_score = test_score
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= args.patience:
                    if RICH_AVAILABLE:
                        console.print(f"[bold red]🛑 Early stopping triggered! No improvement for {args.patience} evaluations[/bold red]")
                    else:
                        print(f"Early stopping triggered! No improvement for {args.patience} evaluations")
                    
                    # Save final model
                    final_checkpoint_dir = Path(args.output_dir) / "final_model"
                    final_checkpoint_dir.mkdir(exist_ok=True)
                    model.save_pretrained(str(final_checkpoint_dir))
                    tokenizer.save_pretrained(str(final_checkpoint_dir))
                    
                    return False  # Signal to stop training
            
            return True
        
        # Patch the trainer
        if hasattr(trainer, '_maybe_log_save_evaluate'):
            trainer._maybe_log_save_evaluate = enhanced_log_save_evaluate
        
        trainer.train()
        
    except KeyboardInterrupt:
        if RICH_AVAILABLE:
            console.print("[bold yellow]⚠️  Training interrupted by user[/bold yellow]")
        else:
            print("Training interrupted by user")
        
        # Save current state
        interrupt_dir = Path(args.output_dir) / "interrupted_model"
        interrupt_dir.mkdir(exist_ok=True)
        model.save_pretrained(str(interrupt_dir))
        tokenizer.save_pretrained(str(interrupt_dir))
        
    except Exception as e:
        if RICH_AVAILABLE:
            console.print(f"[bold red]💥 Training failed: {e}[/bold red]")
        else:
            print(f"Training failed: {e}")
        
        # Save current state
        error_dir = Path(args.output_dir) / "error_recovery_model"
        error_dir.mkdir(exist_ok=True)
        model.save_pretrained(str(error_dir))
        tokenizer.save_pretrained(str(error_dir))
        raise
    
    # Final evaluation and summary
    if RICH_AVAILABLE:
        console.print("[bold green]🎉 Training completed![/bold green]")
    
    final_test_score = test_evaluator.evaluate(model)
    
    # Save final model
    os.makedirs(args.output_dir, exist_ok=True)
    final_model_dir = os.path.join(args.output_dir, "final_model")
    model.save_pretrained(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    
    if RICH_AVAILABLE:
        summary_table = Table(title="🏁 Training Summary", show_header=True, header_style="bold blue")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="green")
        
        summary_table.add_row("Best Test Score", f"{checkpoint_manager.best_score:.4f}")
        summary_table.add_row("Final Test Score", f"{final_test_score:.4f}")
        summary_table.add_row("Total Steps", str(args.num_steps))
        summary_table.add_row("Model Parameters", f"{model.num_parameters():,}")
        summary_table.add_row("Best Model", str(checkpoint_manager.best_checkpoint_path) if checkpoint_manager.best_checkpoint_path else "N/A")
        
        console.print(summary_table)
    
    print(f"Training completed! Final model saved to {final_model_dir}")
    print(f"Metrics saved to {metrics_logger.csv_path}")
    print(f"Best model available at: {args.output_dir}/best_model")


if __name__ == "__main__":
    main()

