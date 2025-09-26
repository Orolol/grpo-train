#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Modular training approach for XML anonymization with progressive complexity.

This script implements a 3-stage training pipeline:
1. Entity Detection (SFT) - Learn to identify entities
2. Rule Application (SFT) - Apply anonymization rules to detected entities
3. Refinement (GRPO) - Optional fine-tuning with external judge

Example:
  python modular_training.py \
    --stage 1 \
    --base_model mistralai/Mistral-7B-Instruct-v0.2 \
    --output_dir out_modular \
    --train_dir data/train \
    --test_dir data/test
"""

from __future__ import annotations

# Configure CUDA allocator before importing torch/unsloth
import os as _os
_alloc_conf = _os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
if "expandable_segments" in _alloc_conf.lower():
    _alloc_conf = ";".join(
        p for p in _alloc_conf.split(";")
        if not p.strip().lower().startswith("expandable_segments")
    )
    if _alloc_conf:
        _os.environ["PYTORCH_CUDA_ALLOC_CONF"] = _alloc_conf
    else:
        _os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
else:
    _os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
        (_alloc_conf + ";") if _alloc_conf else ""
    ) + "expandable_segments:False"

_os.environ.setdefault("UNSLOTH_VLLM_STANDBY", "1")

from unsloth import FastLanguageModel, PatchFastRL

import argparse
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch
from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer, GRPOConfig, GRPOTrainer


def _tokenize_for_generation(tokenizer, prompt: str, max_length: int):
    """Tokenize prompt for generation, with fallbacks for multimodal processors."""
    kwargs = {
        "return_tensors": "pt",
        "truncation": True,
        "max_length": max_length,
    }

    try:
        return tokenizer(prompt, **kwargs)
    except ValueError as exc:
        msg = str(exc)
        if "Invalid input images" not in msg:
            raise

        # Some processors (e.g. Pixtral) require an explicit tokenizer for text-only inputs
        text_tokenizer = getattr(tokenizer, "tokenizer", None) or getattr(tokenizer, "text_tokenizer", None)
        if text_tokenizer is None:
            raise

        return text_tokenizer(prompt, **kwargs)


def _extract_text_from_stage3_prompt(prompt: str) -> str:
    """Extract the source text block embedded in the Stage 3 prompt."""
    match = re.search(r"Texte:\n(.*?)\n\nSortie JSON:", prompt, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


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
    """Extract the most relevant section from XML for processing."""
    # Prefer <NonStructure>, else <TXD>, else full file
    m = re.search(r"<NonStructure>[\s\S]*?</NonStructure>", xml_text)
    if m:
        return m.group(0)
    m = re.search(r"<TXD>[\s\S]*?</TXD>", xml_text)
    if m:
        return m.group(0)
    return xml_text


def extract_entities_from_xml(xml_text: str) -> List[Tuple[str, str]]:
    """Extract potential entities from XML text for training data."""
    entities = []

    # Extract text content from AL tags
    text_content = ""
    for match in re.finditer(r"<AL>(.*?)</AL>", xml_text, re.DOTALL):
        text_content += match.group(1) + " "

    # Find potential names (simplified pattern)
    name_patterns = [
        r"(?:M\.|Mme|Mlle|Me|sieur)\s+([A-Z][a-zé]+(?:\s+[A-Z][a-zé]+)*)",
        r"([A-Z][a-zé]+\s+[A-Z][a-zé]+)(?:\s+c/|\s+contre)",
        r"demandeur:\s*([A-Z][a-zé]+(?:\s+[A-Z][a-zé]+)*)",
    ]
    for pattern in name_patterns:
        for match in re.finditer(pattern, text_content):
            entities.append((match.group(1), "nom"))

    # Find addresses (simplified)
    addr_pattern = r"(\d+\s+(?:rue|avenue|boulevard|cours)[^,]+,\s*\d{5}\s+\w+)"
    for match in re.finditer(addr_pattern, text_content, re.IGNORECASE):
        entities.append((match.group(1), "adresse"))

    # Find dates
    date_pattern = r"(\d{1,2}/\d{1,2}/\d{4})"
    for match in re.finditer(date_pattern, text_content):
        entities.append((match.group(1), "date"))

    # Find phone numbers
    phone_pattern = r"(\d{2}\s*\d{2}\s*\d{2}\s*\d{2}\s*\d{2})"
    for match in re.finditer(phone_pattern, text_content):
        entities.append((match.group(1), "telephone"))

    return entities


def set_seed(seed: int = 3407) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


# -----------------------------
# Stage 1: Entity Detection
# -----------------------------

STAGE1_TEMPLATE = """Tâche: Identifier les entités personnelles dans ce texte juridique.

Instructions:
- Identifie TOUS les noms de personnes physiques (pas les sociétés)
- Identifie les adresses complètes
- Identifie les dates de naissance/mariage/décès
- Identifie les numéros de téléphone et emails
- Format de sortie: liste markdown simple

Texte:
{text}

Entités identifiées:
"""

STAGE1_OUTPUT_FORMAT = """
## Personnes
- {name1} (nom complet)
- {name2} (nom complet)

## Adresses
- {addr1}
- {addr2}

## Dates personnelles
- {date1} (naissance/mariage/décès)

## Contacts
- {phone1} (téléphone)
- {email1} (email)
"""


def build_stage1_dataset(
    xml_dir: str,
    tokenizer,
    max_samples: Optional[int] = None,
) -> Dataset:
    """Build dataset for Stage 1: Entity Detection."""
    rows = []

    for fp in list_xml_files(xml_dir):
        try:
            xml = read_file(fp)
            section = extract_relevant_section(xml)

            # Create input prompt
            prompt = STAGE1_TEMPLATE.format(text=section)

            # Generate expected output from extracted entities
            entities = extract_entities_from_xml(section)

            # Format entities into markdown
            output_lines = []

            # Group by type
            names = [e for e in entities if e[1] == "nom"]
            addrs = [e for e in entities if e[1] == "adresse"]
            dates = [e for e in entities if e[1] == "date"]
            phones = [e for e in entities if e[1] == "telephone"]

            if names:
                output_lines.append("## Personnes")
                for name, _ in names[:5]:  # Limit to avoid too long outputs
                    output_lines.append(f"- {name} (nom complet)")

            if addrs:
                output_lines.append("\n## Adresses")
                for addr, _ in addrs[:3]:
                    output_lines.append(f"- {addr}")

            if dates:
                output_lines.append("\n## Dates personnelles")
                for date, _ in dates[:3]:
                    output_lines.append(f"- {date} (date)")

            if phones:
                output_lines.append("\n## Contacts")
                for phone, _ in phones[:3]:
                    output_lines.append(f"- {phone} (téléphone)")

            if output_lines:
                response = "\n".join(output_lines)
                rows.append({
                    "prompt": prompt,
                    "response": response,
                    "text": prompt + "\n" + response
                })

            if max_samples and len(rows) >= max_samples:
                break

        except Exception as e:
            print(f"Error processing {fp}: {e}")
            continue

    return Dataset.from_list(rows)


# -----------------------------
# Stage 2: Rule Application
# -----------------------------

STAGE2_TEMPLATE = """Tâche: Appliquer les règles d'anonymisation aux entités identifiées.

Règles d'anonymisation:
{rules}

Entités identifiées:
{entities}

Texte original:
{text}

Transformations à appliquer:
"""

STAGE2_OUTPUT_FORMAT = """
## Transformations

### Noms
- "Jean Dupont" → "J. D."
- "Marie Martin" → "M. M."

### Adresses
- "12 rue de la Paix, 75001 Paris" → "[Adresse 1]"
- "45 avenue Victor Hugo, 69000 Lyon" → "[Adresse 2]"

### Dates
- "15/03/1985" → "[Date 1]"

### Contacts
- "06 12 34 56 78" → "[Téléphone 1]"
- "jean.dupont@email.fr" → "[Courriel 1]"
"""


def build_stage2_dataset(
    xml_dir: str,
    rules_text: str,
    max_samples: Optional[int] = None,
) -> Dataset:
    """Build dataset for Stage 2: Rule Application."""
    rows = []

    for fp in list_xml_files(xml_dir):
        try:
            xml = read_file(fp)
            section = extract_relevant_section(xml)

            # Extract entities using rule-based approach (not model)
            entities = extract_entities_from_xml(section)

            # Format entities into markdown for Stage 2 input
            entities_lines = []
            names = [e for e in entities if e[1] == "nom"]
            addrs = [e for e in entities if e[1] == "adresse"]
            dates = [e for e in entities if e[1] == "date"]
            phones = [e for e in entities if e[1] == "telephone"]

            if names:
                entities_lines.append("## Personnes")
                for name, _ in names[:5]:
                    entities_lines.append(f"- {name} (nom complet)")

            if addrs:
                entities_lines.append("\n## Adresses")
                for addr, _ in addrs[:3]:
                    entities_lines.append(f"- {addr}")

            if dates:
                entities_lines.append("\n## Dates personnelles")
                for date, _ in dates[:3]:
                    entities_lines.append(f"- {date} (date)")

            if phones:
                entities_lines.append("\n## Contacts")
                for phone, _ in phones[:3]:
                    entities_lines.append(f"- {phone} (téléphone)")

            entities_detected = "\n".join(entities_lines) if entities_lines else "Aucune entité détectée."

            # Build Stage 2 prompt
            prompt = STAGE2_TEMPLATE.format(
                rules=rules_text,
                entities=entities_detected,
                text=section[:1000]  # Truncate for context
            )

            # Generate expected transformations based on rules
            output_lines = ["## Transformations"]

            # Apply rules to generate transformations
            if names:
                output_lines.append("\n### Noms")
                for name, _ in names[:5]:
                    parts = name.split()
                    if len(parts) >= 2:
                        initials = f"{parts[0][0]}. {parts[-1][0]}."
                        output_lines.append(f'- "{name}" → "{initials}"')

            if addrs:
                output_lines.append("\n### Adresses")
                for i, (addr, _) in enumerate(addrs[:3], 1):
                    output_lines.append(f'- "{addr}" → "[Adresse {i}]"')

            if dates:
                output_lines.append("\n### Dates")
                for i, (date, _) in enumerate(dates[:3], 1):
                    output_lines.append(f'- "{date}" → "[Date {i}]"')

            if phones:
                output_lines.append("\n### Contacts")
                for i, (phone, _) in enumerate(phones[:3], 1):
                    output_lines.append(f'- "{phone}" → "[Téléphone {i}]"')

            # Only create training sample if we have transformations
            if len(output_lines) > 1:
                response = "\n".join(output_lines)
                rows.append({
                    "prompt": prompt,
                    "response": response,
                    "text": prompt + "\n" + response
                })

            if max_samples and len(rows) >= max_samples:
                break

        except Exception as e:
            print(f"Error processing {fp}: {e}")
            continue

    return Dataset.from_list(rows)


# -----------------------------
# Stage 3: GRPO Refinement
# -----------------------------

STAGE3_TEMPLATE = """Tâche: Générer le JSON d'anonymisation final.

Règles:
{rules}

Texte:
{text}

Sortie JSON:
"""


class JudgeClient:
    """Simplified judge client from grpo_simple.py"""
    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 60.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self._client = None
        try:
            from openai import OpenAI
            self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        except Exception:
            raise ImportError("The openai package is required for JudgeClient")

    def score_batch_concurrent(self, prompts: List[str], concurrency: int = 8) -> List[float]:
        """Score a batch of outputs using the judge model."""
        scores = []
        for prompt in prompts:
            try:
                resp = self._client.responses.create(
                    model=self.model,
                    input=prompt,
                    reasoning={"effort": "medium"},
                    text={"verbosity": "low"},
                )
                content = self._extract_response_text(resp)
                scores.append(self._parse_score(content))
            except Exception as e:
                print(f"Judge error: {e}")
                scores.append(0.0)
        return scores

    @staticmethod
    def _extract_response_text(resp: Any) -> str:
        text = getattr(resp, "output_text", None)
        if text:
            return text
        outputs = getattr(resp, "output", None)
        if not outputs:
            return ""
        parts: List[str] = []
        for item in outputs:
            for content in getattr(item, "content", []) or []:
                segment = getattr(content, "text", None)
                if segment:
                    parts.append(getattr(segment, "value", ""))
        return "\n".join(parts)

    @staticmethod
    def _parse_score(text: str) -> float:
        t = text.strip()
        if not t:
            return 0.0
        try:
            obj = json.loads(t)
            if isinstance(obj, dict) and "score" in obj:
                val = float(obj["score"])
                return float(np.clip(val, 0.0, 1.0))
        except Exception:
            pass
        match = re.search(r"[-+]?\d*\.?\d+", t)
        if not match:
            return 0.0
        val = float(match.group(0))
        if val > 10:
            val /= 100.0
        elif val > 1:
            val /= 10.0
        return float(np.clip(val, 0.0, 1.0))


# -----------------------------
# Model Building
# -----------------------------

def build_model_tokenizer(
    base_model: str,
    max_seq_length: int,
    lora_rank: int,
    load_in_4bit: bool,
    stage: int,
    checkpoint_dir: Optional[str] = None,
):
    """Build model and tokenizer for a specific stage."""

    if stage == 3:
        # Stage 3 uses GRPO
        PatchFastRL("GRPO", FastLanguageModel)

    # Determine if we're loading from checkpoint or base model
    is_checkpoint = checkpoint_dir and os.path.exists(checkpoint_dir)
    model_path = checkpoint_dir if is_checkpoint else base_model

    model, tokenizer = FastLanguageModel.from_pretrained(
        dtype=None,
        model_name=model_path,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
    )

    # Only add LoRA adapters if we're loading from base model (not checkpoint)
    if not is_checkpoint:
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_rank,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_alpha=16,
            use_gradient_checkpointing=True,
            lora_dropout=0,
            random_state=3407,
        )
    else:
        # If loading from checkpoint, enable training
        FastLanguageModel.for_training(model)

    return model, tokenizer


# -----------------------------
# Training Functions
# -----------------------------

def train_stage1(args):
    """Train Stage 1: Entity Detection"""
    print("=== Stage 1: Entity Detection ===")

    # Build model
    model, tokenizer = build_model_tokenizer(
        base_model=args.base_model,
        max_seq_length=args.max_seq_length,
        lora_rank=args.lora_rank,
        load_in_4bit=args.load_in_4bit,
        stage=1,
        checkpoint_dir=None,
    )

    # Build dataset
    print("Building Stage 1 dataset...")
    train_dataset = build_stage1_dataset(
        xml_dir=args.train_dir,
        tokenizer=tokenizer,
        max_samples=args.max_samples,
    )

    if len(train_dataset) == 0:
        raise RuntimeError("No training samples found")

    print(f"Training on {len(train_dataset)} samples")
    

    # Training arguments
    output_dir = os.path.join(args.output_dir, "stage1")
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_ratio=0.1,
        learning_rate=args.learning_rate,
        fp16=not args.load_in_4bit,
        bf16=args.load_in_4bit,
        logging_steps=1,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        optim="adamw_8bit",
    )

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        args=training_args,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        packing=False,
    )

    # Train
    trainer.train()

    # Save model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Stage 1 model saved to {output_dir}")

    evaluate_diffs_with_judge(model, tokenizer, args, stage_label="Stage 1")

    return output_dir


def train_stage2(args, stage1_checkpoint: Optional[str] = None):
    """Train Stage 2: Rule Application"""
    print("=== Stage 2: Rule Application ===")

    # Use Stage 1 checkpoint if available
    checkpoint = stage1_checkpoint or os.path.join(args.output_dir, "stage1")

    # Build model
    model, tokenizer = build_model_tokenizer(
        base_model=args.base_model,
        max_seq_length=args.max_seq_length,
        lora_rank=args.lora_rank,
        load_in_4bit=args.load_in_4bit,
        stage=2,
        checkpoint_dir=checkpoint if os.path.exists(checkpoint) else None,
    )

    # Load rules
    rules_text = read_file(args.rules_path)

    # Build dataset (using rule-based entity extraction)
    print("Building Stage 2 dataset...")
    train_dataset = build_stage2_dataset(
        xml_dir=args.train_dir,
        rules_text=rules_text,
        max_samples=args.max_samples,
    )

    if len(train_dataset) == 0:
        raise RuntimeError("No training samples found")

    print(f"Training on {len(train_dataset)} samples")

    # Training arguments
    output_dir = os.path.join(args.output_dir, "stage2")
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        warmup_ratio=0.1,
        learning_rate=args.learning_rate / 2,  # Lower LR for fine-tuning
        fp16=not args.load_in_4bit,
        bf16=args.load_in_4bit,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        optim="adamw_8bit",
    )

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        args=training_args,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        packing=False,
    )

    # Train
    trainer.train()

    # Save model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Stage 2 model saved to {output_dir}")

    evaluate_diffs_with_judge(model, tokenizer, args, stage_label="Stage 2")

    return output_dir


def train_stage3(args, stage2_checkpoint: Optional[str] = None):
    """Train Stage 3: GRPO Refinement"""
    print("=== Stage 3: GRPO Refinement ===")

    # Use Stage 2 checkpoint if available
    checkpoint = stage2_checkpoint or os.path.join(args.output_dir, "stage2")

    # Build model
    model, tokenizer = build_model_tokenizer(
        base_model=args.base_model,
        max_seq_length=args.max_seq_length,
        lora_rank=args.lora_rank,
        load_in_4bit=args.load_in_4bit,
        stage=3,
        checkpoint_dir=checkpoint if os.path.exists(checkpoint) else None,
    )

    # Load templates
    rules_text = read_file(args.rules_path)
    if not os.path.exists(args.judge_prompt):
        raise FileNotFoundError(f"Judge prompt not found: {args.judge_prompt}")
    judge_template = read_file(args.judge_prompt)
    print(f"Stage 3 utilisera le prompt juge: {args.judge_prompt}")

    # Build simplified dataset for GRPO
    rows = []
    for fp in list_xml_files(args.train_dir):
        try:
            xml = read_file(fp)
            section = extract_relevant_section(xml)
            prompt = STAGE3_TEMPLATE.format(rules=rules_text, text=section)
            rows.append({"prompt": prompt})
            if args.max_samples and len(rows) >= args.max_samples:
                break
        except Exception:
            continue

    train_dataset = Dataset.from_list(rows)

    if len(train_dataset) == 0:
        raise RuntimeError("No training samples found")

    print(f"Training on {len(train_dataset)} samples")
    
    print(f"Sample 0:\nPrompt:\n{train_dataset[0]}")


    # Setup judge
    judge = JudgeClient(
        base_url=args.judge_base_url,
        api_key=args.judge_api_key,
        model=args.judge_model,
    )

    # Simple reward function
    def reward_fn(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
        scores = []
        for completion in completions:
            # Check if valid JSON
            try:
                json.loads(completion)
                # If valid JSON, give base score
                scores.append(0.5)
            except:
                scores.append(0.0)

        # Optional: use judge for better scoring
        if args.use_judge and judge_template:
            judge_prompts = []
            for prm, comp in zip(prompts, completions):
                source_text = _extract_text_from_stage3_prompt(prm)
                judge_prompts.append(
                    judge_template.format(
                        rules=rules_text,
                        text=source_text or "[texte indisponible]",
                        candidate=comp,
                    )
                )

            judge_scores = judge.score_batch_concurrent(judge_prompts, concurrency=4)
            # Combine structural and judge scores
            scores = [0.3 * s + 0.7 * j for s, j in zip(scores, judge_scores)]

        return scores

    # GRPO config
    output_dir = os.path.join(args.output_dir, "stage3")
    grpo_config = GRPOConfig(
        learning_rate=args.learning_rate / 4,  # Even lower for GRPO
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        logging_steps=1,
        auto_find_batch_size=True,
        gradient_accumulation_steps=8,
        num_generations=4,
        max_prompt_length=2048,
        max_completion_length=512,
        num_train_epochs=args.epochs,
        report_to="none",
        output_dir=output_dir,
        loss_type="grpo",
    )

    # Create trainer
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_fn],
        args=grpo_config,
        train_dataset=train_dataset,
    )

    # Train
    trainer.train()

    # Save model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Stage 3 model saved to {output_dir}")

    evaluate_diffs_with_judge(model, tokenizer, args, stage_label="Stage 3")

    return output_dir


# -----------------------------
# Evaluation
# -----------------------------

@torch.no_grad()
def evaluate_diffs_with_judge(
    model,
    tokenizer,
    args,
    stage_label: str,
) -> None:
    """Generate anonymisation diffs on test XMLs and score them with the judge."""

    test_files = list_xml_files(args.test_dir)
    if not test_files:
        print(f"[{stage_label}] Aucun fichier de test trouvé dans {args.test_dir}, évaluation sautée.")
        return

    rules_text = read_file(args.rules_path)
    judge_template = ""
    judge: Optional[JudgeClient] = None
    prompt_path = args.judge_prompt
    has_prompt = bool(prompt_path and os.path.exists(prompt_path))
    has_key = bool(args.judge_api_key)
    judge_enabled = has_key and has_prompt

    if judge_enabled:
        try:
            judge_template = read_file(prompt_path)
            judge = JudgeClient(
                base_url=args.judge_base_url,
                api_key=args.judge_api_key,
                model=args.judge_model,
            )
            print(f"[{stage_label}] Juge initialisé avec le prompt {prompt_path}.")
        except Exception as exc:
            print(f"[{stage_label}] Impossible d'initialiser le juge ({exc}), scores non disponibles.")
            judge_enabled = False
    else:
        if not has_key:
            print(f"[{stage_label}] Pas de clé API fournie: le juge ne sera pas lancé.")
        if not has_prompt:
            print(f"[{stage_label}] Prompt de juge introuvable: {prompt_path}")

    model_was_training = model.training
    model.eval()

    eval_samples = args.eval_samples or len(test_files)
    selected_files = test_files[:eval_samples]

    generated_records: List[Tuple[str, str, str]] = []  # (filename, completion, context_text)
    sample_diff_printed = False
    per_prompt_durations: List[float] = []

    for fp in selected_files:
        xml = read_file(fp)
        section = extract_relevant_section(xml)
        context = section

        prompt = STAGE3_TEMPLATE.format(rules=rules_text, text=context)

        start_time = time.perf_counter()
        inputs = _tokenize_for_generation(tokenizer, prompt, args.max_seq_length)
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

        elapsed = time.perf_counter() - start_time
        per_prompt_durations.append(elapsed)

        completion_tokens = outputs[0][len(inputs["input_ids"][0]):]
        completion = tokenizer.decode(completion_tokens, skip_special_tokens=True).strip()

        filename = os.path.basename(fp)
        generated_records.append((filename, completion, context))

        if not sample_diff_printed:
            print(f"[{stage_label}] Exemple de diff généré pour {filename}:\n{completion}\n")
            sample_diff_printed = True

    if judge_enabled and judge and judge_template and generated_records:
        judge_prompts = [
            judge_template.format(rules=rules_text, text=context, candidate=completion)
            for _, completion, context in generated_records
        ]
        scores = judge.score_batch_concurrent(judge_prompts, concurrency=4)

        print(f"[{stage_label}] Scores du juge:")
        for (filename, _, _), score in zip(generated_records, scores):
            print(f"  - {filename}: {score:.3f}")

        if scores:
            avg_score = sum(scores) / len(scores)
            print(f"[{stage_label}] Score moyen: {avg_score:.3f}")

    if model_was_training:
        model.train()

    if per_prompt_durations:
        avg_time = sum(per_prompt_durations) / len(per_prompt_durations)
        print(f"[{stage_label}] Vitesse moyenne: {avg_time:.2f} s / prompt (n={len(per_prompt_durations)})")


# -----------------------------
# CLI
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="Modular training for XML anonymization")

    # Stage selection
    parser.add_argument("--stage", type=int, choices=[1, 2, 3], default=None,
                       help="Training stage (1=entity detection, 2=rule application, 3=GRPO refinement)")
    parser.add_argument("--all_stages", action="store_true",
                       help="Train all stages sequentially")

    # Data
    parser.add_argument("--train_dir", type=str, default="data/train")
    parser.add_argument("--test_dir", type=str, default="data/test")
    parser.add_argument("--rules_path", type=str, default="data/rules.md")
    parser.add_argument("--judge_prompt", type=str, default="data/judge_prompt.md")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--eval_samples", type=int, default=3,
                       help="Nombre de fichiers de test à évaluer après chaque stage")

    # Model
    parser.add_argument("--base_model", type=str, default="unsloth/Qwen3-4B-Thinking-2507-unsloth-bnb-4bit")
    parser.add_argument("--output_dir", type=str, default="output_modular")
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--load_in_4bit", action="store_true")

    # Training
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=2e-4)

    # Judge (for Stage 3)
    parser.add_argument("--use_judge", action="store_true")
    parser.add_argument("--judge_model", type=str, default="gpt-5")
    parser.add_argument("--judge_base_url", type=str,
                       default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--judge_api_key", type=str,
                       default=os.environ.get("OPENAI_API_KEY", ""))

    # Evaluation
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--checkpoint_dir", type=str, help="Checkpoint to evaluate")

    args = parser.parse_args()

    # Set random seed
    set_seed(3407)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Evaluation mode
    if args.eval_only:
        if not args.checkpoint_dir or not args.stage:
            raise ValueError("--checkpoint_dir and --stage required for evaluation")

        model, tokenizer = build_model_tokenizer(
            base_model=args.base_model,
            max_seq_length=args.max_seq_length,
            lora_rank=args.lora_rank,
            load_in_4bit=args.load_in_4bit,
            stage=args.stage,
            checkpoint_dir=args.checkpoint_dir,
        )

        stage_label = f"Stage {args.stage} (eval-only)"
        evaluate_diffs_with_judge(model, tokenizer, args, stage_label=stage_label)
        return

    # Training mode
    if args.all_stages:
        # Train all stages sequentially
        print("Training all stages sequentially...")

        stage1_dir = train_stage1(args)
        print(f"\nStage 1 complete. Checkpoint: {stage1_dir}\n")

        stage2_dir = train_stage2(args, stage1_dir)
        print(f"\nStage 2 complete. Checkpoint: {stage2_dir}\n")

        if args.use_judge and args.judge_api_key:
            stage3_dir = train_stage3(args, stage2_dir)
            print(f"\nStage 3 complete. Checkpoint: {stage3_dir}\n")
        else:
            print("Skipping Stage 3 (GRPO) - no judge API key provided")

        print("\nAll stages complete!")

    elif args.stage:
        # Train specific stage
        if args.stage == 1:
            train_stage1(args)
        elif args.stage == 2:
            train_stage2(args)
        elif args.stage == 3:
            if not args.judge_api_key:
                raise ValueError("Judge API key required for Stage 3 (GRPO)")
            train_stage3(args)
    else:
        print("Please specify --stage or --all_stages")


if __name__ == "__main__":
    main()
