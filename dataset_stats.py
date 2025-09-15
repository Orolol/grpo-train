#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute token length stats for samples built from a directory of legal XML files.

Features
- Scans an input directory (default: data/train_full) for *.xml
- Extracts a relevant section (<NonStructure> else <TXD> else full file)
- Chunks by paragraph tags (<AL>/<P>) or by character window to form samples
- Optionally formats samples with a learner prompt + rules
- Tokenizes each sample with the Mistral tokenizer and prints stats

Examples
  # Raw text stats (no prompt formatting)
  python dataset_stats.py

  # With learner prompt + rules
  python dataset_stats.py \
    --prompt_path data/learner_prompt.md \
    --rules_path data/rules_2.md
"""

from __future__ import annotations

import argparse
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from transformers import AutoTokenizer


# -----------------------------
# I/O helpers
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


def extract_relevant_section(xml_text: str) -> str:
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


# -----------------------------
# Dataset assembly
# -----------------------------

@dataclass
class Sample:
    file: str
    chunk_idx: int
    text: str
    prompt: str


def build_samples(
    data_dir: str,
    max_chars_per_sample: int,
    learner_template: Optional[str],
    rules_text: Optional[str],
) -> List[Sample]:
    samples: List[Sample] = []
    for fp in list_xml_files(data_dir):
        try:
            xml = read_text(fp)
        except Exception:
            continue
        section = extract_relevant_section(xml)
        chunks = chunk_text(section, max_chars_per_sample)
        for i, chunk in enumerate(chunks):
            if learner_template and rules_text is not None:
                prompt = learner_template.format(rules=rules_text, text=chunk)
            else:
                prompt = chunk
            samples.append(Sample(file=fp, chunk_idx=i, text=chunk, prompt=prompt))
    return samples


# -----------------------------
# Stats
# -----------------------------

def describe_lengths(lengths: List[int]) -> Dict[str, float]:
    arr = np.array(lengths, dtype=np.int64)
    if arr.size == 0:
        return {"count": 0}
    desc = {
        "count": int(arr.size),
        "min": int(arr.min()),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": int(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
    }
    return desc


def count_over_thresholds(lengths: List[int], thresholds: Iterable[int]) -> Dict[int, int]:
    res: Dict[int, int] = {}
    for t in thresholds:
        res[int(t)] = int(sum(1 for x in lengths if x > t))
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description="Token length stats for XML samples using Mistral tokenizer")
    ap.add_argument("--data_dir", type=str, default=str(Path("data") / "train_full"))
    ap.add_argument("--prompt_path", type=str, default=None, help="Optional learner prompt template with {rules} and {text}")
    ap.add_argument("--rules_path", type=str, default=None, help="Optional rules text injected into the prompt template")
    ap.add_argument("--base_model", type=str, default="unsloth/mistral-7b-instruct-v0.3-bnb-4bit")
    ap.add_argument("--max_chars_per_sample", type=int, default=20000)
    ap.add_argument("--topk", type=int, default=10, help="Show top-k longest samples")
    ap.add_argument("--thresholds", type=int, nargs="*", default=[512, 1024, 1536, 2048, 4096, 8192])
    args = ap.parse_args()

    learner_template = read_text(args.prompt_path) if args.prompt_path else None
    rules_text = read_text(args.rules_path) if args.rules_path else None

    # Build samples
    samples = build_samples(
        data_dir=args.data_dir,
        max_chars_per_sample=args.max_chars_per_sample,
        learner_template=learner_template,
        rules_text=rules_text,
    )
    if not samples:
        print(f"No samples found under {args.data_dir}")
        return

    # Tokenizer (Mistral)
    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)

    # Tokenize and collect lengths
    lens: List[int] = []
    for s in samples:
        ids = tok(s.prompt, add_special_tokens=False).input_ids
        lens.append(len(ids))

    # Stats
    desc = describe_lengths(lens)
    over = count_over_thresholds(lens, args.thresholds)

    print("Dataset Summary")
    print(f"- Files scanned: {len(set(x.file for x in samples))}")
    print(f"- Samples built: {desc['count']}")
    print("Token Length Stats (prompt-level)")
    print(f"- min: {desc['min'] if 'min' in desc else 'NA'}")
    print(f"- p50: {desc.get('p50', 'NA'):.1f}")
    print(f"- p90: {desc.get('p90', 'NA'):.1f}")
    print(f"- p95: {desc.get('p95', 'NA'):.1f}")
    print(f"- p99: {desc.get('p99', 'NA'):.1f}")
    print(f"- max: {desc['max'] if 'max' in desc else 'NA'}")
    print(f"- mean: {desc.get('mean', 'NA'):.1f} ± {desc.get('std', 0.0):.1f}")

    print("Exceeding Thresholds")
    for t in args.thresholds:
        print(f"- >{t}: {over.get(int(t), 0)}")

    # Top-K longest
    topk = int(args.topk)
    order = np.argsort(np.array(lens)) if lens else []
    if topk > 0 and len(order) > 0:
        print(f"Top-{topk} longest samples")
        for rank, idx in enumerate(order[-topk:][::-1], start=1):
            s = samples[int(idx)]
            print(f"#{rank} | tokens={lens[int(idx)]} | file={s.file} | chunk={s.chunk_idx}")


if __name__ == "__main__":
    main()

