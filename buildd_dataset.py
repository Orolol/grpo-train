#!/usr/bin/env python3
# scripts/build_teacher_dataset.py

import argparse
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from openai import OpenAI

# ---------- XML helpers ----------

def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")

def list_xml_files(root: Path) -> List[Path]:
    return sorted(p for p in root.rglob("*.xml") if p.is_file())

def extract_relevant_section(xml_text: str) -> str:
    for tag in ("NonStructure", "TXD"):
        m = re.search(rf"<{tag}>[\s\S]*?</{tag}>", xml_text)
        if m:
            return m.group(0)
    return xml_text


# ---------- Teacher client ----------

@dataclass
class TeacherConfig:
    model: str
    max_output_tokens: int
    mode: str  # "responses" or "chat"
    retries: int
    retry_sleep: float

def call_teacher(
    client: OpenAI,
    cfg: TeacherConfig,
    prompt: str,
) -> str:
    for attempt in range(1, cfg.retries + 1):
        try:
            resp = client.responses.create(
                model=cfg.model,
                input=prompt,
                max_output_tokens=cfg.max_output_tokens,
                text={"verbosity": "low"},
                reasoning={"effort": "medium"},
            )
            text = getattr(resp, "output_text", None)
            if not text:
                raise RuntimeError("empty teacher response")
            return text.strip()
        except Exception as exc:
            if attempt == cfg.retries:
                raise
            sleep = cfg.retry_sleep * (2 ** (attempt - 1))
            print(f"[warn] teacher call failed ({exc}); retrying in {sleep:.1f}s...", file=sys.stderr)
            time.sleep(sleep)
    raise RuntimeError("teacher call failed after retries")

# ---------- Validation ----------

def validate_diff(candidate: str) -> Tuple[bool, Optional[dict], str]:
    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError as err:
        return False, None, f"invalid JSON: {err}"
    if not isinstance(obj, dict) or "diff" not in obj:
        return False, None, "missing top-level 'diff'"
    if not isinstance(obj["diff"], list):
        return False, None, "'diff' must be a list"
    for item in obj["diff"]:
        if not isinstance(item, dict):
            return False, None, "each diff entry must be an object"
        for key in ("search", "replace", "motif"):
            if key not in item:
                return False, None, f"missing '{key}'"
        if not isinstance(item["search"], list) or not all(isinstance(s, str) for s in item["search"]):
            return False, None, "'search' must be a list of strings"
        if not isinstance(item["replace"], str) or not isinstance(item["motif"], str):
            return False, None, "'replace'/'motif' must be strings"
    return True, obj, ""

# ---------- Dataset writer ----------

def build_prompt(template: str, rules: str, text: str) -> str:
    return template.format(rules=rules, text=text)

def prepare_samples(
    xml_path: Path,
    learner_template: str,
    rules_text: str,
) -> List[Tuple[str, str]]:
    xml = read_text(xml_path)
    section = extract_relevant_section(xml)
    prompt = build_prompt(learner_template, rules_text, section)
    return [(prompt, f"{xml_path.name}#0")]

def build_output_path(xml_path: Path, input_root: Path, output_root: Path) -> Path:
    try:
        rel = xml_path.relative_to(input_root)
    except ValueError:
        rel = Path(xml_path.name)
    else:
        rel = rel
    rel = rel if isinstance(rel, Path) else Path(rel)
    return (output_root / rel).with_suffix(".json")


def write_diff_file(out_path: Path, diff_obj: dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(diff_obj, fh, ensure_ascii=True)
        fh.write("\n")

# ---------- CLI ----------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate anonymization SFT dataset with GPT-5 teacher.")
    ap.add_argument("--input_dir", type=Path, default=Path("data/train_rl"), help="Directory with XML files.")
    ap.add_argument(
        "--output_dir",
        type=Path,
        default=Path("output/teacher_diffs"),
        help="Directory where per-XML diff JSON files will be written.",
    )
    ap.add_argument("--rules_path", type=Path, default=Path("data/rules.md"))
    ap.add_argument("--learner_prompt", type=Path, default=Path("data/learner_prompt.md"))
    ap.add_argument("--sample_rate", type=float, default=1.0, help="Subsample files (0..1].")
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--teacher_model", type=str, default="gpt-5-mini")
    ap.add_argument("--max_output_tokens", type=int, default=4000)
    ap.add_argument("--mode", choices=("responses", "chat"), default="responses")
    ap.add_argument("--retries", type=int, default=4)
    ap.add_argument("--retry_sleep", type=float, default=3.0)
    ap.add_argument("--dry_run", action="store_true", help="Skip teacher calls; just print plan.")
    return ap.parse_args()

def main() -> None:
    args = parse_args()

    input_root = args.input_dir.resolve()
    output_root = args.output_dir

    random.seed(args.seed)

    learner_template = read_text(args.learner_prompt)
    rules_text = read_text(args.rules_path)

    xml_files = list_xml_files(input_root)
    if not xml_files:
        print("[error] no XML files found", file=sys.stderr)
        sys.exit(1)

    if args.shuffle:
        random.shuffle(xml_files)

    if args.sample_rate < 1.0:
        keep = max(1, int(len(xml_files) * args.sample_rate))
        xml_files = xml_files[:keep]

    client = None
    teacher_cfg = TeacherConfig(
        model=args.teacher_model,
        max_output_tokens=args.max_output_tokens,
        mode=args.mode,
        retries=args.retries,
        retry_sleep=args.retry_sleep,
    )

    if not args.dry_run:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("[error] OPENAI_API_KEY not set", file=sys.stderr)
            sys.exit(1)
        base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        client = OpenAI(api_key=api_key, base_url=base_url)

    written_paths: List[Path] = []
    total_calls = 0

    for xml_path in xml_files:
        samples = prepare_samples(xml_path, learner_template, rules_text)
        for prompt, chunk_id in samples:
            out_path = build_output_path(xml_path, input_root, output_root)
            if args.dry_run:
                print(f"[dry-run] would label {chunk_id} -> {out_path}")
                continue
            try:
                raw = call_teacher(client, teacher_cfg, prompt)
            except Exception as exc:
                print(f"[error] teacher failed on {chunk_id}: {exc}", file=sys.stderr)
                continue
            ok, obj, reason = validate_diff(raw)
            if not ok:
                print(f"[warn] invalid diff for {chunk_id}: {reason}", file=sys.stderr)
                continue
            write_diff_file(out_path, obj)
            written_paths.append(out_path)
            total_calls += 1
            if args.max_samples and total_calls >= args.max_samples:
                break
        if args.max_samples and total_calls >= args.max_samples:
            break

    if args.dry_run:
        print("[dry-run] no files written.")
        return

    if not written_paths:
        print("[warn] no valid teacher outputs produced", file=sys.stderr)
        sys.exit(2)

    print(f"[done] wrote {len(written_paths)} diff files to {output_root}")

if __name__ == "__main__":
    main()
