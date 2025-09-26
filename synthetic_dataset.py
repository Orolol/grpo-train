#!/usr/bin/env python3
"""Generate synthetic XML cases with GPT-5 for anonymization training."""

import argparse
import os
import random
import re
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from openai import OpenAI


# ----------- Prompts & scenarios -----------

FIRST_NAMES: List[str] = [
    "Claire",
    "Louis",
    "Sophie",
    "Antoine",
    "Camille",
    "Julien",
    "Lea",
    "Thomas",
    "Nadia",
    "Victor",
]

LAST_NAMES: List[str] = [
    "Dubois",
    "Moreau",
    "Lefevre",
    "Rousseau",
    "Faure",
    "Marchand",
    "Perrin",
    "Renard",
    "Barbier",
    "Leroy",
]

CITY_DATA = {
    "Lyon": {
        "postal": ["69001", "69002", "69003", "69004", "69005", "69006", "69007", "69008", "69009"],
        "streets": [
            "rue des Tilleuls",
            "cours Lafayette",
            "rue de la Barre",
            "avenue Jean Jaures",
            "rue Garibaldi",
        ],
    },
    "Paris": {
        "postal": ["75011", "75012", "75013", "75014", "75015", "75018"],
        "streets": [
            "rue Oberkampf",
            "avenue de Breteuil",
            "rue Lecourbe",
            "boulevard Voltaire",
            "rue des Martyrs",
        ],
    },
    "Marseille": {
        "postal": ["13001", "13002", "13003", "13005", "13006", "13008"],
        "streets": [
            "rue Paradis",
            "avenue du Prado",
            "rue de la Republique",
            "rue Sainte",
            "boulevard Baille",
        ],
    },
    "Bordeaux": {
        "postal": ["33000", "33100", "33200", "33300"],
        "streets": [
            "cours de l'Intendance",
            "rue Judaique",
            "avenue Thiers",
            "rue Fondaudege",
            "rue des Remparts",
        ],
    },
    "Lille": {
        "postal": ["59000", "59160", "59260"],
        "streets": [
            "rue de Solferino",
            "boulevard de la Liberte",
            "rue Nationale",
            "avenue de Dunkerque",
            "rue Gambetta",
        ],
    },
}

SCENARIOS: List[str] = [
    (
        "Litige de voisinage: {claimant_full} (domicilie {claimant_address}, telephone {claimant_phone}) "
        "declare que {defendant_full} (domicilie {defendant_address}, telephone {defendant_phone}) "
        "organise des soirees bruyantes depuis fevrier {year}."
    ),
    (
        "Contrat de renovation: {claimant_full} affirme avoir verse 40 000 EUR a {defendant_full} pour une cuisine "
        "au {claimant_address} mais les travaux sont restes incomplets au {date}."
    ),
    (
        "Conflit salarial: {claimant_full}, assistant administratif, conteste son licenciement annonce par "
        "{defendant_full} le {date} en l'accusant de divulguer son adresse {claimant_address}."
    ),
    (
        "Litige bail: {claimant_full} refuse de payer les reparations exigees par {defendant_full}, proprietaire, "
        "apres un degat des eaux constate au {claimant_address}."
    ),
    (
        "Service traiteur: {defendant_full}, gerant du local situe {defendant_address}, "
        "n'a pas rembourse l'acompte verse par {claimant_full} pour un mariage familial."
    ),
    (
        "Surniveau de bruit: {claimant_full} signale que les machines industrielles de {defendant_full} "
        "emettent des nuisances pres de son domicile {claimant_address}."
    ),
    (
        "Consommation: {claimant_full} demande le remboursement d'un appareil defaillant « AirFrais 2000 » "
        "achete chez {defendant_full}, magasin situe {defendant_address}."
    ),
    (
        "Association: {claimant_full}, tresorier au {claimant_address}, accuse {defendant_full} "
        "d'avoir conserve le materiel sonorisation loue le {date}."
    ),
]

XML_SKELETON = """\
Tu dois produire un XML strictement conforme a cette ossature (meme ordre des balises) :
<?xml version="1.0" encoding="UTF-8"?><JPS xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="JPS" AN="{year}" CODE="JPJ" MOIS="{month:02d}"><DEBUA N="1"></DEBUA><DECI><TDECI></TDECI><DEBUR DATE="{date}" N="1"></DEBUR><INFO COLLA="OUI" ID="{info_id}"><NINFO>1</NINFO><REFDOC DATE="{date}" ID="{ref_id}" JURI-DATE="{date_iso}" JURI-JURISDICTION="{jurisdiction}" JURI-JURISDICTION-REFID="{jurisdiction_code}" JURI-LOCATION="{city}" JURI-LOCATION-REFID="{city}" JURI-NUMBER="{case_number}" NATURE="{nature}" NOM="{case_name}" RC="{case_number}">{short_ref}</REFDOC><ABS><DESC></DESC></ABS><TXD><AL>Paragraph 1</AL><AL>Paragraph 2</AL><AL>Paragraph 3</AL></TXD></INFO><FINUR></FINUR></DECI><FINUA></FINUA></JPS>
Remplace uniquement Paragraph 1/2/3 par 2 a 4 phrases courtes tres simples qui resumment la situation en francais.
Garde tous les attributs et balises conformes au modele exact.
Ne renvoie aucun commentaire ni texte hors XML.
"""


# ----------- Utility helpers -----------


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def select_scenario(rng: random.Random) -> str:
    return rng.choice(SCENARIOS)


def pick_name(rng: random.Random) -> Tuple[str, str]:
    first = rng.choice(FIRST_NAMES)
    last = rng.choice(LAST_NAMES)
    return first, last


def make_email(first: str, last: str) -> str:
    slug = f"{first}.{last}".lower().replace(" ", "")
    return f"{slug}@exemple.fr"


def random_phone_number(rng: random.Random) -> str:
    blocks = [rng.randint(10, 99) for _ in range(4)]
    return "06 " + " ".join(f"{b:02d}" for b in blocks)


def random_birth_date(rng: random.Random) -> str:
    year = rng.randint(1960, 1998)
    month = rng.randint(1, 12)
    day = rng.randint(1, 28)
    return f"{day:02d}/{month:02d}/{year}"


def random_address(rng: random.Random, city: str) -> str:
    city_meta = CITY_DATA[city]
    street = rng.choice(city_meta["streets"])
    postal = rng.choice(city_meta["postal"])
    number = rng.randint(1, 180)
    return f"{number} {street}, {postal} {city}"


def build_generation_prompt(
    rules: Optional[str],
    scenario: str,
    seed_values: dict,
    example_xml: Optional[str] = None,
) -> str:
    instructions = [
        "You generate synthetic training data for an anonymization model.",
        "Follow exactly the XML skeleton provided (same tags, same order, same attributes).",
        "Do not anonymise anything; keep all personal details exactly as provided.",
        "Keep language in French and very simple.",
        "Adapte tous les textes et attributs pour correspondre precisement au scenario.",
        "Ne cree aucune balise supplementaire, ne change pas les attributs existants.",
        "Utilise les informations personnelles fournies (noms complets, adresses, telephones, emails).",
        "Integre explicitement ces informations dans les paragraphes TXD.",
    ]
    rules_block = rules.strip() if rules else "(Aucune regle fournie)"
    participants_block = (
        "Participants et informations personnelles:\n"
        f"- Demandeur: {seed_values['claimant_full']}, ne le {seed_values['claimant_dob']}, "
        f"domicilie {seed_values['claimant_address']}, telephone {seed_values['claimant_phone']}, "
        f"email {seed_values['claimant_email']}.\n"
        f"- Defendeur: {seed_values['defendant_full']}, ne le {seed_values['defendant_dob']}, "
        f"domicilie {seed_values['defendant_address']}, telephone {seed_values['defendant_phone']}, "
        f"email {seed_values['defendant_email']}.\n"
    )
    extra_requirements = (
        "Exigences supplementaires:\n"
        "- Mentionne au moins une fois chaque adresse et chaque numero de telephone dans le TXD.\n"
        "- Ne remplace aucune information personnelle par des initiales ou des crochets.\n"
        "- Cite au moins un email dans le texte.\n"
    )
    example_block = ""
    if example_xml:
        example_block = (
            "\n\nExemple de document a imiter (style, ton, connecteurs) :\n"
            + example_xml.strip()
        )
    prompt = (
        "\n".join(instructions)
        + "\n\nRegles a respecter:\n"
        + rules_block
        + "\n\nScenario synthetique:\n"
        + scenario
        + "\n\n"
        + participants_block
        + extra_requirements
        + example_block
        + "\nModele de document a remplir:\n"
        + XML_SKELETON.format(**seed_values)
    )
    return prompt


def extract_xml_payload(text: str) -> str:
    cleaned = text.strip()
    match = re.search(r"```(?:xml)?\s*([\s\S]*?)```", cleaned, re.IGNORECASE)
    if match:
        cleaned = match.group(1).strip()
    return cleaned


def validate_xml_payload(xml_text: str) -> bool:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return False
    return root.tag == "JPS"


@dataclass
class GenerationConfig:
    model: str
    max_output_tokens: int
    retries: int
    retry_sleep: float


def call_model(client: OpenAI, cfg: GenerationConfig, prompt: str) -> str:
    for attempt in range(1, cfg.retries + 1):
        try:
            resp = client.responses.create(
                model=cfg.model,
                input=prompt,
                max_output_tokens=cfg.max_output_tokens,
                text={"verbosity": "high"},
                reasoning={"effort": "minimal"},
            )
            text = getattr(resp, "output_text", None)
            if not text:
                raise RuntimeError("empty model response")
            return text
        except Exception as exc:
            if attempt == cfg.retries:
                raise
            backoff = cfg.retry_sleep * (2 ** (attempt - 1))
            print(
                f"[warn] generation failed ({exc}); retrying in {backoff:.1f}s...",
                file=sys.stderr,
            )
            time.sleep(backoff)
    raise RuntimeError("unreachable")


# ----------- CLI -----------


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Generate synthetic XML documents for anonymization training."
    )
    ap.add_argument("--rules_path", type=Path, default=Path("data/rules.md"))
    ap.add_argument(
        "--example_path",
        type=Path,
        default=Path("data/train/jpj190911.xml"),
        help="Document XML de reference pour guider le style.",
    )
    ap.add_argument("--output_dir", type=Path, default=Path("output/synthetic_xml"))
    ap.add_argument("--count", type=int, default=10, help="Number of XML documents to create.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for scenarios.")
    ap.add_argument("--model", type=str, default="gpt-5-mini")
    ap.add_argument("--max_output_tokens", type=int, default=2000)
    ap.add_argument("--retries", type=int, default=4)
    ap.add_argument("--retry_sleep", type=float, default=2.0)
    ap.add_argument("--dry_run", action="store_true", help="Print planned prompts without calling the API.")
    return ap.parse_args()


def ensure_client(dry_run: bool) -> Optional[OpenAI]:
    if dry_run:
        return None
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("[error] OPENAI_API_KEY not set", file=sys.stderr)
        sys.exit(1)
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    return OpenAI(api_key=api_key, base_url=base_url)


def seed_values_from_rng(rng: random.Random, index: int) -> dict:
    # Produce lightweight deterministic identifiers and dates.
    year = rng.randint(2018, 2023)
    month = rng.randint(1, 12)
    day = rng.randint(1, 28)
    date = f"{day:02d}/{month:02d}/{year}"
    date_iso = f"{year:04d}-{month:02d}-{day:02d}"
    case_number = f"{year}{index:03d}"
    case_id = f"{year}{month:02d}{day:02d}{index:03d}"
    city_options = [
        ("Lyon", "CA"),
        ("Paris", "TJ"),
        ("Marseille", "TJ"),
        ("Bordeaux", "CA"),
        ("Lille", "TJ"),
    ]
    city, jurisdiction_code = rng.choice(city_options)
    jurisdiction = "Cour d'appel" if jurisdiction_code == "CA" else "Tribunal judiciaire"
    claimant_first, claimant_last = pick_name(rng)
    defendant_first, defendant_last = pick_name(rng)
    # Ensure parties are distinct when possible.
    if claimant_first == defendant_first and claimant_last == defendant_last:
        defendant_first, defendant_last = pick_name(rng)

    claimant_full = f"{claimant_first} {claimant_last}"
    defendant_full = f"{defendant_first} {defendant_last}"

    claimant_address = random_address(rng, city)
    defendant_address = random_address(rng, city)
    claimant_phone = random_phone_number(rng)
    defendant_phone = random_phone_number(rng)
    claimant_dob = random_birth_date(rng)
    defendant_dob = random_birth_date(rng)
    claimant_email = make_email(claimant_first, claimant_last)
    defendant_email = make_email(defendant_first, defendant_last)

    case_name = f"{claimant_last} {claimant_first} c/ {defendant_last} {defendant_first}"
    short_ref = (
        f"{jurisdiction} {city} {year} no {case_number} - "
        f"{claimant_full} c/ {defendant_full}"
    )
    nature = "Decision synth." if jurisdiction_code == "CA" else "Jugement synth."
    return {
        "year": year,
        "month": month,
        "date": date,
        "date_iso": date_iso,
        "info_id": f"IJPS{case_id}",
        "ref_id": f"RC{case_id}",
        "jurisdiction": jurisdiction,
        "jurisdiction_code": jurisdiction_code,
        "city": city,
        "case_number": case_number,
        "case_name": case_name,
        "nature": nature,
        "short_ref": short_ref,
        "claimant_full": claimant_full,
        "claimant_address": claimant_address,
        "claimant_phone": claimant_phone,
        "claimant_dob": claimant_dob,
        "claimant_email": claimant_email,
        "defendant_full": defendant_full,
        "defendant_address": defendant_address,
        "defendant_phone": defendant_phone,
        "defendant_dob": defendant_dob,
        "defendant_email": defendant_email,
    }


def write_xml(path: Path, xml_text: str) -> None:
    path.write_text(xml_text, encoding="utf-8")


def main() -> None:
    args = parse_args()

    rng = random.Random(args.seed)
    rules_text = read_text(args.rules_path) if args.rules_path.exists() else ""
    example_text = (
        read_text(args.example_path)
        if args.example_path and args.example_path.exists()
        else ""
    )

    client = ensure_client(args.dry_run)
    cfg = GenerationConfig(
        model=args.model,
        max_output_tokens=args.max_output_tokens,
        retries=args.retries,
        retry_sleep=args.retry_sleep,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    generated_paths: List[Path] = []

    for idx in range(1, args.count + 1):
        seed_values = seed_values_from_rng(rng, idx)
        scenario_template = select_scenario(rng)
        scenario = scenario_template.format(**seed_values)
        prompt = build_generation_prompt(rules_text, scenario, seed_values, example_text)

        if args.dry_run:
            print(f"[dry-run] would generate sample {idx:03d} with scenario: {scenario}")
            continue

        try:
            raw = call_model(client, cfg, prompt)
        except Exception as exc:
            print(f"[error] model failed for sample {idx:03d}: {exc}", file=sys.stderr)
            continue

        xml_text = extract_xml_payload(raw)
        if not validate_xml_payload(xml_text):
            print(
                f"[warn] invalid XML received for sample {idx:03d}; skipping.",
                file=sys.stderr,
            )
            continue

        out_path = args.output_dir / f"synthetic_{idx:03d}.xml"
        write_xml(out_path, xml_text)
        generated_paths.append(out_path)
        print(f"[ok] wrote {out_path}")

    if args.dry_run:
        print("[dry-run] no files written.")
        return

    if not generated_paths:
        print("[warn] no synthetic samples generated", file=sys.stderr)
        sys.exit(2)

    print(f"[done] generated {len(generated_paths)} XML files in {args.output_dir}")


if __name__ == "__main__":
    main()
