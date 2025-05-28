# =========================== filtrer_extraits_local.py ==========================
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyse locale (hors-API) des articles de presse pour repérer les extraits
contenant des affirmations fact-checkables liées au climat/GIEC.

⚙️  Différences avec la version API :
    • utilisation du pipeline local `pipe` (importé depuis pipeline.py)  
    • traitement strictement séquentiel (pas de ThreadPoolExecutor)

Toutes les signatures publiques, la logique de parsing et les chemins de sortie
sont inchangés.
"""

import os
import re
import time
import logging
from typing import List, Dict

import pandas as pd
import nltk
import torch
from tqdm import tqdm

from generate_context_windows import generate_context_windows

# ----------------------------------------------------------------------------- #
# 0.  Logging & pré-requis                                                     #
# ----------------------------------------------------------------------------- #
logging.basicConfig(
    filename="error.log",
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%d-%m-%Y %H:%M:%S",
)

nltk.download("punkt", quiet=True)

try:
    from pipeline import pipe  # objet text-generation local
except ImportError as e:
    raise ImportError(
        "Impossible d’importer le pipeline local. Vérifiez que 'pipeline.py' "
        "expose bien la variable 'pipe'."
    ) from e

# ----------------------------------------------------------------------------- #
# 1.  Parsing des réponses LLM                                                 #
# ----------------------------------------------------------------------------- #
import json, logging
from typing import Tuple

_JSON_BLOCK = re.compile(r"{.*}", re.S)                     # 1er bloc {...}
_QUOTES_KEY = re.compile(r'([,{]\s*)(\w+)\s*:')            # clés non-quotées
_SINGLE_TO_DOUBLE = re.compile(r"'([^']*)'")               # '...' → "..."
_TRAIL_COMMA = re.compile(r",\s*([}\]])")                  # virgule avant } ]

def _coerce_jsonish(text: str) -> str:
    """
    Transforme rapidement un JSON « imparfait » (clés sans guillemets,
    simples quotes, virgule de fin) vers quelque chose d’acceptable
    pour json.loads().
    """
    s = _QUOTES_KEY.sub(r'\1"\2":', text)          # clés -> "clé":
    s = _SINGLE_TO_DOUBLE.sub(r'"\1"', s)          # quotes simples -> doubles
    s = _TRAIL_COMMA.sub(r"\1", s)                 # supprime virgule de fin
    return s

def parse_llm_response_flexible(resp: str) -> Tuple[str | None, List[str]]:
    """
    Parse la sortie du LLM censée respecter le format :
      {"reponse": "0|1", "sujets": ["..."]}
    Renvoie (réponse_binaire, liste_sujets).  Les deux valeurs peuvent être
    None / [] si le parsing échoue.

    La fonction :
      1. isole le premier bloc {...}
      2. tente json.loads()
      3. sinon, tente une « coercion » puis json.loads()
      4. sinon, ultimes regex de secours.
    """
    binary, sujets = None, []

    # 1) isoler le bloc JSON
    m = _JSON_BLOCK.search(resp)
    bloc = m.group(0) if m else resp

    # 2) tentative directe
    for txt in (bloc, _coerce_jsonish(bloc)):
        try:
            data = json.loads(txt)
            if isinstance(data, dict):
                r = str(data.get("reponse", "")).strip().replace('"', "")
                if r in {"0", "1"}:
                    binary = r
                raw_sujets = data.get("sujets", [])
                if isinstance(raw_sujets, list):
                    sujets = [str(s).strip() for s in raw_sujets if str(s).strip()]
                elif isinstance(raw_sujets, str):
                    sujets = [s.strip() for s in raw_sujets.split(",") if s.strip()]
                return binary, sujets
        except Exception:
            pass  # on essaie le tour suivant

    # 3) Fallback regex minimal
    try:
        bin_re = re.search(r'"?reponse"?\s*[:=]\s*"?(0|1)"?', resp, re.I)
        if bin_re:
            binary = bin_re.group(1)
        subj_re = re.search(r'"?sujets"?\s*[:=]\s*(\[[^\]]*\])', resp, re.I)
        if subj_re:
            br = subj_re.group(1)
            try:
                sujets = json.loads(_coerce_jsonish(br))
            except Exception:
                sujets = [s.strip().strip('"') for s in br.strip("[]").split(",") if s.strip()]
    except Exception as e:
        logging.error(f"[PARSE_ERROR] {e} | resp: {resp[:120]}…")

    return binary, sujets


def parsed_responses(df: pd.DataFrame) -> pd.DataFrame:
    parsed_data = []
    for _, row in df.iterrows():
        b, subs = parse_llm_response_flexible(row["climate_related"])
        parsed_data.append(
            {
                "id": row["id"],
                "current_phrase": row["current_phrase"],
                "context": row["context"],
                "binary_response": b,
                "subjects": subs,
            }
        )
    return pd.DataFrame(parsed_data)

# ----------------------------------------------------------------------------- #
# 2.  Génération locale                                                        #
# ----------------------------------------------------------------------------- #
SYSTEM_PROMPT = (
    "Tu es un expert en vérification des faits et en analyse textuelle. Ton objectif est de détecter "
    "toutes les informations factuelles ou affirmations vérifiables qui méritent d'être fact-checkées. "
    "Cela inclut non seulement les données chiffrées, statistiques et allégations précises, "
    "mais aussi toute mention d'événements, de politiques, ou d'informations relatives à l'environnement, "
    "aux institutions internationales telles que les COP et le GIEC, ou à toute autre donnée pertinente pour une vérification des faits. "
    "Ne réponds par '1' que si tu identifies des informations intéressantes à vérifier concernant l'environnement, le climat, "
    "les institutions internationales ou d'autres faits pertinents; sinon, répond '0'. "
    "Ensuite, fournis la liste complète de tous les sujets abordés dans le texte. "
    "IMPORTANT : si la réponse est 1, réponds 1 et non pas 1.0 ou 1,0."
    "AUssi, analyse l'extrait fourni et réponds OBLIGATOIREMENT "
    "sous forme d'un objet JSON **strictement** conforme à :\n"
    '{\n  "reponse": "0" | "1",\n  "sujets": ["sujet1", "sujet2", ...]\n}\n'
    "• \"reponse\" = '1' si l'extrait contient des affirmations vérifiables "
    "liées à l'environnement / climat ; sinon '0'.\n"
    "• \"sujets\" = liste des thèmes mentionnés (même hors climat).\n"
    "AUCUN commentaire, tiret, retour à la ligne ou texte hors des accolades."
)

USER_TEMPLATE = (
    "### Extrait : {phrase}\n"
    "### Contexte : {ctx}\n\n"
    "Renvoie UNIQUEMENT le JSON demandé."
)

PROMPT_TEMPLATE = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
    "{sys}<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
    "{usr}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
)

def _build_prompt(phrase:str, ctx:str)->str:
    return PROMPT_TEMPLATE.format(
        sys=SYSTEM_PROMPT,
        usr=USER_TEMPLATE.format(phrase=phrase, ctx=ctx)
    )


def generate_response_with_local_llm(
    current_phrase: str,
    context: str,
    temperature: float = 0.4,
    max_tokens: int = 512,
    max_attempts: int = 3,
    backoff_base: int = 4,
) -> str:
    """
    Appelle le modèle local (pipe) avec back-off exponentiel en cas d’erreur.
    """
    prompt = _build_prompt(current_phrase, context)

    for attempt in range(1, max_attempts + 1):
        try:
            out = pipe(
                prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                return_full_text=False,
            )
            print(out[0]["generated_text"].strip())
            return out[0]["generated_text"].strip()
        except (RuntimeError, ValueError) as err:
            wait = backoff_base ** attempt
            print(
                f"[WARN] Tentative {attempt}/{max_attempts} échouée : {err}. "
                f"Nouvelle tentative dans {wait}s…"
            )
            time.sleep(wait)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return "[ERREUR] génération impossible"

# ----------------------------------------------------------------------------- #
# 3.  Analyse séquentielle des paragraphes                                     #
# ----------------------------------------------------------------------------- #
def analyze_paragraphs_sequential(splitted_text: List[Dict]) -> List[Dict]:
    results = []
    for entry in tqdm(splitted_text, desc="Analyzing paragraphs"):
        analysis = generate_response_with_local_llm(
            entry["current_phrase"], entry["context"]
        )
        results.append(
            {
                "id": entry["id"],
                "current_phrase": entry["current_phrase"],
                "context": entry["context"],
                "climate_related": analysis,
            }
        )
    return results

# ----------------------------------------------------------------------------- #
# 4.  Pipeline multi-fichiers (séquentiel)                                     #
# ----------------------------------------------------------------------------- #
def identifier_extraits_sur_giec_local(
    input_dir: str,
    output_dir: str,
    block_size: int = 5,
) -> None:
    """
    Pour chaque article *.txt* nettoyé dans *input_dir*, crée deux CSV dans
    *output_dir* : <article>.csv (réponses brutes) et <article>_parsed.csv.
    """
    os.makedirs(output_dir, exist_ok=True)
    txt_files = [f for f in os.listdir(input_dir) if f.endswith(".txt")]

    for file_name in tqdm(txt_files, desc="Processing files"):
        input_path = os.path.join(input_dir, file_name)
        out_raw = os.path.join(output_dir, file_name.replace(".txt", ".csv"))
        out_parsed = os.path.join(output_dir, file_name.replace(".txt", "_parsed.csv"))

        try:
            with open(input_path, encoding="utf-8") as f:
                text = f.read()
        except Exception as e:
            msg = f"Error reading file {file_name}: {e}"
            logging.error(msg)
            print(msg)
            continue

        sentences = nltk.sent_tokenize(text)
        splitted = generate_context_windows(sentences)

        analysis_results = analyze_paragraphs_sequential(splitted)

        df_raw = pd.DataFrame(analysis_results)
        df_raw.to_csv(out_raw, index=False)

        df_parsed = parsed_responses(df_raw)
        df_parsed["subjects"] = df_parsed["subjects"].apply(lambda lst: ", ".join(lst))
        df_parsed.to_csv(out_parsed, index=False)

        print(f"[INFO] Results saved for {file_name} → {out_parsed}")