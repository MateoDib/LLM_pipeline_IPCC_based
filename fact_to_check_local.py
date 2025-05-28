# ====================== fact_to_check_local.py =======================
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extraction locale des affirmations factuelles (questions) à vérifier
dans les rapports du GIEC.

Différences vs version API :
    • Appel au modèle local `pipe` (pipeline.py) – plus de Replicate.
    • Traitement *totalement séquentiel* (pas de ThreadPoolExecutor).

Tout le reste (chemins, format CSV, parsing JSON) est conservé.
"""

import os
import time
import json
import logging
from typing import List

import pandas as pd
import nltk
import torch
from tqdm import tqdm

# ------------------------------------------------------------------ #
# 0.  Logging & pré-requis                                           #
# ------------------------------------------------------------------ #
logging.basicConfig(
    filename="error.log",
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%d-%m-%Y %H:%M:%S",
)

nltk.download("punkt", quiet=True)

try:
    from pipeline import pipe  # pipeline text-generation
except ImportError as e:
    raise ImportError("Impossible d’importer le pipeline local ‘pipe’.") from e

# ------------------------------------------------------------------ #
# 1.  Parsing de la réponse JSON                                     #
# ------------------------------------------------------------------ #
def parse_response(response: str) -> str:
    """
    Extrait la clé \"affirmations\" du JSON renvoyé par le LLM.
    Retourne \"\" si 'affirmations' vaut \"NA\" ou est vide.
    """
    try:
        parsed = json.loads(response)
        affirm = parsed.get("affirmations", "")
        if isinstance(affirm, str):
            return "" if affirm.strip() == "NA" else affirm.strip()
        if isinstance(affirm, list):
            return "; ".join([str(a).strip() for a in affirm if str(a).strip()])
    except Exception:
        response = response.strip()
        return "" if response == "NA" else response
    return ""

# ------------------------------------------------------------------ #
# 2.  Prompt & appel LLM local                                       #
# ------------------------------------------------------------------ #

SYSTEM_PROMPT = (
    "Tu es un expert en extraction de faits environnementaux vérifiables. Ta mission est d'extraire, "
    "réformuler et lister toutes les affirmations relatives à l'environnement, au climat, aux énergies renouvelables "
    "et aux impacts environnementaux présentes dans l'extrait d'un article de presse. "
    "Tu dois ignorer toute information non vérifiable. Réponds uniquement par un objet JSON conforme à la structure demandée."
)

USER_TEMPLATE = (
    "Extrait de l'article : {phrase}\n\n"
    "En te basant sur cet extrait, extrais et reformule toutes les affirmations "
    "factuelles susceptibles d'être vérifiées par les rapports du GIEC.\n\n"
    "La réponse **doit** être un objet JSON au format exact :\n"
    '{{\n  "affirmations": ["affirmation1", "affirmation2", ...]\n}}\n\n'   
    "Si aucune affirmation vérifiable n'est présente, renvoie exactement :\n"
    '{{\n  "affirmations": "NA"\n}}\n\n'                                  
    "Ne rajoute aucune explication ni commentaire."
)

PROMPT_TEMPLATE = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
    "{sys}<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
    "{usr}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
)

def _build_prompt(phrase: str) -> str:
    return PROMPT_TEMPLATE.format(
        sys=SYSTEM_PROMPT,
        usr=USER_TEMPLATE.format(phrase=phrase)  # seul placeholder valide
    )

def generate_question_with_local_llm(
    current_phrase: str,
    temperature: float = 0.4,
    max_tokens: int = 512,
    max_attempts: int = 3,
    backoff_base: int = 4,
) -> str:
    prompt = _build_prompt(current_phrase)
    print(prompt)

    for attempt in range(1, max_attempts + 1):
        try:
            out = pipe(
                prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                return_full_text=False,
            )
            raw = out[0]["generated_text"].strip()
            clean = parse_response(raw)
            print(clean)
            return clean
        except (RuntimeError, ValueError) as err:
            wait = backoff_base ** attempt
            print(f"[WARN] Génération échouée ({attempt}/{max_attempts}) : {err}. "
                  f"Retry dans {wait}s…")
            time.sleep(wait)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return ""

# ------------------------------------------------------------------ #
# 3.  Génération séquentielle des questions                          #
# ------------------------------------------------------------------ #
def generate_questions_sequential(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[dict] = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating questions"):
        question = generate_question_with_local_llm(row["current_phrase"])
        row["question"] = question
        rows.append(row)
    return pd.DataFrame(rows)

# ------------------------------------------------------------------ #
# 4.  Pipeline multi-fichiers (séquentiel)                           #
# ------------------------------------------------------------------ #
def question_generation_process_local_multi(
    input_dir: str,
    output_dir: str,
) -> None:
    """
    Parcourt chaque CSV *_parsed.csv*, génère les questions et écrit
    *_with_questions.csv* dans *output_dir*.
    """
    os.makedirs(output_dir, exist_ok=True)
    csv_files = [f for f in os.listdir(input_dir) if f.endswith("_parsed.csv")]

    for file in tqdm(csv_files, desc="Processing files"):
        in_path = os.path.join(input_dir, file)
        out_path = os.path.join(output_dir, file.replace("_parsed.csv", "_with_questions.csv"))

        try:
            df = pd.read_csv(in_path)
        except Exception as e:
            logging.error(f"Error reading {file}: {e}")
            print(f"[ERROR] Reading {file}: {e}")
            continue

        df_env = df[df["binary_response"] == 1]  # on garde les phrases marquées '1'
        if df_env.empty:
            print(f"[INFO] Aucun extrait pertinent dans {file}")
            continue

        questions_df = generate_questions_sequential(df_env)

        try:
            questions_df.to_csv(out_path, index=False)
            print(f"[INFO] Questions saved → {out_path}")
        except Exception as e:
            logging.error(f"Error saving {out_path}: {e}")
            print(f"[ERROR] Saving {out_path}: {e}")