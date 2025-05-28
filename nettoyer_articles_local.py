# ===================== nettoyer_articles_local.py ====================
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module 'nettoyer_articles_local' – Version locale (hors API) pour nettoyer et
reformater des sections d'articles à l'aide d'un LLM chargé via 🤗 Transformers.

• Plus d'appel réseau : on utilise le pipeline importé depuis `pipeline.py`.
• Pas de parallélisation : traitement séquentiel pour limiter la consommation
  mémoire (important avec un modèle volumineux).
• Gestion simple des erreurs et du vidage mémoire GPU.
"""

from __future__ import annotations

import os
import time
from typing import List, Dict

import pandas as pd
import torch
from nltk import sent_tokenize
from tqdm import tqdm

# ------------------------------------------------------------------
# 1.  Récupération du pipeline LLM local ----------------------------
# ------------------------------------------------------------------

try:
    from pipeline import pipe  # type: ignore
except ImportError as e:
    raise ImportError(
        "Impossible d'importer le pipeline local. Assurez‑vous que 'pipeline.py' "
        "est présent dans le PYTHONPATH et qu'il définit la variable 'pipe'."
    ) from e

# ------------------------------------------------------------------
# 2.  Génération du prompt -----------------------------------------
# ------------------------------------------------------------------

SYSTEM_PROMPT = (
    "Tu es un expert en rédaction et mise en forme de contenu. Ton objectif est de nettoyer"
    "et reformater le texte fourni, en améliorant sa clarté sans en altérer aucunement son contenu.\n"
    "En effet, ton travail réside uniquement dans la reconstitution du texte et pas dans quelconque modification du contenu,"
    "sauf lorsque le texte est magnifestement mal transcrit et nécessite des modifcations pour le rendre cohérent.\n"
    "Tu disposes à la fois du contexte précédent et du contexte suivant " 
    "afin que tu puisses comprendre le contexte dans lequel le bloc de phrases a été écris.\n"
    "Ne réponds que par le texte réécrit de la section 'Texte à nettoyer', sans aucune phrase d'introduction ni commentaire."
)

PROMPT_TEMPLATE = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
    "{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
    "### Contexte précédent : {prev}\n\n"
    "### Texte à nettoyer : {section}\n\n"
    "### Contexte suivant : {next}\n\n"
    "Réponds uniquement par le texte nettoyé, sans ajout de préambule ou de conclusion.<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n"
)


def build_prompt(section_text: str, contexte_precedent: str, contexte_suivant: str) -> str:
    return PROMPT_TEMPLATE.format(
        system_prompt=SYSTEM_PROMPT,
        prev=contexte_precedent,
        section=section_text,
        next=contexte_suivant,
    )

# ------------------------------------------------------------------
# 3.  Appel robuste au modèle --------------------------------------
# ------------------------------------------------------------------

def clean_text_with_local_llm(
    section_text: str,
    contexte_precedent: str,
    contexte_suivant: str,
    temperature: float = 0.4,
    max_tokens: int = 512,
    max_attempts: int = 3,
    backoff_base: int = 4,
) -> str:
    """Appelle le pipeline local avec back‑off exponentiel en cas d'erreur."""

    prompt = build_prompt(section_text, contexte_precedent, contexte_suivant)
    

    for attempt in range(1, max_attempts + 1):
        try:
            outputs = pipe(
                prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                return_full_text=False,
            )
            text = outputs[0]["generated_text"].strip() if isinstance(outputs, list) else str(outputs)
            print(text)
            return text
        except (RuntimeError, ValueError) as err:
            wait = backoff_base ** attempt
            print(f"[WARN] Tentative {attempt}/{max_attempts} échouée : {err}. Retry in {wait}s…")
            time.sleep(wait)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return "[ERREUR] Nettoyage impossible après plusieurs tentatives."

# ------------------------------------------------------------------
# 4.  Découpage du texte en blocs -----------------------------------
# ------------------------------------------------------------------

def build_blocks_with_context(text: str, block_size: int) -> List[Dict[str, str]]:
    sentences = sent_tokenize(text)
    blocks: List[Dict[str, str]] = []
    for i in range(0, len(sentences), block_size):
        current = " ".join(sentences[i : i + block_size])
        prev = sentences[i - 1] if i > 0 else ""
        nxt = sentences[i + block_size] if i + block_size < len(sentences) else ""
        blocks.append({"previous": prev, "current": current, "next": nxt})
    return blocks

# ------------------------------------------------------------------
# 5.  Traitement séquentiel des sections ---------------------------
# ------------------------------------------------------------------

def clean_sections_dataframe(df: pd.DataFrame) -> List[Dict[str, str]]:
    results: List[Dict[str, str]] = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Nettoyage des sections"):
        cleaned = clean_text_with_local_llm(
            section_text=row["section_text"],
            contexte_precedent=row["previous"],
            contexte_suivant=row["next"],
        )
        results.append(
            {
                "id": row["id"],
                "section_text": row["section_text"],
                "contexte_precedent": row["previous"],
                "contexte_suivant": row["next"],
                "texte_nettoye": cleaned,
            }
        )
    return results

# ------------------------------------------------------------------
# 6.  Fonction principale de traitement ----------------------------
# ------------------------------------------------------------------

def process_cleaning_local(
    input_dir: str,
    output_dir: str,
    output_txt_dir: str,
    block_size: int = 10,
) -> None:
    """Traite chaque fichier *.txt* du dossier en le nettoyant avec le LLM local."""

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_txt_dir, exist_ok=True)
    txt_files = [f for f in os.listdir(input_dir) if f.endswith(".txt")]

    for txt_file in tqdm(txt_files, desc="Fichiers à traiter"):
        input_path = os.path.join(input_dir, txt_file)
        output_csv_path = os.path.join(output_dir, txt_file.replace(".txt", "_cleaned.csv"))
        output_txt_path = os.path.join(output_txt_dir, txt_file)

        with open(input_path, "r", encoding="utf-8") as f:
            text = f.read().strip()

        blocks = build_blocks_with_context(text, block_size)
        df_blocks = pd.DataFrame(blocks)
        df_blocks["id"] = range(len(blocks))
        df_blocks["section_text"] = df_blocks["current"]

        cleaned_sections = clean_sections_dataframe(df_blocks)
        df_cleaned = pd.DataFrame(cleaned_sections)
        df_cleaned.to_csv(output_csv_path, index=False, quotechar="\"")

        reconstituted = "\n\n".join(df_cleaned.sort_values("id")["texte_nettoye"].tolist())
        with open(output_txt_path, "w", encoding="utf-8") as out_f:
            out_f.write(reconstituted)

        print(f"[INFO] Article nettoyé sauvegardé dans {output_txt_path}")

    print("[INFO] Traitement terminé pour l'ensemble des fichiers.")
