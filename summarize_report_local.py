# =========================== summarize_report_local.py ===========================
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fusion des anciens modules *summarize_utils.py* et *summarize_report.py* pour
travailler exclusivement avec le LLM local (variable `pipe` exposée par
pipeline.py).  
• Appels strictement séquentiels (aucune parallélisation).  
• Back-off exponentiel + vidage GPU en cas d’erreur.  
• Interface publique inchangée : `summarize_pdf_report(...)`.

Ne rien modifier côté `pdf_processing.py` ou `embeddings_creation_for_summarized.py`.
"""

from __future__ import annotations
import os
import json
import time
from typing import List, Dict

import torch
from chunking_utils import chunk_text_in_sentences

# ----------------------------------------------------------------------------- #
# 1. Pipeline local (Meta-Llama-3.1-70B-Instruct, 4-bit)                        #
# ----------------------------------------------------------------------------- #
try:
    from pipeline import pipe           # objet pipeline(text-generation)
except ImportError as e:
    raise ImportError(
        "Impossible d’importer le pipeline local. "
        "Vérifiez que 'pipeline.py' est accessible et définit bien la variable 'pipe'."
    ) from e

# ----------------------------------------------------------------------------- #
# 2. Prompt de résumé                                                           #
# ----------------------------------------------------------------------------- #
SYSTEM_PROMPT = (
    "Tu es un expert du GIEC. Résume fidèlement l'extrait suivant, "
    "en bullet points, sans omettre d'éléments importants. "
    "Réponds uniquement par le résumé, sans phrase d'introduction comme 'Here is a summary of the excerpt in bullet points:' ou autre. "
    "Reste le plus fidèle possible au rapport et intègre le plus d'informations possibles dans ton résumé. "
    "Réponds en Français."
)
PROMPT_TEMPLATE = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
    "{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
    "Extrait:\n{chunk}\n\nFais un résumé en bullet points.<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n"
)


def _build_summary_prompt(chunk_text: str) -> str:
    """Construit le prompt complet (system + user) pour un chunk donné."""
    return PROMPT_TEMPLATE.format(system_prompt=SYSTEM_PROMPT, chunk=chunk_text)


# ----------------------------------------------------------------------------- #
# 3. Appel robuste au modèle local                                              #
# ----------------------------------------------------------------------------- #
def _call_local_llm_for_summary(prompt: str,
                                max_tokens: int = 1024,
                                temperature: float = 0.3,
                                max_attempts: int = 3,
                                backoff_base: int = 4) -> str:
    """
    Génère un résumé pour *prompt* via le pipeline local, avec back-off exponentiel.
    """
    for attempt in range(1, max_attempts + 1):
        try:
            outputs = pipe(prompt,
                           max_new_tokens=max_tokens,
                           temperature=temperature,
                           return_full_text=False)
            # ↩️ `outputs` est une liste de dicts [{'generated_text': '...'}]
            print(outputs[0]["generated_text"].strip())
            return outputs[0]["generated_text"].strip()
        except (RuntimeError, ValueError) as err:
            wait = backoff_base ** attempt
            print(f"[WARN] Résumé échoué ({attempt}/{max_attempts}) : {err}. "
                  f"Nouvelle tentative dans {wait}s…")
            time.sleep(wait)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return "[ERREUR] Impossible de générer le résumé après plusieurs tentatives]"


# ----------------------------------------------------------------------------- #
# 4. Fonction principale : résumé complet d’un rapport                          #
# ----------------------------------------------------------------------------- #
def summarize_pdf_report(input_json_path: str,
                         output_dir: str = "Data/IPCC/rapports_summarized",
                         max_words: int = 1000) -> Dict:
    """
    Lit un fichier *raw.json* (« cleaned_text », issu de pdf_processing),
    découpe le texte en chunks (~*max_words* mots), résume chaque chunk via le
    LLM local puis sauvegarde :
        <report_name>_summarized.json
    avec la structure :
    {
        "report_name": …,
        "summaries": [
            {"chunk_id": 0, "original_text": "...", "summary": "..."},
            …
        ]
    }
    Renvoie le même dictionnaire.
    """
    if not os.path.exists(input_json_path):
        raise FileNotFoundError(f"Fichier JSON introuvable : {input_json_path}")

    with open(input_json_path, encoding="utf-8") as f:
        data = json.load(f)

    report_name = data.get("report_name", "unknown_report")
    raw_text = data.get("cleaned_text", "").strip()
    if not raw_text:
        print(f"[INFO] Champ 'cleaned_text' vide pour {input_json_path}")
        return {}

    os.makedirs(output_dir, exist_ok=True)
    summarized_path = os.path.join(output_dir, f"{report_name}_summarized.json")

    # On ne refait pas un résumé déjà présent
    if os.path.exists(summarized_path):
        with open(summarized_path, encoding="utf-8") as ff:
            print(f"[INFO] Résumé déjà existant : {summarized_path}")
            return json.load(ff)

    # 1) Chunking
    print(f"Chunking '{report_name}'… (≈{max_words} mots/chunk)")
    chunks: List[str] = chunk_text_in_sentences(raw_text, max_words=max_words)

    # 2) Résumés séquentiels
    summaries: List[Dict] = []
    for i, ch in enumerate(chunks):
        prompt = _build_summary_prompt(ch)
        summary_text = _call_local_llm_for_summary(prompt)
        summaries.append({
            "chunk_id": i,
            "original_text": ch[:300] + ("…" if len(ch) > 300 else ""),
            "summary": summary_text
        })

    # 3) Sauvegarde
    data_to_save = {"report_name": report_name, "summaries": summaries}
    with open(summarized_path, "w", encoding="utf-8") as ff:
        json.dump(data_to_save, ff, ensure_ascii=False, indent=4)

    print(f"[INFO] Résumé complet sauvegardé dans {summarized_path}")
    return data_to_save


