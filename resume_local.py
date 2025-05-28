# ============================ resume_local.py ============================
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Résumé des sections d’articles (sources) en **local** :

• plus d’appel à Replicate – on emploie le pipeline local `pipe` ;
• traitement **séquentiel** (pas de ThreadPoolExecutor) pour limiter la VRAM ;
• toutes les I/O, le filtrage d’embeddings et le cache restent inchangés.

Interface publique conservée : `process_resume_local_multi(...)`
"""

from __future__ import annotations
import os
import re
import time
import json
import logging
from functools import lru_cache
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from difflib import SequenceMatcher
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

# ---------------------- configuration & imports LLM --------------------- #
logging.basicConfig(
    filename="error.log",
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%d-%m-%Y %H:%M:%S",
)

try:
    from pipeline import pipe  # pipeline text-generation local
except ImportError as e:
    raise ImportError("Impossible d’importer le pipeline local ‘pipe’.") from e

# ------------------------- outils titres & cache ------------------------ #
_WS_RE = re.compile(r"\s+")
_SUFFIX = "_with_questions.csv"

def normalize_title(raw: str) -> str:
    if raw.endswith(_SUFFIX):
        raw = raw[: -len(_SUFFIX)]
    raw = raw.replace("_", " ")
    raw = _WS_RE.sub(" ", raw)
    return raw.strip().lower()

@lru_cache(maxsize=1)
def get_metadata(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Métadonnées introuvables : {path}")
    df = pd.read_csv(path)
    required = {"Title", "rapport_GIEC"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans '{path}': {', '.join(missing)}")
    df["Title"] = df["Title"].astype(str).apply(normalize_title)
    return df

@lru_cache(maxsize=None)
def get_embeddings_sections(json_path: str) -> Tuple[List[Sequence[float]], List[str]]:
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON introuvable : {json_path}")
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    emb, texts = [], []
    for ch in data.get("chunks", []):
        for bp in ch.get("bullet_points", []):
            emb.append(bp["embedding"])
            texts.append(bp["text"])
    return emb, texts

@lru_cache(maxsize=1)
def get_embed_model() -> SentenceTransformer:
    return SentenceTransformer("Alibaba-NLP/gte-Qwen2-1.5B-instruct", device="cpu")

# ------------------- identification du rapport GIEC -------------------- #
def find_report_by_file_name(
    file_name: str, metadata_path: str, threshold: float = 0.9
) -> str:
    meta = get_metadata(metadata_path)
    article = normalize_title(os.path.basename(file_name))
    best, score = "", 0.0
    for _, row in meta.iterrows():
        rep = str(row["rapport_GIEC"])
        s = SequenceMatcher(None, row["Title"], article).ratio()
        if s >= threshold and s > score:
            best, score = rep, s
    return best.replace(":", "") if best else "AR5 Synthesis Report Climate Change 2014"

# ------------------------ filtrage des sections ------------------------- #
def embed_texts(texts: List[str], model: SentenceTransformer):
    return model.encode(texts, convert_to_tensor=True, device="cpu")

def filtrer_sections_pertinentes(
    df_q: pd.DataFrame,
    model: SentenceTransformer,
    embeddings: List[Sequence[float]],
    bullets: List[str],
    top_k: int = 10,
) -> pd.DataFrame:
    emb_tensor = torch.tensor(embeddings, device="cpu")
    retrieved = []
    for q in df_q["question"]:
        if not str(q).strip():
            retrieved.append("")
            continue
        q_emb = embed_texts([q], model)[0]
        sims = util.cos_sim(q_emb, emb_tensor)[0].cpu()
        idx = np.argsort(-sims)[:top_k]
        retrieved.append("\n---\n".join([bullets[i] for i in idx if bullets[i].strip()]))
    df = df_q.copy()
    df["retrieved_sections"] = retrieved
    return df

# ------------------- prompt & appel LLM local (résumé) ------------------ #
SYSTEM_PROMPT_SUM = (
    "Tu es un expert en résumé scientifique. Ton objectif est de fournir un résumé "
    "détaillé et structuré des faits scientifiques contenus dans les bullet points du "
    "rapport du GIEC, en les reliant directement aux éléments Fact-Checkables."
    "Ces éléments Fact-Checkables seront ensuite évalués au regard des informations du rapport que tu auras résumées."
    "Ne réponds qu'en listant les faits, sans phrase introductive. Constitue une liste "
    "d'informations tirées du rapport qui permettraient de Fact-Checker les éléments. "
    "Tu ne dois jamais directement Fact-Checker ces informations, mais seulement "
    "lister les éléments pertinents qui pourraient les Fact-Checker."
)

USER_TEMPLATE_SUM = (
    "### Éléments Fact-Checkables de l'article ': {question}\n\n"
    "### Bullet points du rapport :\n{retrieved_sections}\n\n"
    "Constitue une liste d'informations tirées des bullets points du rapport qui permettraient de "
    "Fact-Checker les éléments . Si certains éléments des bullets points du rapport ne sont pas en "
    "rapport avec l'élément à FactChecker, ne les résume pas.\nRéponds sous forme "
    "d'une liste, en commençant par les faits les plus directement liés aux "
    "Éléments Fact-Checkables, suivis des éléments contextuels. Chaque point doit "
    "être rédigé en une ou deux phrases, sans ajout de commentaire introductif. Si "
    "aucune information parmi ces bullet points ne permet de Fact-Checker les "
    "éléments, renvoie 'Aucun élément du rapport ne permettrait de Fact-Checker les "
    "éléments'"
)

PROMPT_TEMPLATE_SUM = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{sys}\n<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n{usr}\n<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n"
)

def _build_summary_prompt(question: str, sections: str) -> str:
    user = USER_TEMPLATE_SUM.format(question=question, retrieved_sections=sections)
    return PROMPT_TEMPLATE_SUM.format(sys=SYSTEM_PROMPT_SUM, usr=user)

def summarize_with_local_llm(
    prompt: str,
    temperature: float = 0.35,
    max_tokens: int = 1024,
    max_attempts: int = 3,
    backoff_base: int = 4,
) -> str:
    for attempt in range(1, max_attempts + 1):
        try:
            out = pipe(
                prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                return_full_text=False,
            )
            response = out[0]["generated_text"].strip()
            print(response)
            return response
        except (RuntimeError, ValueError) as err:
            wait = backoff_base ** attempt
            print(f"[WARN] Résumé échoué ({attempt}) : {err} – retry dans {wait}s")
            time.sleep(wait)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return "[ERREUR] Résumé impossible"

# ---------------------- résumé séquentiel d’un fichier ----------------- #
def generer_resume_sequential(df_q: pd.DataFrame) -> pd.DataFrame:
    resumes = []
    for _, row in tqdm(df_q.iterrows(), total=len(df_q), desc="Résumé des sections"):
        q = str(row["question"]).strip()
        if not q:
            resumes.append("")
            continue
        prompt = _build_summary_prompt(q, row["retrieved_sections"])
        print(prompt)
        resumes.append(summarize_with_local_llm(prompt))
    df = df_q.copy()
    df["resume_sections"] = resumes
    return df

# --------------------- traitement multi-fichiers séquentiel ------------ #
def process_resume_local_multi(
    input_dir: str,
    chemin_rapport_embeddings: str,
    metadata_path: str,
    output_dir: str,
    top_k: int = 20,
):
    os.makedirs(output_dir, exist_ok=True)
    files = [f for f in os.listdir(input_dir) if f.endswith(_SUFFIX)]

    for file_name in tqdm(files, desc="Fichiers"):
        out_csv = os.path.join(output_dir, file_name.replace(_SUFFIX, "_resume_sections.csv"))
        if os.path.exists(out_csv) and os.path.getsize(out_csv) > 0:
            print(f"→ Skip {file_name} (déjà traité)")
            continue

        in_path = os.path.join(input_dir, file_name)
        try:
            df_q = pd.read_csv(in_path)
        except Exception as e:
            logging.error(f"Erreur lecture {file_name}: {e}")
            print(f"[ERREUR] lecture {file_name}: {e}")
            continue

        # identification du rapport & chargement des embeddings
        try:
            report = find_report_by_file_name(in_path, metadata_path)
            json_path = os.path.join(
                chemin_rapport_embeddings, f"{report}_summary_chunks.json"
            )
            emb, bullets = get_embeddings_sections(json_path)
            model = get_embed_model()
        except Exception as e:
            logging.error(f"Embeddings/metadata error pour {file_name}: {e}")
            print(f"[ERREUR] embeddings/metadata {file_name}: {e}")
            continue

        # masquage des lignes utiles
        mask = df_q["question"].notna() & df_q["question"].astype(str).str.strip().astype(bool)
        df_q["retrieved_sections"] = ""
        df_q["resume_sections"] = ""

        if mask.any():
            df_use = df_q.loc[mask].copy()
            df_filt = filtrer_sections_pertinentes(df_use, model, emb, bullets, top_k)
            df_q.loc[mask, "retrieved_sections"] = df_filt["retrieved_sections"].values

            df_res = generer_resume_sequential(df_filt)
            df_q.loc[mask, "resume_sections"] = df_res["resume_sections"].values

        df_q.to_csv(out_csv, index=False, quotechar="\"")
        print(f"[INFO] Résumés sauvegardés → {out_csv}")