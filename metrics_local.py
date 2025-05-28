# =============================== metrics_local.py ==============================
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Évaluation locale des extraits d’articles (exactitude, biais, sujets, actions).

Différences par rapport à la version API :
    • plus d’appel à Replicate – on utilise le pipeline local `pipe`;
    • aucun ThreadPoolExecutor – traitement 100 % séquentiel;
    • les fonctions de parsing, les prompts et la structure CSV sont conservés.

Interface publique identique : `process_evaluation_local_multi(...)`.
"""

from __future__ import annotations
import os
import time
import logging
import json
import ast
from typing import List

import pandas as pd
import torch
from tqdm import tqdm

# ----------------------------------------------------------------------------- #
# 0.  Logging & pipeline local                                                 #
# ----------------------------------------------------------------------------- #
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

# ----------------------------------------------------------------------------- #
# 1.  Utilitaires de parsing (inchangés)                                        #
# ----------------------------------------------------------------------------- #
def get_metric(data, metric_name):
    """
    Extrait une métrique de type numérique (score et justification) depuis un dictionnaire.
    Si la valeur associée est un dictionnaire, on recherche "score" et "justification".
    Sinon, on traite la donnée comme une chaîne brute.
    """
    score = None
    justification = None
    val = data.get(metric_name, None)
    if isinstance(val, dict):
        score_val = val.get("score", None)
        if score_val is not None:
            if str(score_val).strip().upper() == "NA":
                score = None
            else:
                try:
                    score = int(score_val)
                except Exception:
                    score = None
        justification = val.get("justification", None)
    else:
        # Traitement de la donnée brute
        val_str = str(val).strip()
        if val_str.upper() == "NA":
            score = None
            justification = val_str
        else:
            try:
                score = int(val_str)
                justification = ""
            except Exception:
                score = None
                justification = val_str
    return score, justification

def get_list_metric(data, metric_name, list_key):
    """
    Extrait une métrique de type liste (avec une justification) depuis un dictionnaire.
    Si la valeur associée est un dictionnaire, on recherche la clé list_key et "justification".
    Sinon, on tente de convertir la donnée en liste.
    """
    liste = None
    justification = None
    val = data.get(metric_name, None)
    if isinstance(val, dict):
        liste = val.get(list_key, None)
        justification = val.get("justification", None)
    else:
        try:
            liste = json.loads(val)
        except Exception:
            liste = val
        justification = ""
    return liste, justification

def parse_metric_response(response_text: str, metric_name: str, is_list: bool = False, list_key: str = None):
    """
    Parse la réponse du LLM (attendue au format JSON ou quasi-JSON) pour extraire la métrique 'metric_name'.
    Si is_list est True, utilise get_list_metric avec la clé list_key, sinon get_metric.
    """
    # Extraction du bloc JSON entre le premier '{' et le dernier '}'
    start_index = response_text.find('{')
    end_index = response_text.rfind('}')
    if start_index != -1 and end_index != -1 and end_index > start_index:
        json_text = response_text[start_index:end_index+1]
    else:
        json_text = response_text

    data = None
    try:
        data = json.loads(json_text)
    except Exception as e:
        try:
            data = ast.literal_eval(json_text)
        except Exception as e2:
            # Si le parsing échoue, on retourne None et la réponse brute en justification
            return (None, response_text)
    
    if not isinstance(data, dict):
        return (None, response_text)
    
    if is_list:
        return get_list_metric(data, metric_name, list_key)
    else:
        return get_metric(data, metric_name)

# ----------------------------------------------------------------------------- #
# 2.  Prompts (inchangés)                                                       #
# ----------------------------------------------------------------------------- #
def creer_prompt_exactitude(current_phrase: str, resume_sections: str) -> str:
    system_prompt = (
        "Tu es un expert en vérification scientifique et en analyse critique des médias. "
        "Ta mission est d'évaluer l'exactitude factuelle d'un extrait d'article de presse concernant le rapport du GIEC et/ou des sujets environnementaux. "
        "En te basant uniquement sur les faits vérifiables figurant dans les sections du rapport, "
        "attribue un score selon les critères suivants :\n"
        "  - NA : Aucun fait vérifiable n'est présent.\n"
        "  - 1 : Les faits vérifiables contredisent clairement le rapport.\n"
        "  - 3 : Les faits sont globalement corrects mais certaines informations manquent de précision.\n"
        "  - 5 : Tous les faits vérifiables concordent parfaitement avec le rapport.\n\n"
        "Justifie précisément ta note en te référant aux passages pertinents."
    )
    user_prompt = (
        f"### Éléments à fact-checker : {current_phrase}\n\n"
        f"### Sections du rapport du GIEC :\n{resume_sections}\n\n"
        "Évalue l'exactitude factuelle en répondant uniquement au format JSON comme suit :\n"
        "{\n"
        '  "exactitude_factuelle": {"score": "NA ou 1-5", "justification": "..."}\n'
        "}\n"
    )
    prompt_template = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        "{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
        "{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
    )
    return prompt_template.format(system_prompt=system_prompt, user_prompt=user_prompt)

def creer_prompt_biais(current_phrase: str, resume_sections: str) -> str:
    system_prompt = (
        "Tu es un expert en vérification scientifique et en analyse critique des médias. "
        "Ta mission est d'évaluer le biais idéologique d'un extrait d'article portant sur le rapport du GIEC et/ou des sujets environnementaux. "
        "En te basant uniquement sur les faits vérifiables du rapport, "
        "attribue un score selon les critères suivants :\n"
        "  - NA : Aucun fait vérifiable n'indique de tendance idéologique.\n"
        "  - 1 : L'extrait présente clairement une position climatosceptique.\n"
        "  - 3 : L'extrait est neutre.\n"
        "  - 5 : L'extrait présente clairement une position pro-environnementale.\n\n"
        "Justifie ta note en citant les éléments du rapport sur lesquels tu te bases."
    )
    user_prompt = (
        f"### Extrait de l'article : {current_phrase}\n\n"
        f"### Sections du rapport du GIEC :\n{resume_sections}\n\n"
        "Évalue le biais idéologique en répondant uniquement au format JSON comme suit :\n"
        "{\n"
        '  "biais_idéologique": {"score": "NA ou 1-5", "justification": "..."}\n'
        "}\n"
    )
    prompt_template = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        "{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
        "{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
    )
    return prompt_template.format(system_prompt=system_prompt, user_prompt=user_prompt)

def creer_prompt_sujets(current_phrase: str, resume_sections: str) -> str:
    system_prompt = (
        "Tu es un expert en vérification scientifique et en analyse critique des médias. "
        "Ta mission est d'identifier les sujets principaux abordés dans un extrait d'article de presse concernant le rapport du GIEC et/ou des sujets environnementaux. "
        "Liste les thèmes majeurs et justifie succinctement leur pertinence. "
        "Si aucun thème n'est identifiable, renvoie une liste vide avec une justification."
    )
    user_prompt = (
        f"### Extrait de l'article : {current_phrase}\n\n"
        "Identifie les sujets principaux en répondant uniquement au format JSON comme suit :\n"
        "{\n"
        '  "sujets_principaux": {"liste": ["Sujet1", "Sujet2", "..."], "justification": "..."}\n'
        "}\n"
    )
    prompt_template = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        "{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
        "{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
    )
    return prompt_template.format(system_prompt=system_prompt, user_prompt=user_prompt)

def creer_prompt_actions(current_phrase: str, resume_sections: str) -> str:
    system_prompt = (
        "Tu es un expert en vérification scientifique et en analyse critique des médias. "
        "Ta mission est d'analyser si un extrait d'article propose des actions concrètes pour répondre aux enjeux environnementaux, "
        "et de les classifier selon les catégories suivantes :\n"
        "  1. Investissement dans la mitigation\n"
        "  2. Investissement dans l'adaptation\n"
        "  3. Changement des comportements individuels de consommation\n"
        "  4. Actions collectives\n"
        "  5. Mobilisation citoyenne\n"
        "  6. Politiques publiques concernant les ménages\n"
        "  7. Politiques publiques concernant les entreprises\n\n"
        "Si aucune action n'est proposée, indique 'NA' et justifie l'absence d'actions."
    )
    user_prompt = (
        f"### Extrait de l'article : {current_phrase}\n\n"
        "Analyse les actions proposées en répondant uniquement au format JSON comme suit :\n"
        "{\n"
        '  "actions_proposees": {"categories": ["Investissement financier dans la mitigation", "Investissement financier dans l\'adaptation", "Changement des comportements individuels de consommation", "Actions collectives", "Mobilisation citoyenne"], "justification": "..."}\n'
        "}\n"
    )
    prompt_template = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        "{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
        "{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
    )
    return prompt_template.format(system_prompt=system_prompt, user_prompt=user_prompt)

# ----------------------------------------------------------------------------- #
# 3.  Appel LLM local                                                           #
# ----------------------------------------------------------------------------- #
def appeler_llm_local(prompt: str,
                      max_tokens: int = 1000,
                      temperature: float = 0.35,
                      max_attempts: int = 3,
                      backoff_base: int = 4) -> str:
    for attempt in range(1, max_attempts + 1):
        try:
            out = pipe(prompt,
                       max_new_tokens=max_tokens,
                       temperature=temperature,
                       return_full_text=False)
            txt = out[0]["generated_text"]
            return txt if isinstance(txt, str) else str(txt)
        except (RuntimeError, ValueError) as err:
            wait = backoff_base ** attempt
            print(f"[WARN] Appel LLM échoué ({attempt}/{max_attempts}) : {err} – retry {wait}s")
            time.sleep(wait)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return "[ERREUR]"

# ----------------------------------------------------------------------------- #
# 4.  Évaluation d’une ligne                                                    #
# ----------------------------------------------------------------------------- #
def evaluer_phrase(row: pd.Series) -> dict:
    out = {
        "id":              row["id"],
        "question":        row["question"],
        "current_phrase":  row["current_phrase"],
        "resume_sections": row["resume_sections"],
    }

    # Exactitude factuelle (si question non vide)
    if str(row["question"]).strip():
        p = creer_prompt_exactitude(row["current_phrase"], row["resume_sections"])
        print(p)
        r = appeler_llm_local(p)
        print(r)
        score, justif = parse_metric_response(r, "exactitude_factuelle")
        out["exactitude_factuelle_score"] = score
        out["exactitude_factuelle_justification"] = justif
    else:
        out["exactitude_factuelle_score"] = None
        out["exactitude_factuelle_justification"] = ""

    # Biais idéologique
    p = creer_prompt_biais(row["current_phrase"], row["resume_sections"])
    print(p)
    r = appeler_llm_local(p)
    print(r)
    score, justif = parse_metric_response(r, "biais_idéologique")
    out["biais_idéologique_score"] = score
    out["biais_idéologique_justification"] = justif

    # Sujets principaux
    p = creer_prompt_sujets(row["current_phrase"], row["resume_sections"])
    print(p)
    r = appeler_llm_local(p)
    print(r)
    lst, justif = parse_metric_response(r, "sujets_principaux", True, "liste")
    out["sujets_principaux_liste"] = lst
    out["sujets_principaux_justification"] = justif

    # Actions proposées
    p = creer_prompt_actions(row["current_phrase"], row["resume_sections"])
    print(p)
    r = appeler_llm_local(p)
    print(r)
    lst, justif = parse_metric_response(r, "actions_proposees", True, "categories")
    out["actions_proposees_categories"] = lst
    out["actions_proposees_justification"] = justif

    return out

# ----------------------------------------------------------------------------- #
# 5.  Traitement d’un fichier                                                   #
# ----------------------------------------------------------------------------- #
def process_single_file(in_path: str, out_path: str):
    try:
        df = pd.read_csv(in_path)
    except Exception as e:
        logging.error(f"Lecture {in_path}: {e}")
        print(f"[ERREUR] lecture {in_path}: {e}")
        return

    results: List[dict] = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Évaluation des extraits"):
        try:
            results.append(evaluer_phrase(row))
        except Exception as e:
            logging.error(f"Évaluation ligne id={row['id']}: {e}")
            print(f"[ERREUR] évaluation ligne id={row['id']}: {e}")

    pd.DataFrame(results).to_csv(out_path, index=False, quotechar='"')
    print(f"[INFO] Résultats sauvegardés → {out_path}")

# ----------------------------------------------------------------------------- #
# 6.  Traitement multi-fichiers séquentiel                                      #
# ----------------------------------------------------------------------------- #
def process_evaluation_local_multi(input_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    files = [f for f in os.listdir(input_dir) if f.endswith("_resume_sections.csv")]

    for file in tqdm(files, desc="Fichiers"):
        out_csv = os.path.join(output_dir, file.replace("_resume_sections.csv", "_evaluations_parsed.csv"))
        if os.path.exists(out_csv) and os.path.getsize(out_csv) > 0:
            print(f"→ Skip {file} (déjà évalué)")
            continue
        process_single_file(os.path.join(input_dir, file), out_csv)