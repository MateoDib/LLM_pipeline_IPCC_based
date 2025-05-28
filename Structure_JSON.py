#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
structurer_json.py

Combine les fichiers d'évaluation CSV et les fichiers d'articles CSV pour générer des fichiers JSON structurés.
Chaque fichier JSON contiendra :
  - "article_title" : le nom de l'article (issu du nom de fichier ou d'une colonne),
  - "phrases" : un dictionnaire dont la clé est l'identifiant de la phrase (id)
      et dont la valeur est un objet comprenant :
        * le texte de la phrase,
        * son contexte éventuel,
        * l'analyse des métriques (exactitude, biais, sujets, actions, etc.).

Si une métrique n'est pas attribuée (score None, liste vide, etc.), une justification par défaut
"Aucune information pertinente détectée par le fact checking." est utilisée.

Pour améliorer les performances, chaque fichier est traité en parallèle (thread) 
grâce à ThreadPoolExecutor (max_workers=10) et une barre de progression est affichée via tqdm.
"""

import os
import re
import json
import traceback
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm  # Pour la barre de progression


def traiter_fichier_evaluation(filename: str, evaluation_dir: str, article_dir: str, output_dir: str) -> None:
    """
    Traite un seul fichier CSV d'évaluation (suffixe '_evaluations_parsed.csv') pour générer le JSON correspondant.
    """
    # Construire le chemin complet vers le fichier d'évaluation
    chemin_evaluation = os.path.join(evaluation_dir, filename)
    
    # Remplacer "_evaluations_parsed.csv" par ".csv" pour trouver le fichier article correspondant
    article_filename = filename.replace("_evaluations_parsed.csv", ".csv")
    chemin_article = os.path.join(article_dir, article_filename)

    # Nettoyer le suffixe "_evaluations_parsed" pour nommer le JSON de sortie
    filename_clean = filename.replace("_evaluations_parsed", "")
    output_json_path = os.path.join(output_dir, f"{os.path.splitext(filename_clean)[0]}.json")
    
    # Lecture du CSV d'évaluation
    try:
        df_evaluation = pd.read_csv(chemin_evaluation)
        # Décommentez la ligne suivante pour charger plus vite si vous n'avez pas besoin de toutes les colonnes :
        # df_evaluation = pd.read_csv(chemin_evaluation, usecols=["id","score","..."], dtype=str)
        print(f"[OK] Fichier d'évaluation chargé : {chemin_evaluation}")
    except Exception:
        print(f"[ERROR] Échec chargement de {chemin_evaluation} :\n{traceback.format_exc()}")
        return
    
    # Vérifier l'existence du fichier article correspondant
    if not os.path.exists(chemin_article):
        print(f"[WARN] Fichier article introuvable : {chemin_article}. Ignoré.")
        return
    
    # Lecture du CSV d'article
    try:
        df_article = pd.read_csv(chemin_article)
        print(f"[OK] Fichier article chargé : {chemin_article}")
    except Exception:
        print(f"[ERROR] Échec chargement de {chemin_article} :\n{traceback.format_exc()}")
        return
    
    # Vérifier les colonnes indispensables
    required_columns = ['id', 'current_phrase']
    missing_cols = [col for col in required_columns if col not in df_article.columns]
    if missing_cols:
        print(f"[WARN] Colonnes manquantes dans {chemin_article} : {missing_cols}. Fichier ignoré.")
        return

    # Fusion sur la clé 'id'
    try:
        df_evaluation_drop = df_evaluation.drop(columns=['current_phrase'], errors='ignore')
        df_merged = df_article.merge(df_evaluation_drop, on="id", how="left")
    except Exception:
        print(f"[ERROR] Erreur fusion des données pour {filename} :\n{traceback.format_exc()}")
        return

    # Récupération du titre (on se base sur le nom du fichier article sans extension)
    article_title = os.path.splitext(article_filename)[0]

    # Préparer la structure JSON
    structured_data = {
        "article_title": article_title,
        "phrases": {}
    }

    # Détecter dynamiquement les préfixes de métriques
    metric_prefixes = set()
    for col in df_merged.columns:
        match = re.match(r"^(.*)_(score|justification|liste|categories)$", col)
        if match:
            metric_prefixes.add(match.group(1))

    # Construire le contenu phrase par phrase
    for _, row in df_merged.iterrows():
        analysis_data = {}
        
        # Rassembler chaque metric_name
        for metric_name in metric_prefixes:
            score_val = row.get(f"{metric_name}_score", None)
            justif_val = row.get(f"{metric_name}_justification", None)
            liste_val = row.get(f"{metric_name}_liste", None)
            cat_val   = row.get(f"{metric_name}_categories", None)

            # Convertir le score en int ou None
            if pd.notnull(score_val):
                try:
                    score_val = int(score_val)
                except ValueError:
                    score_val = None
            else:
                score_val = None

            # Justification par défaut
            if pd.isnull(justif_val) or str(justif_val).strip() == "":
                justif_val = "Aucune information pertinente détectée par le fact checking."

            # Gérer "liste_val"
            if pd.notnull(liste_val):
                if isinstance(liste_val, str):
                    import ast
                    try:
                        liste_val = ast.literal_eval(liste_val)
                    except:
                        liste_val = [s.strip() for s in liste_val.split(',') if s.strip()]
                # Sinon, déjà une liste
            else:
                liste_val = []

            # Gérer "cat_val"
            if pd.notnull(cat_val):
                if isinstance(cat_val, str):
                    import ast
                    try:
                        cat_val = ast.literal_eval(cat_val)
                    except:
                        cat_val = [s.strip() for s in cat_val.split(',') if s.strip()]
                # Sinon, déjà une liste
            else:
                cat_val = []

            # Renseigner dans analysis_data
            analysis_data[metric_name] = {
                "score": score_val,
                "justifications": justif_val,
                "liste": liste_val,
                "categories": cat_val
            }

        # Récupérer le texte de la phrase et éventuellement le contexte
        phrase_text = row.get('current_phrase', "")
        context_text = row.get('context', "")

        phrase_data = {
            "text": phrase_text,
            "context": context_text,
            "analysis": analysis_data
        }
        # Clé = l'id de la phrase converti en string
        structured_data["phrases"][str(row['id'])] = phrase_data

    # Écriture du JSON final
    try:
        with open(output_json_path, 'w', encoding='utf-8') as f_out:
            json.dump(structured_data, f_out, indent=4, ensure_ascii=False)
        print(f"[OK] JSON structuré créé : {output_json_path}")
    except Exception:
        print(f"[ERROR] Échec sauvegarde JSON pour {filename} :\n{traceback.format_exc()}")

def structurer_json(evaluation_dir: str, article_dir: str, output_dir: str, max_workers: int = 10) -> None:
    """
    Parcourt les fichiers d'évaluation CSV (suffixe '_evaluations_parsed.csv') dans evaluation_dir,
    et pour chacun, génère le JSON final en parallèle via ThreadPoolExecutor (max_workers=10).
    Affiche une barre de progression grâce à tqdm.
    """
    os.makedirs(output_dir, exist_ok=True)
    all_eval_files = [f for f in os.listdir(evaluation_dir) if f.endswith("_evaluations_parsed.csv")]

    nb_files = len(all_eval_files)
    print(f"[INFO] Nombre de fichiers à traiter : {nb_files}")

    # Barre de progression : "desc" définit le texte affiché à gauche
    with ThreadPoolExecutor(max_workers=max_workers) as executor, \
         tqdm(total=nb_files, desc="Structuring JSONs", unit="file") as pbar:
        
        futures = []
        for filename in all_eval_files:
            future = executor.submit(traiter_fichier_evaluation, filename, evaluation_dir, article_dir, output_dir)
            futures.append(future)
        
        # Chaque fois qu'un fichier est traité, on incrémente la barre de progression
        for future in as_completed(futures):
            try:
                future.result()  # Pour lever les éventuelles exceptions
            except Exception as e:
                print(f"[ERROR] Une erreur est survenue dans un thread : {e}")
            finally:
                # Mettre à jour la progression dans tous les cas (succès ou exception)
                pbar.update(1)

