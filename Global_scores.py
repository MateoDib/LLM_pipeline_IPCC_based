#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Ce script calcule à la fois :
1) La moyenne des scores (non nuls) pour 'exactitude_factuelle' et 'biais_idéologique' (ton).
2) La concaténation (unique) de toutes les listes de sujets_principaux.liste.
3) La concaténation (unique) de toutes les actions_proposees.categories.

Les résultats sont ajoutés au fichier JSON sous des clés distinctes :
  - data["global_scores"] = { "exactitude_factuelle": x.xx, "biais_idéologique": y.yy }
  - data["global_sujets"] = ["Sujet1", "Sujet2", ...]
  - data["global_actions"] = ["Mobilisation citoyenne", "Actions collectives", ...]

Le script parcourt tous les fichiers JSON dans un dossier d'entrée, 
et sauvegarde dans un autre dossier.
"""

import os
import json

def calculer_scores_et_listes(data: dict) -> tuple[dict, list, list]:
    """
    Calcule la moyenne des scores (non nuls) pour exactitude_factuelle et biais_idéologique,
    concatène toutes les listes de sujets_principaux et d'actions_proposees.
    
    Args:
        data (dict): Contenu JSON d'un article, qui contient une clé "phrases"
                     où chaque phrase a "analysis" 
                     ex: data["phrases"][<id>]["analysis"]["exactitude_factuelle"]["score"]
                         data["phrases"][<id>]["analysis"]["sujets_principaux"]["liste"]
                         etc.
                     
    Returns:
        (dict, list, list): 
            - dict pour les global_scores (ex. {"exactitude_factuelle": 3.5, "biais_idéologique": 4.0}),
            - liste unique pour global_sujets,
            - liste unique pour global_actions
    """
    # Initialisation des compteurs pour la moyenne
    exactitude_cumul = 0.0
    exactitude_count = 0
    biais_cumul = 0.0
    biais_count = 0

    # Sets pour éviter les doublons
    sujets_set = set()
    actions_set = set()

    phrases = data.get("phrases", {})
    for _, phrase_data in phrases.items():
        analysis = phrase_data.get("analysis", {})

        # =====================
        # 1) Scores exactitude et biais
        # =====================
        # exactitude_factuelle
        exact_data = analysis.get("exactitude_factuelle", {})
        try:
            exact_score = float(exact_data.get("score", 0.0))
        except (TypeError, ValueError):
            exact_score = 0.0
        if exact_score != 0.0:
            exactitude_cumul += exact_score
            exactitude_count += 1

        # biais_idéologique
        biais_data = analysis.get("biais_idéologique", {})
        try:
            biais_score = float(biais_data.get("score", 0.0))
        except (TypeError, ValueError):
            biais_score = 0.0
        if biais_score != 0.0:
            biais_cumul += biais_score
            biais_count += 1

        # =====================
        # 2) Récupération des sujets
        # =====================
        sujets_data = analysis.get("sujets_principaux", {})
        liste_sujets = sujets_data.get("liste", [])
        if isinstance(liste_sujets, list):
            for s in liste_sujets:
                sujets_set.add(str(s).strip())

        # =====================
        # 3) Récupération des actions
        # =====================
        actions_data = analysis.get("actions_proposees", {})
        liste_actions = actions_data.get("categories", [])
        if isinstance(liste_actions, list):
            for a in liste_actions:
                actions_set.add(str(a).strip())

    # Calcul final des moyennes
    global_scores = {}
    # exactitude_factuelle
    if exactitude_count > 0:
        global_scores["exactitude_factuelle"] = round(exactitude_cumul / exactitude_count, 2)
    else:
        global_scores["exactitude_factuelle"] = None

    # biais_idéologique
    if biais_count > 0:
        global_scores["biais_idéologique"] = round(biais_cumul / biais_count, 2)
    else:
        global_scores["biais_idéologique"] = None

    # Conversion des sets en listes triées, ou non triées si on préfère
    global_sujets = sorted(sujets_set) if sujets_set else []
    global_actions = sorted(actions_set) if actions_set else []

    return global_scores, global_sujets, global_actions

def ajouter_scores_globaux(json_input_dir: str, json_output_dir: str):
    """
    Parcourt les fichiers JSON dans json_input_dir, pour chacun :
      - Calcule la moyenne des scores exactitude et biais
      - Concatène les listes de sujets et d'actions
    Ajoute ces informations dans le JSON aux clés :
      data["global_scores"] = {...}
      data["global_sujets"] = [...]
      data["global_actions"] = [...]
    Puis sauvegarde le JSON dans json_output_dir.
    """
    os.makedirs(json_output_dir, exist_ok=True)
    json_files = [f for f in os.listdir(json_input_dir) if f.endswith('.json')]
    nb_total = len(json_files)
    nb_mis_a_jour = 0

    for file_name in json_files:
        input_path = os.path.join(json_input_dir, file_name)
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"[ERROR] Lecture du fichier {file_name}: {e}")
            continue

        # Calculs
        global_scores, global_sujets, global_actions = calculer_scores_et_listes(data)

        # On ajoute ces infos dans le JSON
        data["global_scores"] = global_scores
        data["global_sujets"] = global_sujets
        data["global_actions"] = global_actions

        # Sauvegarde
        output_path = os.path.join(json_output_dir, file_name)
        try:
            with open(output_path, 'w', encoding='utf-8') as f_out:
                json.dump(data, f_out, indent=4, ensure_ascii=False)
            nb_mis_a_jour += 1
            print(f"[OK] Fichier {file_name} mis à jour (scores+listes).")
        except Exception as e:
            print(f"[ERROR] Sauvegarde du fichier {file_name}: {e}")

    print(f"\nCompte rendu: {nb_mis_a_jour} fichiers mis à jour sur {nb_total}.")

    
def process_global_scores(
    json_input_dir: str,
    json_output_dir: str
) -> None:
    """
    Parcourt les JSON d'entrée, calcule les global_scores, global_sujets, global_actions
    et écrit les JSON enrichis dans json_output_dir.
    """
    os.makedirs(json_output_dir, exist_ok=True)
    json_files = [f for f in os.listdir(json_input_dir) if f.endswith('.json')]
    for file_name in json_files:
        input_path = os.path.join(json_input_dir, file_name)
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # calcul des scores et listes
        global_scores, global_sujets, global_actions = calculer_scores_et_listes(data)
        data["global_scores"]  = global_scores
        data["global_sujets"]  = global_sujets
        data["global_actions"] = global_actions
        # sauvegarde
        output_path = os.path.join(json_output_dir, file_name)
        with open(output_path, 'w', encoding='utf-8') as f_out:
            json.dump(data, f_out, indent=4, ensure_ascii=False)

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Injection des scores/actions/sujets globaux dans les JSON"
    )
    parser.add_argument("-i", "--input-dir",  required=True,
                        help="Répertoire des JSON d'entrée")
    parser.add_argument("-o", "--output-dir", required=True,
                        help="Répertoire des JSON enrichis")
    args = parser.parse_args()

    process_global_scores(
        json_input_dir=args.input_dir,
        json_output_dir=args.output_dir
    )
