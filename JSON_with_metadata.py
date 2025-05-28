#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script pour associer les métadonnées des articles à leurs fichiers JSON, 
en utilisant un taux de correspondance (similarité) au lieu d'une égalité stricte.
"""

import os
import json
import re
import pandas as pd

from difflib import get_close_matches

def nettoyer_titre(titre: str) -> str:
    """
    Nettoie un titre en supprimant les underscores et en le mettant en minuscules.
    Supprime également les espaces multiples.
    """
    titre_nettoye = titre.replace("_", " ").strip().lower()
    titre_nettoye = re.sub(r'\s+', ' ', titre_nettoye)
    return titre_nettoye

def charger_metadata(metadata_path: str) -> dict:
    """
    Charge le fichier metadata.csv et retourne un dictionnaire de correspondance :
    clé = titre nettoyé, valeur = dict contenant les informations metadata.
    """
    df = pd.read_csv(metadata_path)
    metadata_dict = {}
    
    for _, row in df.iterrows():
        titre = str(row.get("Title", ""))
        titre_nettoye = nettoyer_titre(titre)
        # On peut construire un dictionnaire avec toutes les colonnes restantes
        metadata_dict[titre_nettoye] = row.to_dict()
    
    return metadata_dict

def trouver_meilleure_correspondance(
    titre_nettoye: str, 
    metadata_keys: list[str], 
    cutoff: float = 0.7
) -> str:
    """
    Utilise difflib.get_close_matches pour trouver la meilleure correspondance
    entre titre_nettoye et la liste de clés 'metadata_keys'.
    Si aucune correspondance ne dépasse le cutoff (0.7 par défaut), renvoie None.
    """
    # get_close_matches renvoie une liste de chaînes les plus proches
    best_matches = get_close_matches(titre_nettoye, metadata_keys, n=1, cutoff=cutoff)
    if best_matches:
        return best_matches[0]  # la meilleure correspondance
    else:
        return None

def ajouter_metadata_aux_json(json_input_dir: str, json_output_dir: str, metadata_dict: dict, cutoff: float = 0.7) -> tuple:
    """
    Parcourt les fichiers JSON d'évaluation dans json_input_dir,
    et pour chacun, ajoute une clé 'metadata' avec les informations metadata associées
    (en cherchant la meilleure correspondance dans metadata_dict).
    Les fichiers mis à jour sont sauvegardés dans json_output_dir.
    
    Retourne un tuple (nb_matched, nb_total) indiquant le nombre d'articles avec metadata
    et le nombre total d'articles traités.
    
    cutoff : float entre 0 et 1. Représente le seuil de similarité.
             1 = correspondance parfaite requise.
    """
    os.makedirs(json_output_dir, exist_ok=True)
    json_files = [f for f in os.listdir(json_input_dir) if f.endswith('.json')]
    
    nb_total = len(json_files)
    nb_matched = 0

    # Liste des clés de metadata_dict
    metadata_keys = list(metadata_dict.keys())

    for file_name in json_files:
        input_path = os.path.join(json_input_dir, file_name)
        with open(input_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except Exception as e:
                print(f"[ERROR] Lecture JSON {file_name} : {e}")
                continue

        # Nettoyer le nom du fichier (sans extension) pour obtenir un titre
        titre_json = os.path.splitext(file_name)[0]
        titre_nettoye = nettoyer_titre(titre_json)

        # Chercher la meilleure correspondance dans metadata_dict
        best_key = trouver_meilleure_correspondance(titre_nettoye, metadata_keys, cutoff)
        if best_key:
            # On trouve le dictionnaire de métadonnées associé
            metadata_info = metadata_dict[best_key]
            data["metadata"] = metadata_info
            nb_matched += 1
            print(f"[OK] Meilleure correspondance pour '{titre_json}' => '{best_key}'")
        else:
            print(f"[WARN] Aucune metadata trouvée pour l'article : {titre_json}")
            data["metadata"] = {}

        # Sauvegarder le fichier JSON mis à jour
        output_path = os.path.join(json_output_dir, file_name)
        try:
            with open(output_path, 'w', encoding='utf-8') as f_out:
                json.dump(data, f_out, indent=4, ensure_ascii=False)
            print(f"[OK] Fichier JSON mis à jour : {output_path}")
        except Exception as e:
            print(f"[ERROR] Sauvegarde JSON {file_name} : {e}")
    
    return nb_matched, nb_total

def process_metadata(
    metadata_path: str,
    json_input_dir: str,
    json_output_dir: str,
    cutoff: float = 0.7
) -> tuple[int,int]:
    """
    Charge les metadata et les injecte dans les JSON d'entrée,
    en écrivant les fichiers enrichis dans json_output_dir.
    Retourne (nb_matched, nb_total).
    """
    metadata_dict = charger_metadata(metadata_path)
    return ajouter_metadata_aux_json(
        json_input_dir=json_input_dir,
        json_output_dir=json_output_dir,
        metadata_dict=metadata_dict,
        cutoff=cutoff
    )

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Associe les metadata aux JSON d'articles"
    )
    parser.add_argument("--metadata", required=True, help="Chemin vers metadata.csv")
    parser.add_argument("--input-dir", required=True, help="Répertoire JSON d'entrée")
    parser.add_argument("--output-dir", required=True, help="Répertoire JSON enrichis")
    parser.add_argument("--cutoff", type=float, default=0.7,
                        help="Seuil de similarité (entre 0 et 1)")
    args = parser.parse_args()

    nb_matched, nb_total = process_metadata(
        metadata_path=args.metadata,
        json_input_dir=args.input_dir,
        json_output_dir=args.output_dir,
        cutoff=args.cutoff
    )
    print(f"\nCompte rendu : {nb_matched} / {nb_total}")