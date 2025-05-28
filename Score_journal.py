#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Ce script lit les JSON "enrichis" avec :
- metadata
- global_scores
- global_sujets
- global_actions
et construit un DataFrame permettant d'agréger les scores et le nombre d'occurrences
de chaque catégorie d'action, ainsi que la concaténation des sujets.
Il propose également une visualisation des moyennes d'exactitude/biais par journal 
et des occurrences de catégories (somme) par journal.
"""

import os
import json
import re
import pandas as pd
from dateutil import parser

# Visualisation
import matplotlib.pyplot as plt
import seaborn as sns

# EXEMPLE : liste fixe des catégories potentielles
LISTE_CATEGORIES = [
    "Investissement dans la mitigation",
    "Investissement dans l'adaptation",
    "Changement des comportements individuels de consommation",
    "Actions collectives",
    "Mobilisation citoyenne",
    "Politiques publiques concernant les ménages",
    "Politiques publiques concernant les entreprises"
]

def nettoyer_titre(titre: str) -> str:
    """ Nettoie un titre/journal en supprimant underscores et en passant en minuscules. """
    titre_nettoye = titre.replace("_", " ").strip().lower()
    titre_nettoye = re.sub(r'\s+', ' ', titre_nettoye)
    return titre_nettoye

def extraire_date(date_str: str) -> pd.Timestamp:
    """ Convertit une date FR en un Timestamp. """
    jours_fr_en = {
        "lundi": "Monday", "mardi": "Tuesday", "mercredi": "Wednesday", "jeudi": "Thursday",
        "vendredi": "Friday", "samedi": "Saturday", "dimanche": "Sunday"
    }
    mois_fr_en = {
        "janvier": "January", "février": "February", "mars": "March", "avril": "April",
        "mai": "May", "juin": "June", "juillet": "July", "août": "August", 
        "septembre": "September", "octobre": "October", "novembre": "November", "décembre": "December"
    }
    date_str_lower = date_str.lower()
    for fr, en in jours_fr_en.items():
        date_str_lower = re.sub(r'^' + fr + r'\s+', en + " ", date_str_lower)
    for fr, en in mois_fr_en.items():
        date_str_lower = re.sub(fr, en, date_str_lower)
    try:
        dt = parser.parse(date_str_lower, fuzzy=True)
        return pd.Timestamp(dt.date())
    except Exception:
        return pd.NaT

def charger_donnees_json_actions(json_dir: str) -> pd.DataFrame:
    """
    Parcourt json_dir, lit chaque JSON, et construit un DataFrame
    avec les colonnes :
        - journal (normalisé)
        - date (Timestamp)
        - exactitude_factuelle
        - biais_idéologique
        - sujets_concat (liste des sujets en chaîne)
        - <catégorie_1>, <catégorie_2>, ... (0/1 occurrences)
    """
    rows = []
    json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
    
    for file_name in json_files:
        file_path = os.path.join(json_dir, file_name)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"[ERROR] Lecture {file_name}: {e}")
            continue

        # Extraction des champs
        metadata = data.get("metadata", {})
        journal_raw = str(metadata.get("Nom_source", "")).strip()
        journal = nettoyer_titre(journal_raw)

        date_str = str(metadata.get("Date", ""))
        date_pub = extraire_date(date_str)

        global_scores = data.get("global_scores", {})
        exactitude = global_scores.get("exactitude_factuelle", None)
        biais = global_scores.get("biais_idéologique", None)

        sujets = data.get("global_sujets", [])
        sujets_concat = "; ".join(sujets) if sujets else ""

        actions = data.get("global_actions", [])
        # Pour chaque catégorie possible, on met 1 si présente, 0 sinon
        cat_dict = {}
        for cat in LISTE_CATEGORIES:
            if cat in actions:
                cat_dict[f"cat_{cat}"] = 1
            else:
                cat_dict[f"cat_{cat}"] = 0
        
        row = {
            "journal": journal,
            "date": date_pub,
            "exactitude_factuelle": exactitude,
            "biais_idéologique": biais,
            "sujets_concat": sujets_concat,
        }
        # On fusionne le dict des catégories
        row.update(cat_dict)
        rows.append(row)
    
    df = pd.DataFrame(rows)
    return df

def visualiser_resultats(df: pd.DataFrame):
    """
    Exemple de visualisation:
      - Moyenne d'exactitude et de biais par journal (barres groupées)
      - Occurrences totales de chaque catégorie par journal (stacked bar ou barres multiples)
    """
    # Convertir les scores en numérique (pour calculer la moyenne)
    df["exactitude_factuelle"] = pd.to_numeric(df["exactitude_factuelle"], errors="coerce")
    df["biais_idéologique"] = pd.to_numeric(df["biais_idéologique"], errors="coerce")

    # Groupby journal : moyenne d'exactitude / biais
    df_scores = df.groupby("journal").agg(
        exactitude_moy=("exactitude_factuelle", "mean"),
        biais_moy=("biais_idéologique", "mean"),
        nb_articles=("journal", "count")
    ).reset_index()

    # On arrondit
    df_scores["exactitude_moy"] = df_scores["exactitude_moy"].round(2)
    df_scores["biais_moy"] = df_scores["biais_moy"].round(2)

    print("\n=== Moyenne d'exactitude / biais par journal ===")
    print(df_scores)

    # Visualisation bar chart
    plt.figure(figsize=(10,5))
    df_scores_melt = df_scores.melt(id_vars=["journal"], 
                                    value_vars=["exactitude_moy", "biais_moy"], 
                                    var_name="metric", value_name="valeur")
    sns.barplot(data=df_scores_melt, x="journal", y="valeur", hue="metric")
    plt.title("Moyenne d'exactitude et de biais par journal")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

    # Groupby journal : somme de chaque catégorie
    cat_cols = [c for c in df.columns if c.startswith("cat_")]
    df_cats = df.groupby("journal")[cat_cols].sum().reset_index()

    print("\n=== Occurrences totales des catégories par journal ===")
    print(df_cats)

    # Visualisation : barplot multiple (une barre par catégorie)
    # On convertit en format long (melt)
    df_cats_melt = df_cats.melt(id_vars=["journal"], 
                                value_vars=cat_cols,
                                var_name="categorie", value_name="occurrences")
    plt.figure(figsize=(10,5))
    sns.barplot(data=df_cats_melt, x="journal", y="occurrences", hue="categorie")
    plt.title("Occurrences totales des catégories par journal")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

    
def process_score_journal(
    json_dir: str
) -> None:
    """
    Agrège les scores par journal, affiche un aperçu et génère les graphiques.
    """
    df = charger_donnees_json_actions(json_dir)
    print("[INFO] Aperçu du DataFrame articles+actions:")
    print(df.head(10))
    visualiser_resultats(df)

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Agrégation des scores et visualisation par journal"
    )
    parser.add_argument("-i", "--input-dir", required=True,
                        help="Répertoire des JSON enrichis")
    args = parser.parse_args()

    process_score_journal(json_dir=args.input_dir)
