#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script principal pour le traitement d'articles de presse et de rapports du GIEC (API uniquement).
"""

import os
import argparse

# ------------------------------------------------------------------
#  Utilitaires d'affichage (inchangés)
# ------------------------------------------------------------------

def log_system_resources(stage: str):
    """Affiche un marqueur de progression simple."""
    print(f"\n=== {stage} ===")

# ------------------------------------------------------------------
# 1.  Nettoyage & reformatage des articles (version locale) ----------
# ------------------------------------------------------------------

def clean_and_reformat_articles():
    """Pipeline de nettoyage des articles via un LLM local."""
    from nettoyer_articles_local import process_cleaning_local

    log_system_resources("Avant nettoyage et reformattage des articles (LLM local)")

    input_csv_path = "Data/presse/articles_brutes/"
    output_csv_path = "Data/presse/articles_csv/"
    output_txt_path = "Data/presse/articles_cleaned/"

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_txt_path), exist_ok=True)

    process_cleaning_local(
        input_dir=input_csv_path,
        output_dir=output_csv_path,
        output_txt_dir=output_txt_path,
        block_size=10,
    )

    log_system_resources("Après nettoyage et reformattage des articles (LLM local)")



# ------------------------------------------------------------------
# 2.  Nettoyage & reformatage des Rapports du GIEC
# ------------------------------------------------------------------

def process_ipcc_reports():
    """
    Pipeline : extraction PDF -> résumé chunk par chunk (LLM local) -> embeddings
    """
    from pdf_processing import process_pdf_to_text
    from summarize_report_local import summarize_pdf_report 
    from embeddings_creation_for_summarized import create_embeddings_from_summarized

    chemin_rapports_pdf = "Data/IPCC/rapports/"
    chemin_output_indexed = "Data/IPCC/rapports_indexed/"

    fichiers_rapports = [f for f in os.listdir(chemin_rapports_pdf) if f.endswith(".pdf")]

    for fichier in fichiers_rapports:
        report_name = os.path.splitext(fichier)[0]
        chemin_pdf = os.path.join(chemin_rapports_pdf, fichier)
        chemin_raw_json = os.path.join(chemin_output_indexed, f"{report_name}_raw.json")

        # 1) Extraction texte
        process_pdf_to_text(chemin_rapport_pdf=chemin_pdf,
                            chemin_output_json=chemin_raw_json,
                            report_name=report_name)

        # 2) Résumé via LLM local
        summarize_pdf_report(input_json_path=chemin_raw_json,
                             output_dir="Data/IPCC/rapports_summarized",
                             max_words=1000)

        # 3) Embeddings
        create_embeddings_from_summarized(report_name=report_name,
                                          summarized_dir="Data/IPCC/rapports_summarized",
                                          output_dir="Data/IPCC/rapports_indexed")




# ------------------------------------------------------------------
# 3.  Détection des extraits liés au GIEC dans la presse
# ------------------------------------------------------------------

def extract_relevant_ipcc_references_multi():
    from filtrer_extraits_local import identifier_extraits_sur_giec_local

    log_system_resources("Avant extraction des références au GIEC")

    chemin_articles_nettoyes = "Data/presse/articles_cleaned/"
    chemin_output_chunked = "Data/presse/articles_chunked/"
    os.makedirs(chemin_output_chunked, exist_ok=True)

    identifier_extraits_sur_giec_local(input_dir=chemin_articles_nettoyes,
                                       output_dir=chemin_output_chunked,
                                       block_size=5)

    log_system_resources("Après extraction des références au GIEC")


# ------------------------------------------------------------------
# 4.  Identifications des informations à Fact-Checker 
# ------------------------------------------------------------------

def generate_fact_to_check_multi():
    """
    Quatrième Partie : extraction des affirmations factuelles (LLM local).
    """
    from fact_to_check_local import question_generation_process_local_multi
    log_system_resources("Avant génération des questions")

    src = "Data/presse/articles_chunked/"
    dst = "Data/resultats/resultats_intermediaires/questions/"
    os.makedirs(dst, exist_ok=True)

    question_generation_process_local_multi(input_dir=src, output_dir=dst)

    log_system_resources("Après génération des questions")
    

# ------------------------------------------------------------------
# 5.  Résumés des sections du rapport du GIEC pour Fact-Checker 
# ------------------------------------------------------------------    

def summarize_source_sections_multi():
    """
    Cinquième Partie : résumé des sources via LLM local.
    """
    from resume_local import process_resume_local_multi
    log_system_resources("Avant résumé des sources")

    qs_dir = "Data/resultats/resultats_intermediaires/questions/"
    out_dir = "Data/resultats/resultats_intermediaires/sources_resumees/"
    emb_dir = "Data/IPCC/rapports_indexed/"
    meta_csv = "Data/Index/metadata_with_GIEC.csv"

    os.makedirs(out_dir, exist_ok=True)

    process_resume_local_multi(
        input_dir=qs_dir,
        chemin_rapport_embeddings=emb_dir,
        metadata_path=meta_csv,
        output_dir=out_dir,
        top_k=20,
    )

    log_system_resources("Après résumé des sources")


# ------------------------------------------------------------------
# 6.  Évaluation finale des articles
# ------------------------------------------------------------------  

def evaluate_generated_responses_multi():
    """
    Sixième Partie : évaluation des articles (LLM local).
    """
    from metrics_local import process_evaluation_local_multi
    log_system_resources("Avant évaluation")

    src = "Data/resultats/resultats_intermediaires/sources_resumees/"
    dst = "Data/resultats/resultats_finaux/resultats_csv/"
    os.makedirs(dst, exist_ok=True)

    process_evaluation_local_multi(input_dir=src, output_dir=dst)

    log_system_resources("Après évaluation")


def structure_evaluation_results():
    """
    Septième Partie : Structuration des résultats d'évaluation en JSON.
    """
    from Structure_JSON import structurer_json
    log_system_resources("Avant structuration des résultats")

    evaluation_dir = 'Data/resultats/resultats_finaux/resultats_csv/'
    article_dir = 'Data/presse/articles_chunked/'
    output_dir = 'Data/resultats/resultats_finaux/json_structured/'
    os.makedirs(output_dir, exist_ok=True)

    structurer_json(evaluation_dir, article_dir, output_dir)
    log_system_resources("Après structuration des résultats")

def html_visualisation_creation():
    """
    Huitième Partie: Création du HTML pour la visualisation des résultats.
    """
    from Creation_code_HTML import generate_html_from_json
    log_system_resources("Avant création du HTML pour visualisation")

    json_dir = "Data/resultats/resultats_finaux/json_structured/"
    output_html = "Data/resultats/Visualisation_results.html"
    articles_data_dir = 'Data/presse/articles_chunked/'
    
    generate_html_from_json(json_dir, output_html, articles_data_dir)
    log_system_resources("Après création du HTML pour visualisation")


def JSON_with_metadata():
    """
    Intégration des metadata aux JSON d'articles, 
    avec chemins définis dans main.py.
    """
    log_system_resources("Début de JSON_with_metadata")

    # Définition des chemins ici
    metadata_path = os.path.join(
        os.getcwd(),
        "Data/Index/metadata_with_GIEC.csv"
    )
    json_input_dir = os.path.join(
        os.getcwd(),
        "Data/resultats/resultats_finaux/json_structured"
    )
    json_output_dir = os.path.join(
        os.getcwd(),
        "Data/resultats/json_structured_with_metadata"
    )

    # On importe la fonction refactorée
    from JSON_with_metadata import process_metadata

    # Appel avec passage explicite des chemins
    nb_matched, nb_total = process_metadata(
        metadata_path=metadata_path,
        json_input_dir=json_input_dir,
        json_output_dir=json_output_dir,
        cutoff=0.7
    )
    print(f"[INFO] {nb_matched} articles appariés sur {nb_total}")

    log_system_resources("Fin de JSON_with_metadata")


def Global_scores():
    """
    Calcul des scores agrégés (metadata + global_scores) via Global_scores.py
    """
    log_system_resources("Début de Global_scores")
    from Global_scores import process_global_scores

    json_input_dir = os.path.join(
        os.getcwd(),
        "Data/resultats/json_structured_with_metadata"
    )
    json_output_dir = os.path.join(
        os.getcwd(),
        "Data/resultats/json_structured_with_metadata_and_global_scores"
    )

    process_global_scores(
        json_input_dir=json_input_dir,
        json_output_dir=json_output_dir
    )
    log_system_resources("Fin de Global_scores")


def Score_journal():
    """
    Visualisation des scores par journal via Score_journal.py
    """
    log_system_resources("Début de Score_journal")
    from Score_journal import process_score_journal

    json_dir = os.path.join(
        os.getcwd(),
        "Data/resultats/json_structured_with_metadata_and_global_scores"
    )

    process_score_journal(json_dir=json_dir)
    log_system_resources("Fin de Score_journal")

def run_full_processing_pipeline():
    """
    Exécute toutes les parties du script, dans l'ordre (exemple).
    Ajustez la séquence selon vos besoins réels.
    """
    log_system_resources("Début de la pipeline complète")
    
    clean_and_reformat_articles()
    process_ipcc_reports()
    extract_relevant_ipcc_references_multi()
    generate_fact_to_check_multi()
    summarize_source_sections_multi()
    evaluate_generated_responses_multi()
    structure_evaluation_results()
    html_visualisation_creation()
    
    # Ensuite, on lance la phase d'enrichissement : JSON_with_metadata
    JSON_with_metadata()
    # On calcule les scores globaux
    Global_scores()
    # On évalue les journaux
    Score_journal()
    
    log_system_resources("Fin de la pipeline complète")

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Script principal pour le traitement d'articles de presse et de rapports du GIEC (API uniquement)."
    )
    parser.add_argument("--task", type=str, default="12",
                        help="Indique quelle(s) partie(s) exécuter. "
                             "Vous pouvez fournir un entier (ex: 1) ou une plage (ex: 1-8).")
    return parser.parse_args()

if __name__ == "__main__":
    log_system_resources("Début du script principal")

    args = parse_arguments()
    task_input = args.task

    # Dictionnaire associant chaque numéro de tâche à sa fonction
    tasks = {
        1: clean_and_reformat_articles,
        2: process_ipcc_reports,
        3: extract_relevant_ipcc_references_multi,
        4: generate_fact_to_check_multi,
        5: summarize_source_sections_multi,
        6: evaluate_generated_responses_multi,
        7: structure_evaluation_results,
        8: html_visualisation_creation,
        9: JSON_with_metadata,
        10: Global_scores,
        11: Score_journal,
        12: run_full_processing_pipeline
    }

    # Vérification si l'argument est une plage (ex: "7-11")
    if '-' in task_input:
        try:
            start_str, end_str = task_input.split('-')
            start, end = int(start_str), int(end_str)
            if start > end:
                raise ValueError("La borne de début doit être inférieure ou égale à la borne de fin.")
            for t in range(start, end + 1):
                if t in tasks:
                    print(f"\n--- Exécution de la tâche {t} ---")
                    tasks[t]()
                else:
                    print(f"Tâche {t} invalide.")
        except ValueError as e:
            print(f"Erreur de parsing de la plage: {e}")
    else:
        # Exécution d'une seule tâche
        try:
            t = int(task_input)
            if t in tasks:
                tasks[t]()
            else:
                print("Tâche invalide. Veuillez choisir une option valide.")
        except ValueError:
            print("Argument invalide, veuillez fournir un entier ou une plage d'entiers (ex: 1 ou 1-8).")

    log_system_resources("Fin du script principal")