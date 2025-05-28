# pdf_processing.py

import os
import json
import re
import nltk
from pdfminer.high_level import extract_text
nltk.download('punkt')  # Si nécessaire, sinon on peut retirer

def clean_text(text: str) -> str:
    """
    Nettoie un texte : supprime numéros de page, espaces multiples, et sauts de ligne en excès.
    """
    # Supprimer les mentions "Page X"
    text = re.sub(r'\bPage\s+\d+\b', '', text)
    # Supprimer les espaces multiples
    text = re.sub(r'\s+', ' ', text)
    # Supprimer les sauts de ligne en excès
    text = re.sub(r'\n+', '\n', text)
    return text.strip()

def process_pdf_to_text(chemin_rapport_pdf: str,
                        chemin_output_json: str,
                        report_name: str = None) -> None:
    """
    Extrait le texte d'un PDF, le nettoie et le stocke dans un fichier JSON
    sous forme d'un seul champ "cleaned_text".

    Exemple de format JSON :
    {
      "report_name": "xxx",
      "cleaned_text": "Texte complet nettoyé..."
    }

    :param chemin_rapport_pdf: Chemin vers le PDF à extraire.
    :param chemin_output_json: Chemin de sortie du JSON.
    :param report_name: Nom du rapport (facultatif).
    """
    if not os.path.exists(chemin_rapport_pdf):
        raise FileNotFoundError(f"Fichier PDF introuvable : {chemin_rapport_pdf}")

    if report_name is None:
        # Nom du rapport déduit du nom de fichier sans extension
        report_name = os.path.splitext(os.path.basename(chemin_rapport_pdf))[0]

    # Extraction du texte brut avec pdfminer
    raw_text = extract_text(chemin_rapport_pdf)
    if not raw_text.strip():
        print(f"Aucun texte n'a été extrait de {chemin_rapport_pdf}.")
        return

    # Nettoyage
    cleaned = clean_text(raw_text)

    # Sauvegarde JSON
    data = {
        "report_name": report_name,
        "cleaned_text": cleaned
    }
    os.makedirs(os.path.dirname(chemin_output_json), exist_ok=True)
    with open(chemin_output_json, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print(f"Texte nettoyé enregistré dans {chemin_output_json}")
