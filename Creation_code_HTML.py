#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Creation_code_HTML.py

Optimisations proposées :
- Lecture des fichiers JSON en parallèle avec ThreadPoolExecutor (max_workers=8 par exemple).
- Optionnel : on peut supprimer la réécriture de chaque JSON dans articles_data_dir si inutile.
"""

import os
import json
import concurrent.futures

def _read_single_json(json_file_path):
    """
    Fonction de lecture d'un fichier JSON unique.
    Renvoie (article_key, article_data).
    """
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            article_data = json.load(f)
        article_key = os.path.splitext(os.path.basename(json_file_path))[0]
        return (article_key, article_data)
    except Exception as e:
        print(f"[ERROR] Echec lecture {json_file_path} : {e}")
        return (None, None)

def generate_html_from_json(json_dir, output_html, articles_data_dir, max_workers=8):
    """
    Génère un fichier HTML permettant de visualiser les articles et leur analyse,
    à partir des fichiers JSON structurés.
    
    Args:
        json_dir (str): Répertoire contenant les fichiers JSON structurés.
        output_html (str): Chemin complet du fichier HTML de sortie.
        articles_data_dir (str): Répertoire pour sauvegarder éventuellement des copies JSON (optionnel).
        max_workers (int): nombre de workers pour la parallélisation de la lecture.
    """
    print(f"[INFO] Lecture des JSON depuis : {json_dir}")
    all_json_files = [f for f in os.listdir(json_dir) if f.endswith(".json")]
    json_paths = [os.path.join(json_dir, fname) for fname in all_json_files]

    # Lecture en parallèle
    articles_data = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {executor.submit(_read_single_json, path): path for path in json_paths}
        
        for future in concurrent.futures.as_completed(future_to_file):
            (article_key, article_obj) = future.result()
            if article_key is not None and article_obj is not None:
                articles_data[article_key] = article_obj

    # (Optionnel) Sauvegarder chaque article dans un fichier JSON séparé
    # --> Si vous n'en avez pas besoin, supprimez simplement ce bloc
    if articles_data_dir is not None:
        os.makedirs(articles_data_dir, exist_ok=True)
        for article_key, article_obj in articles_data.items():
            json_file_path = os.path.join(articles_data_dir, f"{article_key}.json")
            try:
                with open(json_file_path, 'w', encoding='utf-8') as f:
                    json.dump(article_obj, f, indent=4, ensure_ascii=False)
            except Exception as e:
                print(f"[ERROR] Echec écriture JSON {json_file_path} : {e}")

    # Construire le HTML (identique à votre code existant)
    html_content = """<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <title>Environmental News Checker</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background: #f2f2f2;
        }
        .container {
            width: 90%;
            max-width: 960px;
            margin: 20px auto;
            background: #fff;
            padding: 20px;
        }
        select {
            margin: 10px 0;
            padding: 5px;
            width: 100%;
        }
        .phrase {
            border: 1px solid #ccc;
            margin: 10px 0;
            padding: 10px;
            background: #fafafa;
        }
        .metric-block {
            margin: 5px 0;
            padding-left: 20px;
        }
        .metric-title {
            font-weight: bold;
            text-transform: capitalize;
        }
        .score {
            font-weight: bold;
            margin-right: 10px;
        }
        .liste, .categories {
            font-style: italic;
            display: inline-block;
            margin-left: 10px;
        }
        .justification {
            display: block;
            margin-top: 4px;
            color: #666;
        }
        .justification-title {
            font-weight: bold;
        }
    </style>
</head>
<body>
<div class="container">
    <h1>Environmental News Checker</h1>
    <p>Analyse des articles de presse (exactitude, biais, sujets, actions, etc.).</p>
    <select id="article-select" onchange="loadArticle(this.value)">
        <option value="">-- Sélectionnez un article --</option>"""

    # Ajout des options du menu déroulant
    for article_key, article_obj in articles_data.items():
        html_content += f'\n        <option value="{article_key}">{article_obj["article_title"]}</option>'
    
    html_content += """
    </select>
    
    <div id="article-content"></div>
</div>
<script>
    const articlesData = {};
"""
    # Intégration des données JSON dans le script
    for article_key, article_obj in articles_data.items():
        html_content += f'\n    articlesData["{article_key}"] = {json.dumps(article_obj, ensure_ascii=False)};'

    # On recolle votre code existant JS :
    html_content += r"""

    function loadArticle(articleKey) {
        const article = articlesData[articleKey];
        const container = document.getElementById('article-content');
        container.innerHTML = "";
        if (!article) return;
        
        // Titre de l'article
        let titleEl = document.createElement('h2');
        titleEl.innerText = article.article_title;
        container.appendChild(titleEl);

        // Parcours des phrases
        let phrases = article.phrases;
        for (let pid in phrases) {
            let phraseData = phrases[pid];
            let phraseDiv = document.createElement('div');
            phraseDiv.className = 'phrase';

            // Affichage du texte de la phrase
            let textP = document.createElement('p');
            textP.innerText = phraseData.text;
            phraseDiv.appendChild(textP);

            // Analyse
            let analysis = phraseData.analysis;
            for (let metricName in analysis) {
                let metricObj = analysis[metricName];
                
                let metricBlock = document.createElement('div');
                metricBlock.className = 'metric-block';
                
                let metricTitle = document.createElement('div');
                metricTitle.className = 'metric-title';
                metricTitle.innerText = metricName.replace(/_/g, ' ');
                metricBlock.appendChild(metricTitle);

                const LIST_ONLY_METRICS = ["sujets_principaux", "actions_proposees"];
                let score = metricObj.score;

                if (score !== null && score !== undefined) {
                    let scoreSpan = document.createElement('span');
                    scoreSpan.className = 'score';
                    scoreSpan.innerText = "Score : " + score + "/5";
                    metricBlock.appendChild(scoreSpan);
                } else if (!LIST_ONLY_METRICS.includes(metricName)) {
                    let scoreSpan = document.createElement('span');
                    scoreSpan.className = 'score';
                    scoreSpan.innerText = "Score : N/A";
                    metricBlock.appendChild(scoreSpan);
                }
                
                if (Array.isArray(metricObj.liste) && metricObj.liste.length > 0) {
                    let listeSpan = document.createElement('span');
                    listeSpan.className = 'liste';
                    listeSpan.innerText = "[Liste] " + metricObj.liste.join(', ');
                    metricBlock.appendChild(listeSpan);
                }
                
                if (Array.isArray(metricObj.categories) && metricObj.categories.length > 0) {
                    let catSpan = document.createElement('span');
                    catSpan.className = 'categories';
                    catSpan.innerText = "[Catégories] " + metricObj.categories.join(', ');
                    metricBlock.appendChild(catSpan);
                }

                let justificationText = metricObj.justifications;
                let justificationDiv = document.createElement('div');
                justificationDiv.className = 'justification';
                justificationDiv.innerHTML = "<span class='justification-title'>Justification :</span> " 
                                             + (justificationText || "Aucune information disponible.");
                metricBlock.appendChild(justificationDiv);

                phraseDiv.appendChild(metricBlock);
            }
            container.appendChild(phraseDiv);
        }
    }
</script>
</body>
</html>"""
    
    # Écriture du fichier HTML final
    with open(output_html, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"[OK] Fichier HTML créé : {output_html}")

