# embeddings_creation_for_summarized.py

import json
import os
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def split_bullet_points(summary_text):
    """
    Sépare un texte résumé en bullet points.
    On suppose que chaque bullet point est précédé du caractère "•".
    La fonction retourne une liste de chaînes (chaque chaîne correspondant à un bullet point).
    """
    bullet_points = []
    summary_text = summary_text.strip()
    parts = summary_text.split("•")
    for part in parts:
        point = part.strip()
        if point:
            bullet_points.append("• " + point)
    return bullet_points

def create_embeddings_from_summarized(report_name,
                                      summarized_dir="Data/IPCC/rapports_summarized",
                                      output_dir="Data/IPCC/rapports_indexed",
                                      max_workers=4):
    """
    Charge un fichier de type 'Data/IPCC/rapports_summarized/<report_name>_summarized.json'
    puis, pour chaque chunk résumé, sépare le texte en bullet points et calcule
    l'embedding pour chaque bullet point individuellement en parallèle.
    
    La structure de sortie est la suivante :
    {
        "report_name": <report_name>,
        "chunks": [
            {
                "chunk_id": 0,
                "bullet_points": [
                    {"text": "bullet point 1", "embedding": [...]},
                    {"text": "bullet point 2", "embedding": [...]},
                    ...
                ]
            },
            ...
        ]
    }
    
    Si le fichier de sortie existe déjà, il est simplement renvoyé sans recalcul.
    Le résultat est sauvegardé dans un fichier JSON dans output_dir.
    """
    summarized_json_path = os.path.join(summarized_dir, f"{report_name}_summarized.json")
    indexable_json_path = os.path.join(output_dir, f"{report_name}_summary_chunks.json")
    
    # Vérifier si le fichier d'embeddings existe déjà
    if os.path.exists(indexable_json_path):
        print(f"Embeddings déjà existants dans {indexable_json_path}")
        return indexable_json_path

    if not os.path.exists(summarized_json_path):
        raise FileNotFoundError(
            f"Le résumé {summarized_json_path} est introuvable. Assurez-vous d'avoir résumé le rapport avant."
        )
    
    with open(summarized_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    chunk_summaries = data.get("summaries", [])
    if not chunk_summaries:
        print(f"Aucune section 'summaries' trouvée dans {summarized_json_path}")
        return

    output_data = {
        "report_name": report_name,
        "chunks": []
    }
    
    # Charger le modèle (ici on utilise Alibaba-NLP/gte-Qwen2-1.5B-instruct, ajustez si besoin)
    model = SentenceTransformer("Alibaba-NLP/gte-Qwen2-1.5B-instruct", device="cpu")
    
    # Pour chaque chunk, on traite en parallèle la création d'embeddings pour chaque bullet point
    for chunk_info in tqdm(chunk_summaries, desc="Traitement des chunks résumés"):
        chunk_id = chunk_info.get("chunk_id")
        summary_text = chunk_info.get("summary", "")
        bullet_points = split_bullet_points(summary_text)
        
        # Calcul parallèle pour chaque bullet point du chunk
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            bullet_embeddings = list(executor.map(
                lambda bullet: {
                    "text": bullet,
                    "embedding": model.encode(bullet, convert_to_tensor=True, device="cpu").cpu().numpy().tolist()
                },
                bullet_points
            ))
        
        output_data["chunks"].append({
            "chunk_id": chunk_id,
            "bullet_points": bullet_embeddings
        })
    
    os.makedirs(output_dir, exist_ok=True)
    with open(indexable_json_path, 'w', encoding='utf-8') as fw:
        json.dump(output_data, fw, ensure_ascii=False, indent=4)
    
    print(f"Embeddings par bullet points sauvegardés dans {indexable_json_path}")
    return indexable_json_path
