"""
pipeline.py – expose UN objet `pipe` réutilisé par tous les modules.
"""
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
import torch
import os

# Configuration du modèle
model_id = "meta-llama/Meta-Llama-3.1-70B-Instruct"

# Configuration pour BitsAndBytes avec déchargement CPU
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    llm_int8_enable_fp32_cpu_offload=True  # Activer le déchargement des poids sur le CPU
)

# Charger le pipeline avec déchargement optimisé
pipe = pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",  # Répartition automatique entre GPU et CPU
        "quantization_config": quantization_config,
        "offload_folder": "/tmp",
    },
)

tokenizer = AutoTokenizer.from_pretrained(model_id)

# Ajouter un pad_token_id distinct si nécessaire
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
