"""
pipeline.py – expose UN objet `pipe` réutilisé par tous les modules.
"""
from transformers import pipeline, AutoTokenizer, BitsAndBytesConfig
import torch

MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"

# --- choisissez l’une des deux configs : -----------------------------
USE_4BIT = False  # True si vous voulez absolument quantifier

kwargs = dict(torch_dtype=torch.bfloat16, device_map="auto")

if USE_4BIT:
    kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True,
        llm_int8_enable_fp32_cpu_offload=True,
    )
    kwargs["offload_folder"] = "/tmp"

pipe = pipeline("text-generation", model=MODEL_ID, **kwargs)

# pad_token pour génération « chat »
tok = AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token