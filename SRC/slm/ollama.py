import io
import json
import requests
import streamlit as st
from typing import Optional, List
import textwrap


OLLAMA_ENDPOINT = "http://localhost:11434"
MODEL_NAME      = "qwen3:8b"
TEMPERATURE     = 0.2

USE_CHUNKING          = True           # si el texto supera MAX_CHARS_PER_CHUNK, se parte
MAX_CHARS_PER_CHUNK   = 15000         # <- cambiá acá (p.ej., 10k)
OVERLAP               = 10           # solapamiento entre chunks

# Importante para evitar cortes por contexto/salida en Ollama:

NUM_CTX               = 16384          # contexto (tokens) del modelo en Ollama
NUM_PREDICT           = 9000           # tokens de salida máximos (subí si necesitás más)

def ollama_generate(
    model: str,
    prompt: str,
    endpoint: str = OLLAMA_ENDPOINT,
    temperature: float = TEMPERATURE,
    options: Optional[dict] = None,
) -> str:
    """
    Llama a /api/generate de Ollama en modo stream y acumula la respuesta.
    Fijamos num_ctx y num_predict para evitar truncamientos por defecto.
    """
    base_opts = {
        "temperature": temperature,
        "num_ctx": NUM_CTX,
        "num_predict": NUM_PREDICT,
    }
    if options:
        base_opts.update(options)

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": base_opts,
    }
    url = f"{endpoint.rstrip('/')}/api/generate"
    resp = requests.post(url, json=payload, stream=True, timeout=600)
    resp.raise_for_status()

    text = []
    for line in resp.iter_lines():
        if not line:
            continue
        chunk = json.loads(line)
        part = chunk.get("response", "")
        if part:
            text.append(part)
    return "".join(text).strip()