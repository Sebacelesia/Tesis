import json
from typing import Optional

import requests

from llm.model_config import (
    OLLAMA_ENDPOINT,
    TEMPERATURE,
    NUM_CTX,
    NUM_PREDICT,
    TOP_P,
    TOP_K,
    REPEAT_PENALTY,
    REPEAT_LAST_N,
    PRESENCE_PENALTY,
)


def ollama_generate(
    model: str,
    prompt: str,
    endpoint: str = OLLAMA_ENDPOINT,
    temperature: float = TEMPERATURE,
    options: Optional[dict] = None,
) -> str:
    """
    Envía un prompt a Ollama y devuelve el texto generado por el modelo.
    """

    base_opts = {
        "temperature":      temperature,
        "num_ctx":          NUM_CTX,
        "num_predict":      NUM_PREDICT,
        "top_p":            TOP_P,
        "top_k":            TOP_K,
        "repeat_penalty":   REPEAT_PENALTY,
        "repeat_last_n":    REPEAT_LAST_N,
        "presence_penalty": PRESENCE_PENALTY,
    }
    if options:
        base_opts.update(options)

    payload = {
        "model":   model,
        "prompt":  prompt,
        "stream":  True,
        "options": base_opts,
    }

    url = f"{endpoint.rstrip('/')}/api/generate"
    resp = requests.post(url, json=payload, stream=True, timeout=600)
    resp.raise_for_status()

    text_parts = []
    for line in resp.iter_lines():
        if not line:
            continue
        chunk = json.loads(line)
        part = chunk.get("response", "")
        if part:
            text_parts.append(part)

    return "".join(text_parts).strip()
