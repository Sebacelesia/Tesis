import json
import ast
import re
from typing import List, Optional

from llm.model_config import USE_THINK_TRUNC


def strip_think_segment(text: str) -> str:
    """
    Elimina cualquier contenido posterior al marcador '/think' en el texto.
    """
    marker = "/think"
    idx = text.find(marker)
    if idx == -1:
        return text.strip()
    return text[:idx].rstrip()


def maybe_strip_think(text: str) -> str:
    """
    Aplica strip_think_segment solo si USE_THINK_TRUNC es True.
    """
    if text is None:
        text = ""
    if USE_THINK_TRUNC:
        return strip_think_segment(text)
    return text.strip()


def chunk_text_by_chars(text: str, max_chars: int, overlap: int) -> List[str]:
    """
    Divide un texto largo en fragmentos de hasta max_chars caracteres.
    """
    if max_chars <= 0:
        return [text]
    chunks = []
    i = 0
    n = len(text)
    while i < n:
        j = min(i + max_chars, n)
        chunks.append(text[i:j])
        if j == n:
            break
        i = j - overlap if overlap > 0 else j
    return chunks


def normalize_punctuation_in_list(seq: List[str]) -> List[str]:
    """
    Limpia la puntuación y espacios en una lista de strings.
    """
    cleaned: List[str] = []
    for elem in seq:
        s = str(elem)
        s = re.sub(r"[,\.;:]", " ", s)
        s = re.sub(r"\s+", " ", s)
        s = s.strip()
        if s:
            cleaned.append(s)
    return cleaned


def parse_llm_list(text: str) -> List[str]:
    """
    Convierte la salida del LLM en una lista de Python.
    """
    t = text.strip()
    start = t.find("[")
    end = t.rfind("]")
    if start != -1 and end != -1:
        t = t[start : end + 1]

    try:
        return json.loads(t)
    except json.JSONDecodeError:
        return ast.literal_eval(t)
