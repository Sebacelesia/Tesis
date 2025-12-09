import re
from typing import List, Tuple

from llm.prompts import PROMPT1_TEMPLATE
from llm.ollama_client import ollama_generate
from llm.model_config import MODEL_NAME, OLLAMA_ENDPOINT, TEMPERATURE
from utils.text_utils import parse_llm_list, normalize_punctuation_in_list


def run_prompt1_on_first_pages(
    pages_text: List[str],
    n_pages: int = 2,
) -> str:
    """
    Ejecuta el Prompt 1 sobre las primeras páginas del documento.
    """
    fragment_text = "\n".join(pages_text[:n_pages]).strip()
    if not fragment_text:
        return ""
    prompt = PROMPT1_TEMPLATE.replace("{text}", fragment_text)
    out = ollama_generate(
        model=MODEL_NAME,
        prompt=prompt,
        endpoint=OLLAMA_ENDPOINT,
        temperature=TEMPERATURE,
    )
    return out.strip()


def postprocess_patient_data(raw_list: List[str]) -> List[str]:
    """
    Estandariza los datos del paciente extraídos por el modelo.
    """
    if len(raw_list) < 3:
        raise ValueError(
            f"Se esperaban al menos 3 elementos [nombre, documento, direccion], llegó: {raw_list}"
        )

    full_name = str(raw_list[0]).strip()
    doc       = str(raw_list[1]).strip()
    address   = str(raw_list[2]).strip()

    name_parts = [p for p in full_name.split() if p]
    clean_doc = re.sub(r"[^0-9A-Za-z]", "", doc)

    return [*name_parts, clean_doc, address]


def extract_patient_data_chain(pages_text: List[str]) -> Tuple[str, List[str], List[str]]:
    """
    Ejecuta toda la cadena de extracción de datos del paciente (Prompt 1).
    """
    result_prompt_1 = run_prompt1_on_first_pages(pages_text, n_pages=2)

    raw_list = parse_llm_list(result_prompt_1)
    raw_list_clean = normalize_punctuation_in_list(raw_list)
    processed_list = postprocess_patient_data(raw_list_clean)

    return result_prompt_1, raw_list_clean, processed_list
