# app.py — Pipeline completo en 1 etapa por bloque:
# Etapa única: Prompt 2 + Prompt 5 + Prompt 3 (redondear carga viral al millar más cercano)
import io
import os
import json
import shutil
import tempfile
import textwrap
import re
from typing import Optional, List, Callable, Tuple

import requests
import streamlit as st

# ====== CONFIGURACIONES DE MODELOS ======
# Ajustá los nombres de modelo ("model_name") según cómo los tengas en Ollama.
MODEL_CONFIGS = {
    "Qwen 8B (Ollama)": {
        "model_name":  "qwen3:8b",
        "temperature": 0.0,
        "num_ctx":     16384,
        "num_predict": 9000,
    },
    "GPT 20B (Ollama)": {
        "model_name":  "gpt-oss:20b",   # <-- CAMBIÁ ESTE NOMBRE AL QUE USES EN OLLAMA
        "temperature": 0.0,
        "num_ctx":     25000,           # ejemplo: más contexto para el modelo grande
        "num_predict": 9000,
    },
}

# ====== PARÁMETROS FIJOS (se inicializan con Qwen 8B y luego se pisan según la selección en la UI) ======
OLLAMA_ENDPOINT = "http://localhost:11434"
MODEL_NAME      = MODEL_CONFIGS["Qwen 8B (Ollama)"]["model_name"]
TEMPERATURE     = MODEL_CONFIGS["Qwen 8B (Ollama)"]["temperature"]

USE_CHUNKING          = True            # si el texto supera MAX_CHARS_PER_CHUNK, se parte
MAX_CHARS_PER_CHUNK   = 15000           # caracteres por chunk de texto (del documento)
OVERLAP               = 0               # solapamiento entre chunks (en caracteres)

# Procesar de a N secciones por bloque lógico
SECTIONS_PER_BLOCK    = 3               # secciones por bloque

# Importante para evitar cortes por contexto/salida en Ollama:
NUM_CTX               = MODEL_CONFIGS["Qwen 8B (Ollama)"]["num_ctx"]      # contexto (tokens del modelo) en Ollama
NUM_PREDICT           = MODEL_CONFIGS["Qwen 8B (Ollama)"]["num_predict"]  # tokens de salida máximos

# Flag de debug para ver longitudes en consola
DEBUG_PIPELINE        = True

# Marcador de nueva sección por bloque (para ayudar al modelo)
SECTION_MARKER        = "---Sección Intermedia---"

# Regex para encabezados de secciones en el texto
SECTION_HEADER_REGEX = re.compile(
    r"(?m)^---\s*Secci[oó]n\s+(\d+)\s*---\s*$"
)

# ====== PROMPT 1: extraer datos del encabezado ======
PROMPT1_TEMPLATE = (
    """Eres un especialista en extraer datos de documentos clinicos.

Todos los documentos estan organizados por secciones indicadas al comienzo de cada una de ellas como cabezal. Por ejemplo:

--- Sección 2 ---

La tarea que necesito que hagas es la siguiente:

De la "sección 0 (Encabezado)" del siguiente fragmento preciso que extraigas el nombre completo, el documento y la dirección del paciente.
La respuesta que me des tiene que incluir los tres datos en formato de lista en lenguaje python. Es decir, por ejemplo: ["Juan Perez", "1234567-8", "Avenida Libertad 109"]
La respuesta que me des no debe incluir comentarios de ningun tipo, solamente la lista que te pedí previamente en ese formato.

El fragmento es el siguiente:
{text}"""
)

# ====== PROMPT 2: censurar datos usando la lista ======
PROMPT2_TEMPLATE = """
Eres un especialista en censurar datos de personas en documentos.

El objetivo general es que, a continuación, te daré una lista con datos y debes buscar dichos datos en el texto y siempre que estos aparezcan reemplazarlos por '[CENSURADO]' siguiendo ciertos criterios.

Dentro de la lista encontraras nombres, apellidos, un documento y una dirección de residencia.

Criterios:
1) Siempre que detectes un nombre o apellido de los que está en la lista cambialos por [CENSURADO].
2) Siempre que detectes el documento de la lista en el documento cambialo por [CENSURADO]. Este documento no tiene porque aparecer en el documento tal cual como está en la lista. Por dar algunos ejemplos, si en la lista está "12345678" y en el documento encuentras "1234567-8", "1.234.567-8" o "1234567 8" esto debes cambiar por [CENSURADO] también.
3) Siempre que detectes la dirección de la lista en el documento cambialo por [CENSURADO]. Esta dirección de la lista no tiene porque ser idéntica a la que encuentres en el documento. Por dar algunos ejemplos, si en la lista está "Avenida Libertad 123" y en el documento aparece "Av. Libertad 123" esto debes cambiar por [CENSURADO] también. Si en la lista está "Mateo Cortéz 2395" y en el documento está "M. Cortéz 2395", "Cortéz 2395" o incluso "Cortéz numero 2395", esto debes cambiar por [CENSURADO] también.
4) Por favor manten el formato (no utilices negritas ni aumentes el tamaño de la letra).
5) MUY IMPORTANTE: si en el texto NO encuentras ninguno de los datos de la lista, devuelve el texto ORIGINAL sin ningún cambio y sin añadir frases como "no hay nada que censurar" ni otros comentarios.
6) Devuelve ÚNICAMENTE el documento censurado (o el original si no hay cambios), sin explicaciones ni notas adicionales.

A continuación te muestro la lista:

{lista}

A continuación te comparto el documento:
{text}
"""

# ====== PROMPT 3: marcar carga viral y redondear al millar más cercano ======
PROMPT3_TEMPLATE = """
Eres un especialista en análisis de textos clínicos en español.

Tu tarea es identificar TODAS las menciones de carga viral en el texto y reemplazar el número por un intervalo como se muestra a continuacion.

Instrucciones obligatorias:
1) Considera como mención de carga viral expresiones en las que aparezcan términos como
   "carga viral", "Carga viral", "CV", "cv" cerca de un número, por ejemplo:
   - "carga viral: 120000"
   - "carga viral 120000"
   - "CV: 120000"
   - "cv: 120000"
   - "CV 120000"
   u otras variantes similares.

2) Para cada número que represente una carga viral sustituye este por el redondeo a las decenas de millar mas cercano.

   Ejemplos de como cambiar el numero:

   - 123456 → 120000
   - 1201 → 0
   - 14999 → 10000
   - 10000 → 10000
   - -234567 → -230000

3) NO modifiques ningún otro número que no esté claramente asociado a carga viral.
4) Si en el texto NO hay ninguna mención de carga viral, devuelve el texto ORIGINAL sin ningún cambio.
5) Devuelve ÚNICAMENTE el documento censurado (o el original si no hay cambios), sin explicaciones ni notas adicionales.

Texto a procesar:
{text}
"""

# ====== PROMPT 5: tratar sección "Responsables del registro" ======
PROMPT5_TEMPLATE = """
Eres un asistente especializado en anonimizar historias clínicas en español.

Tu ÚNICA tarea en este paso es tratar apellidos.

Instrucciones obligatorias:
1) Si en el texto aparece un encabezado "Responsables del registro:" (o una variante muy similar, sin importar mayúsculas/minúsculas), debes conservar el encabezado tal cual.
    Luego debes anonimizar los nombres y apellidos que se encuentren a continuacion. 
    
    Por ejemplo:

    Responsables del registro:
    AE R. VILLANUEVA
    LIC DOS SANTOS

    Pasaria a ser:

    Responsables del registro:
    [CENSURADO]
    [CENSURADO]

2) Si identificas cualquier nombre o apellido en el texto, cambialo por [CENSURADO]. Ejemplos: Juan Perez → [CENSURADO], Rodriguez → [CENSURADO], Dr. Benitez → [CENSURADO].
3) Devuelve ÚNICAMENTE el documento censurado (o el original si no hay cambios), sin explicaciones ni notas adicionales.

Texto a procesar:
{text}
"""

# =======================================
# ---- PDF → lista de páginas (PyMuPDF) ----
def pdf_bytes_to_pages(pdf_bytes: bytes) -> List[str]:
    import fitz  # PyMuPDF
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []
    for page in doc:
        pages.append(page.get_text().strip())
    doc.close()
    return pages

# ---- Llamada a Ollama (streaming) ----
def ollama_generate(
    model: str,
    prompt: str,
    endpoint: str = OLLAMA_ENDPOINT,
    temperature: float = TEMPERATURE,
    options: Optional[dict] = None,
    # parámetros extra a nivel de request (no van dentro de options)
    think: Optional[bool | str] = None,   # False / True / "low" / "medium" / "high"
    raw: bool = False,
    logprobs: bool = False,
    top_logprobs: int = 0,
) -> str:
    """
    Wrapper para /api/generate de Ollama.

    - Usa un set de hiperparámetros por defecto (base_opts).
    - Permite sobreescribir cualquier cosa con 'options'.
    - Permite activar thinking mode con 'think'.
    - Permite pedir logprobs.
    """

    # ==== OPCIONES POR DEFECTO (se pueden pisar con 'options') ====
    base_opts = {
        # Creatividad / aleatoriedad
        "temperature":       temperature,

        # Longitud de contexto y de salida
        "num_ctx":           NUM_CTX,      # tokens de contexto máximos
        "num_predict":       NUM_PREDICT,  # tokens de salida máximos
        "num_keep":          0,            # cuántos tokens del prompt sí o sí se preservan al truncar

        # Muestreo (sampling)
        "top_k":             40,           # nº máximo de tokens candidatos
        "top_p":             0.9,          # nucleus sampling (prob acumulada)
        "min_p":             0.0,          # prob mínima; tokens debajo de esto se descartan
        "tfs_z":             1.0,          # Tail Free Sampling (1.0 = desactivado)
        "typical_p":         1.0,          # Typical sampling (1.0 = desactivado)

        # Penalizaciones por repetición / presencia / frecuencia
        "repeat_last_n":     256,          # ventana de tokens recientes a considerar
        "repeat_penalty":    1.1,          # >1 penaliza repetir (1.0 = sin penalización)
        "presence_penalty":  0.0,          # penaliza aparecer al menos una vez
        "frequency_penalty": 0.0,          # penaliza según cuántas veces aparece
        "penalize_newline":  False,        # si también penalizar saltos de línea

        # Mirostat (otro esquema de muestreo)
        "mirostat":          0,            # 0 = apagado, 1 o 2 = mirostat activado
        "mirostat_tau":      5.0,          # entropía objetivo
        "mirostat_eta":      0.1,          # tasa de adaptación

        # Reproducibilidad
        "seed":              0,            # 0 = aleatorio; >0 = mismo resultado con mismo prompt

        # Dónde cortar la generación (STOP tokens)
        "stop":              [],
    }

    # Si nos pasan un dict 'options', pisa los defaults anteriores
    if options:
        base_opts.update(options)

    # ==== Construir payload principal ====
    payload: dict = {
        "model":   model,
        "prompt":  prompt,
        "stream":  True,
        "options": base_opts,
        "raw":     raw,
    }

    # Thinking mode (si el modelo lo soporta)
    if think is not None:
        payload["think"] = think

    # Probabilidades de tokens (logprobs)
    if logprobs:
        payload["logprobs"] = True
        if top_logprobs > 0:
            payload["top_logprobs"] = int(top_logprobs)

    url = f"{endpoint.rstrip('/')}/api/generate"
    resp = requests.post(url, json=payload, stream=True, timeout=600)
    resp.raise_for_status()

    text_parts: List[str] = []
    for line in resp.iter_lines():
        if not line:
            continue
        chunk = json.loads(line)

        # El texto "normal" viene en la clave 'response'
        part = chunk.get("response", "")
        if part:
            text_parts.append(part)

        # Si quisieras capturar también el "thinking":
        # thinking_part = chunk.get("thinking")

    return "".join(text_parts).strip()

# ---- Helper: cortar en el primer /think (para todos los prompts salvo el 1) ----
def strip_think_segment(text: str) -> str:
    """
    Si en el texto aparece '/think', devuelve todo lo que está antes del primer '/think'.
    Si no aparece, devuelve el texto tal cual (solo con strip).
    """
    marker = "/think"
    idx = text.find(marker)
    if idx == -1:
        return text.strip()
    return text[:idx].rstrip()

# ---- Helpers para el marcador de sección ----
def add_section_marker(text: str) -> str:
    """Agrega el marcador SECTION_MARKER al inicio del bloque, sin pisar nada."""
    if not text.strip():
        return text
    return f"{SECTION_MARKER}\n\n{text}"

def remove_section_marker(text: str) -> str:
    """
    Elimina TODAS las ocurrencias exactas de SECTION_MARKER del texto.
    SOLO borra ese fragmento, nada más.
    """
    cleaned = text.replace(SECTION_MARKER, "")
    # Opcional: limpiar espacios duplicados que puedan quedar
    return cleaned.strip()

# ---- Chunking por caracteres (solo el TEXTO, no el template) ----
def chunk_text_by_chars(text: str, max_chars: int, overlap: int) -> List[str]:
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

# ---- Texto → PDF (PyMuPDF) ----
def text_to_pdf_bytes(
    text: str,
    paper: str = "A4",
    fontname: str = "Courier",
    fontsize: int = 10,
    margin: int = 36,
    line_spacing: float = 1.4,
) -> bytes:
    import fitz
    doc = fitz.open()
    if paper.upper() == "A4":
        width, height = 595, 842
    else:
        width, height = 612, 792

    usable_w = width - 2 * margin
    usable_h = height - 2 * margin

    char_w = fontsize * 0.6
    max_chars_per_line = max(20, int(usable_w / char_w))

    line_h = int(fontsize * line_spacing)
    max_lines_per_page = max(10, int(usable_h / line_h))

    all_lines: List[str] = []
    for para in text.splitlines():
        if not para.strip():
            all_lines.append("")
            continue
        wrapped = textwrap.wrap(para, width=max_chars_per_line, break_long_words=False)
        if not wrapped:
            all_lines.append("")
        else:
            all_lines.extend(wrapped)

    page = None
    x = margin
    y = margin
    lines_on_page = 0

    for line in all_lines:
        if page is None or lines_on_page >= max_lines_per_page:
            page = doc.new_page(width=width, height=height)
            x, y = margin, margin
            lines_on_page = 0
        page.insert_text((x, y), line, fontsize=fontsize, fontname=fontname)
        y += line_h
        lines_on_page += 1

    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes

# ---- Unir varios PDFs en uno solo (PyMuPDF) ----
def merge_pdfs(pdf_paths: List[str]) -> bytes:
    import fitz
    out_doc = fitz.open()
    for path in pdf_paths:
        src = fitz.open(path)
        out_doc.insert_pdf(src)
        src.close()
    merged_bytes = out_doc.tobytes()
    out_doc.close()
    return merged_bytes

# ---- Split del texto completo en secciones ----
def split_text_into_sections(text: str) -> List[Tuple[int, str]]:
    """
    Divide el texto completo en una lista de tuplas (numero_seccion, texto_de_la_seccion).

    Los encabezados esperados son líneas tipo:
        --- Sección 0 ---
        --- Seccion 1 ---
    (con o sin tilde en 'ó').

    Si no se encuentra ninguna sección, devuelve una sola "sección" 0 con todo el texto.
    También, si hay texto antes de la primera sección, se agrega como sección -1 (si no está vacío).
    """
    matches = list(SECTION_HEADER_REGEX.finditer(text))
    if not matches:
        # No hay encabezados: todo es una sola sección 0
        return [(0, text)]

    sections: List[Tuple[int, str]] = []

    # Texto antes de la primera sección
    first_start = matches[0].start()
    preamble = text[:first_start].strip()
    if preamble:
        sections.append((-1, preamble))  # sección "especial" que nunca se procesa

    for i, m in enumerate(matches):
        sec_num_str = m.group(1)
        sec_num = int(sec_num_str)
        start = m.start()
        # Hasta el inicio del siguiente encabezado o fin de texto
        if i + 1 < len(matches):
            end = matches[i + 1].start()
        else:
            end = len(text)

        full_section_text = text[start:end].strip("\n")
        sections.append((sec_num, full_section_text))

    return sections

# =======================================
# ---- Helpers Prompt 1 ----
def run_prompt1_on_first_pages(
    pages_text: List[str],
    n_pages: int = 2,
) -> str:
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
    # IMPORTANTE: Prompt 1 se devuelve sin strip_think_segment
    return out.strip()


def parse_llm_list(text: str) -> List[str]:
    t = text.strip()
    start = t.find("[")
    end = t.rfind("]")
    if start != -1 and end != -1:
        t = t[start : end + 1]

    try:
        return json.loads(t)
    except json.JSONDecodeError:
        import ast
        return ast.literal_eval(t)


def postprocess_patient_data(raw_list: List[str]) -> List[str]:
    """
    Ejemplos de entrada:
      ["BENGELO RAKOTO MOHAMUD", "AA 6723.098-0", "ABUBAKAR 431"]

    Salida:
      ["BENGELO", "RAKOTO", "MOHAMUD", "AA67230980", "ABUBAKAR 431"]
    """
    if len(raw_list) < 3:
        raise ValueError(
            f"Se esperaban al menos 3 elementos [nombre, documento, direccion], llegó: {raw_list}"
        )

    full_name = str(raw_list[0]).strip()
    doc       = str(raw_list[1]).strip()
    address   = str(raw_list[2]).strip()

    name_parts = [p for p in full_name.split() if p]

    # dejar solo letras y números en el documento (sin espacios ni puntuación)
    clean_doc = re.sub(r"[^0-9A-Za-z]", "", doc)

    return [*name_parts, clean_doc, address]


def extract_patient_data_chain(pages_text: List[str]) -> Tuple[str, List[str], List[str]]:
    """
    Ejecuta todo Prompt 1:
      - corre el modelo sobre las 2 primeras páginas,
      - parsea la lista,
      - la postprocesa.
    """
    result_prompt_1 = run_prompt1_on_first_pages(pages_text, n_pages=2)
    raw_list = parse_llm_list(result_prompt_1)
    processed_list = postprocess_patient_data(raw_list)
    return result_prompt_1, raw_list, processed_list

# =======================================
# ---- Prompt 2: censurar un bloque usando la lista ----
def censor_block_text(block_text: str, patient_data_list: List[str]) -> str:
    if not block_text.strip():
        return ""

    lista_str = json.dumps(patient_data_list, ensure_ascii=False)

    if USE_CHUNKING and len(block_text) > MAX_CHARS_PER_CHUNK:
        chunks = chunk_text_by_chars(block_text, MAX_CHARS_PER_CHUNK, OVERLAP)
        out_parts: List[str] = []
        for ch in chunks:
            prompt = PROMPT2_TEMPLATE.format(lista=lista_str, text=ch)
            out = ollama_generate(
                model=MODEL_NAME,
                prompt=prompt,
                endpoint=OLLAMA_ENDPOINT,
                temperature=TEMPERATURE,
            )
            # Cortar en el primer /think (si lo hay)
            out = strip_think_segment(out)
            out_parts.append(out)
        return "\n\n".join([p for p in out_parts if p.strip()]).strip()
    else:
        prompt = PROMPT2_TEMPLATE.format(lista=lista_str, text=block_text)
        out = ollama_generate(
            model=MODEL_NAME,
            prompt=prompt,
            endpoint=OLLAMA_ENDPOINT,
            temperature=TEMPERATURE,
        )
        # Cortar en el primer /think (si lo hay)
        out = strip_think_segment(out)
        return out

# =======================================
# ---- Prompt 3: marcar carga viral en un bloque ----
def tag_cv_in_block_text(block_text: str) -> str:
    if not block_text.strip():
        return ""

    if USE_CHUNKING and len(block_text) > MAX_CHARS_PER_CHUNK:
        chunks = chunk_text_by_chars(block_text, MAX_CHARS_PER_CHUNK, OVERLAP)
        out_parts: List[str] = []
        for ch in chunks:
            prompt = PROMPT3_TEMPLATE.replace("{text}", ch)
            out = ollama_generate(
                model=MODEL_NAME,
                prompt=prompt,
                endpoint=OLLAMA_ENDPOINT,
                temperature=TEMPERATURE,
            )
            # Cortar en el primer /think (si lo hay)
            out = strip_think_segment(out)
            out_parts.append(out)
        return "\n\n".join([p for p in out_parts if p.strip()]).strip()
    else:
        prompt = PROMPT3_TEMPLATE.replace("{text}", block_text)
        out = ollama_generate(
            model=MODEL_NAME,
            prompt=prompt,
            endpoint=OLLAMA_ENDPOINT,
            temperature=TEMPERATURE,
        )
        # Cortar en el primer /think (si lo hay)
        out = strip_think_segment(out)
        return out

# =======================================
# ---- Prompt 5: tratar sección "Responsables del registro" ----
def responsables_block_text(block_text: str) -> str:
    """
    Aplica PROMPT 5 al texto del bloque para censurar la sección
    'Responsables del registro' y nombres/apellidos asociados.
    """
    if not block_text.strip():
        return ""

    if USE_CHUNKING and len(block_text) > MAX_CHARS_PER_CHUNK:
        chunks = chunk_text_by_chars(block_text, MAX_CHARS_PER_CHUNK, OVERLAP)
        out_parts: List[str] = []
        for ch in chunks:
            prompt = PROMPT5_TEMPLATE.replace("{text}", ch)
            out = ollama_generate(
                model=MODEL_NAME,
                prompt=prompt,
                endpoint=OLLAMA_ENDPOINT,
                temperature=TEMPERATURE,
            )
            # Cortar en el primer /think (si lo hay)
            out = strip_think_segment(out)
            out_parts.append(out)
        return "\n\n".join([p for p in out_parts if p.strip()]).strip()
    else:
        prompt = PROMPT5_TEMPLATE.replace("{text}", block_text)
        out = ollama_generate(
            model=MODEL_NAME,
            prompt=prompt,
            endpoint=OLLAMA_ENDPOINT,
            temperature=TEMPERATURE,
        )
        # Cortar en el primer /think (si lo hay)
        out = strip_think_segment(out)
        return out

# =======================================
# ---- Etapa única por bloque ----
def process_block_full(block_text: str, patient_data_list: List[str]) -> str:
    """
    Etapa única por bloque:
      - Prompt 2: censurar datos del paciente (lista).
      - Prompt 5: tratar sección 'Responsables del registro'.
      - Prompt 3: marcar carga viral y redondear al millar más cercano.

    Regla adicional:
      Si algún prompt devuelve texto vacío (len == 0 tras strip),
      se conserva el resultado del paso anterior para no perder contenido.
    """
    if not block_text.strip():
        return ""

    if DEBUG_PIPELINE:
        print("\n===== Nuevo bloque — ETAPA ÚNICA =====")
        print("LEN original bloque:", len(block_text))

    # -------- Prompt 2 --------
    censored_text = censor_block_text(block_text, patient_data_list)
    if not censored_text.strip():
        if DEBUG_PIPELINE:
            print("Prompt 2 devolvió vacío, se conserva el texto original del bloque.")
        censored_text = block_text

    if DEBUG_PIPELINE:
        print("LEN después de Prompt 2 (censura específica):", len(censored_text))

    # -------- Prompt 5 --------
    text_resp = responsables_block_text(censored_text)
    if not text_resp.strip():
        if DEBUG_PIPELINE:
            print("Prompt 5 devolvió vacío, se conserva el resultado de Prompt 2.")
        text_resp = censored_text

    if DEBUG_PIPELINE:
        print("LEN después de Prompt 5 (responsables):", len(text_resp))

    # -------- Prompt 3 --------
    final_text = tag_cv_in_block_text(text_resp)
    if not final_text.strip():
        if DEBUG_PIPELINE:
            print("Prompt 3 devolvió vacío, se conserva el resultado de Prompt 5.")
        final_text = text_resp

    if DEBUG_PIPELINE:
        print("LEN después de Prompt 3 (redondeo CV):", len(final_text))

    return final_text

# =======================================
# ---- CORE: PDF → PDF final usando 1 etapa por bloque (por SECCIONES) ----
def full_pipeline_pdf_pages_to_merged_pdf(
    pages_text: List[str],
    patient_data_list: List[str],
    sections_per_block: int = SECTIONS_PER_BLOCK,
    section_start: Optional[int] = None,
    section_end: Optional[int] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    tempdir_callback: Optional[Callable[[str], None]] = None,
) -> bytes:
    """
    Nuevo comportamiento:
    - Se une el texto de todas las páginas.
    - Se divide en secciones usando encabezados tipo '--- Sección N ---'.
    - Se define un rango [section_start, section_end] de secciones a procesar.
    - Solo esas secciones se pasan por el pipeline (P2+P5+P3) en bloques de `sections_per_block`.
    - El resto de las secciones se copia tal cual (sin procesar).
    - Cada bloque (procesado o crudo) se convierte a un PDF parcial; al final se unen todos.
    """
    full_text = "\n".join(pages_text).strip()
    if not full_text:
        return b""

    sections = split_text_into_sections(full_text)  # List[(sec_id, sec_text)]

    # Determinar rango por defecto si no se especifica
    section_ids = [sid for sid, _ in sections if sid >= 0]
    if section_ids:
        min_sec = min(section_ids)
        max_sec = max(section_ids)
    else:
        min_sec = 0
        max_sec = 0

    if section_start is None:
        section_start = min_sec
    if section_end is None:
        section_end = max_sec

    # Set de secciones que vamos a procesar
    sections_to_process = {
        sid for sid in section_ids if section_start <= sid <= section_end
    }

    # Construimos "chunks" de texto:
    #   - Chunks procesables: hasta `sections_per_block` secciones consecutivas dentro del rango.
    #   - Chunks crudos: secciones fuera del rango, tal cual.
    chunks: List[Tuple[str, bool]] = []  # (texto, debe_procesarse)
    buffer: List[str] = []               # para agrupar secciones procesables
    buffer_count = 0

    def flush_buffer():
        nonlocal buffer, buffer_count
        if buffer:
            block_text = "\n".join(buffer).strip()
            if block_text:
                chunks.append((block_text, True))
        buffer = []
        buffer_count = 0

    for sid, sec_text in sections:
        in_range = (sid in sections_to_process) if sid >= 0 else False

        if not in_range:
            # Si hay un bloque procesable pendiente, lo cerramos
            flush_buffer()
            # Esta sección se agrega cruda
            if sec_text.strip():
                chunks.append((sec_text.strip(), False))
        else:
            # Sección dentro del rango a procesar
            buffer.append(sec_text.strip())
            buffer_count += 1
            if buffer_count >= sections_per_block:
                flush_buffer()

    # Si quedó un bloque procesable pendiente al final
    flush_buffer()

    # Si por alguna razón no hay chunks (texto vacío), devolvemos vacío
    if not chunks:
        return b""

    # Contar cuantos bloques realmente van a pasar por el modelo (para la barra de progreso)
    total_blocks_to_process = sum(1 for _, do_proc in chunks if do_proc)

    temp_dir = tempfile.mkdtemp(prefix="anon_sections_")
    if tempdir_callback is not None:
        tempdir_callback(temp_dir)

    final_block_paths: List[str] = []

    try:
        processed_blocks = 0

        for idx, (chunk_text, do_process) in enumerate(chunks, start=1):
            if do_process:
                # Bloque que va por el pipeline
                if DEBUG_PIPELINE:
                    print(f"\n### Procesando bloque de secciones (chunk #{idx}) ###")
                    print("LEN chunk original:", len(chunk_text))

                # 1) Agregar marcador de nueva sección al inicio del bloque
                block_text_with_marker = add_section_marker(chunk_text)

                # 2) Procesar el bloque completo (P2 + P5 + P3)
                block_result_text = process_block_full(block_text_with_marker, patient_data_list)

                # 3) Eliminar el marcador ---Sección Intermedia--- antes de guardar
                block_result_text_clean = remove_section_marker(block_result_text)

                # 4) Convertir a PDF
                block_pdf_bytes = text_to_pdf_bytes(block_result_text_clean)

                processed_blocks += 1
                if progress_callback is not None and total_blocks_to_process > 0:
                    progress_callback(processed_blocks, total_blocks_to_process)
            else:
                # Bloque crudo (fuera de rango), sin pasar por el modelo
                if DEBUG_PIPELINE:
                    print(f"\n### Bloque sin procesar (chunk #{idx}) ###")
                    print("LEN chunk original:", len(chunk_text))

                block_pdf_bytes = text_to_pdf_bytes(chunk_text)

            block_filename = f"block_{idx:04d}.pdf"
            block_path = os.path.join(temp_dir, block_filename)
            with open(block_path, "wb") as f:
                f.write(block_pdf_bytes)

            final_block_paths.append(block_path)

            # Liberar memoria de este chunk
            del block_pdf_bytes
            del chunk_text

        # Unir todos los PDFs (procesados + crudos)
        merged_pdf_bytes = merge_pdfs(sorted(final_block_paths))
        return merged_pdf_bytes

    finally:
        try:
            shutil.rmtree(temp_dir)
        except Exception:
            pass

# =======================================
# ---- UI Streamlit ----
def main():
    global MODEL_NAME, TEMPERATURE, NUM_CTX, NUM_PREDICT  # para poder pisar los globales según la selección

    st.set_page_config(page_title="PDF → 🧠 Anonimización con LLM (Ollama)", layout="centered")
    st.title("📄 PDF → 🧠 Anonimización con LLM (Ollama)")

    # ==== Selector de modelo ====
    model_label = st.selectbox(
        "Elegí el modelo a utilizar",
        list(MODEL_CONFIGS.keys()),
        index=0,
    )
    cfg = MODEL_CONFIGS[model_label]

    # Actualizar parámetros globales en base al modelo elegido
    MODEL_NAME  = cfg["model_name"]
    TEMPERATURE = cfg["temperature"]
    NUM_CTX     = cfg["num_ctx"]
    NUM_PREDICT = cfg["num_predict"]

    st.caption(
        f"Modelo seleccionado: **{model_label}** "
        f"(id: `{MODEL_NAME}`, temperatura: {TEMPERATURE}, num_ctx: {NUM_CTX}, num_predict: {NUM_PREDICT})"
    )

    uploaded = st.file_uploader("Subí un PDF", type=["pdf"])

    if uploaded is not None:
        pdf_bytes = uploaded.read()
        with st.spinner("Extrayendo texto del PDF..."):
            try:
                pages_text = pdf_bytes_to_pages(pdf_bytes)
            except Exception as e:
                st.error(f"Error al leer el PDF: {e}")
                st.stop()

        num_pages = len(pages_text)
        full_text = "\n".join(pages_text).strip()

        # Dividir en secciones para mostrar info y permitir elegir rango
        sections = split_text_into_sections(full_text)
        section_ids = [sid for sid, _ in sections if sid >= 0]
        if section_ids:
            min_sec = min(section_ids)
            max_sec = max(section_ids)
        else:
            min_sec = 0
            max_sec = 0

        st.success(f"PDF leído correctamente. Páginas detectadas: {num_pages}")
        st.caption(
            f"Caracteres extraídos (total): {len(full_text)} | "
            f"chunk: {MAX_CHARS_PER_CHUNK} | overlap: {OVERLAP} | "
            f"bloque de secciones: {SECTIONS_PER_BLOCK}"
        )
        st.caption(
            f"Secciones detectadas: {len([sid for sid in section_ids])} "
            f"(rango numérico: {min_sec}–{max_sec})"
        )

        st.text_area(
            "Vista previa del texto (primeras páginas)",
            value=full_text[:2000] + ("..." if len(full_text) > 2000 else ""),
            height=200,
        )

        # Controles para elegir qué secciones procesar
        start_section = st.number_input(
            "Sección inicial a procesar",
            min_value=int(min_sec),
            max_value=int(max_sec),
            value=int(min_sec),
            step=1,
        )
        end_section = st.number_input(
            "Sección final a procesar",
            min_value=int(start_section),
            max_value=int(max_sec),
            value=int(max_sec),
            step=1,
        )
        st.caption(
            f"Se procesarán solo las secciones desde la {start_section} hasta la {end_section} "
            f"en bloques de {SECTIONS_PER_BLOCK} secciones."
        )

        if st.button("🚀 Ejecutar modelo (P1 + (P2+P5+P3-redondeo-CV) en 1 etapa por bloque de SECCIONES)"):
            # Paso 1: Prompt 1 para extraer datos del encabezado (se mantiene sobre las primeras páginas)
            with st.spinner("Paso 1: ejecutando Prompt 1 (extraer datos del encabezado)..."):
                try:
                    result_prompt_1, raw_list, patient_data_list = extract_patient_data_chain(
                        pages_text
                    )
                except Exception as e:
                    st.error(f"Error en Prompt 1: {e}")
                    st.stop()

            st.session_state["prompt1_raw_response"] = result_prompt_1
            st.session_state["prompt1_raw_list"] = raw_list
            st.session_state["patient_data_list"] = patient_data_list

            st.success("Prompt 1 completado. Datos detectados:")
            st.write("Lista procesada (nombres, doc, dirección):", patient_data_list)

            progress_bar = st.progress(0)
            status_placeholder = st.empty()
            tempdir_placeholder = st.empty()

            def progress_cb(current_block: int, total_blocks: int) -> None:
                pct = int(current_block / total_blocks * 100)
                progress_bar.progress(pct)
                status_placeholder.write(
                    f"Procesando bloque {current_block} de {total_blocks} (1 etapa por bloque de SECCIONES)..."
                )

            def tempdir_cb(path: str) -> None:
                tempdir_placeholder.caption(f"Carpeta temporal usada: {path}")

            with st.spinner(
                "Procesando bloques en 1 etapa: (P2+P5+P3 con redondeo de carga viral, por secciones)..."
            ):
                try:
                    final_pdf_bytes = full_pipeline_pdf_pages_to_merged_pdf(
                        pages_text,
                        patient_data_list=patient_data_list,
                        sections_per_block=SECTIONS_PER_BLOCK,
                        section_start=int(start_section),
                        section_end=int(end_section),
                        progress_callback=progress_cb,
                        tempdir_callback=tempdir_cb,
                    )
                except Exception as e:
                    st.error(f"Error durante el procesamiento por bloques: {e}")
                    st.stop()

            progress_bar.progress(100)
            status_placeholder.write("Procesamiento completado ✅")

            if not final_pdf_bytes:
                st.warning(
                    "La salida está vacía. Revisá el PDF original o ajustá parámetros."
                )
            else:
                st.subheader("📄 Descarga")
                st.download_button(
                    "📥 Descargar PDF anonimizado",
                    data=io.BytesIO(final_pdf_bytes),
                    file_name="salida_ollama_anonimizada.pdf",
                    mime="application/pdf",
                )
                st.success(
                    "¡Listo! Se ejecutó el pipeline completo en 1 etapa por bloque de secciones y se generó el PDF anonimizado ✅"
                )

    else:
        st.info("Subí un PDF para comenzar.")


if __name__ == "__main__":
    main()
