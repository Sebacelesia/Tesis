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
from datetime import datetime, date

# ====== CONFIGURACIONES DE MODELOS ======
# Ajustá los nombres de modelo ("model_name") según cómo los tengas en Ollama.
# Acá también se definen los hiperparámetros de decodificación por defecto.
MODEL_CONFIGS = {
    "Qwen 8B (Ollama)": {
        "model_name":       "qwen3:8b",
        "temperature":      0.0,
        "num_ctx":          16384,
        "num_predict":      9000,
        "top_p":            0.95,
        "top_k":            20,
        "repeat_penalty":   1.1,
        "repeat_last_n":    256,
        "presence_penalty": 0.5,
        "use_think_trunc":  True,   # cortar en /think
    },
    "Qwen 4B (Ollama)": {
        "model_name":       "qwen3:4b",   # <-- CAMBIÁ ESTE NOMBRE AL QUE USES EN OLLAMA
        "temperature":      0.0,
        "num_ctx":          12000,           # más contexto para el modelo grande
        "num_predict":      6000,
        "top_p":            0.95,
        "top_k":            20,
        "repeat_penalty":   1.1,
        "repeat_last_n":    256,
        "presence_penalty": 0.0,
        "use_think_trunc":  True,
    },
}

# ====== PARÁMETROS FIJOS (se inicializan con Qwen 8B y luego se pisan según la selección en la UI) ======
OLLAMA_ENDPOINT = "http://localhost:11434"

# Tomamos como base la config de Qwen 8B
_DEFAULT_CFG = MODEL_CONFIGS["Qwen 8B (Ollama)"]
MODEL_NAME      = _DEFAULT_CFG["model_name"]
TEMPERATURE     = _DEFAULT_CFG["temperature"]

USE_CHUNKING          = True            # si el texto supera MAX_CHARS_PER_CHUNK, se parte
MAX_CHARS_PER_CHUNK   = 15000           # caracteres por chunk de texto (del documento)
OVERLAP               = 0               # solapamiento entre chunks (en caracteres)

# Procesar de a N secciones por bloque lógico
SECTIONS_PER_BLOCK    = 1               # secciones por bloque

# Importante para evitar cortes por contexto/salida en Ollama:
NUM_CTX               = _DEFAULT_CFG["num_ctx"]      # contexto (tokens del modelo) en Ollama
NUM_PREDICT           = _DEFAULT_CFG["num_predict"]  # tokens de salida máximos

# Flag de debug para ver longitudes en consola
DEBUG_PIPELINE        = True

# ====== Parámetros de decodificación adicionales (se pisan según el modelo elegido) ======
TOP_P              = _DEFAULT_CFG["top_p"]
TOP_K              = _DEFAULT_CFG["top_k"]
REPEAT_PENALTY     = _DEFAULT_CFG["repeat_penalty"]
REPEAT_LAST_N      = _DEFAULT_CFG["repeat_last_n"]
PRESENCE_PENALTY   = _DEFAULT_CFG["presence_penalty"]
USE_THINK_TRUNC    = _DEFAULT_CFG["use_think_trunc"]

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

Tu tarea es identificar TODAS las menciones de carga viral en el texto y reemplazar el número por un numero como se muestra a continuacion.

Instrucciones obligatorias:
1) Considera como mención de carga viral expresiones en las que aparezcan términos como
   "carga viral", "Carga viral", "CV", "cv" cerca de un número, por ejemplo:
   - "carga viral: 120000"
   - "carga viral 120000"
   - "CV: 120000"
   - "cv: 120000"
   - "CV 120000"
   u otras variantes similares.

2) Para cada número que represente una carga viral sustituye este por el redondeo al millar mas cercano, exceptuando los valores entre 0 y 1000.

   Ejemplos de como cambiar el numero:

   - 123456 → 123000
   - 1201 → 1000
   - 625 → 625
   - 10000 → 10000
   - -234567 → -235000

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

# =======================================
# ---- construir texto seccionado por "Fecha:" ----
def build_sectioned_text_from_pages(pages_text: List[str]) -> str:
    """
    Une todas las páginas en un solo texto, lo divide por 'Fecha:'
    y arma secciones numeradas empezando en 1:

        --- Sección 1 ---
        Fecha: ...
        ...

        --- Sección 2 ---
        Fecha: ...
        ...
    """
    full_text = "\n".join(pages_text).strip()
    if not full_text:
        return ""

    # Dividir evoluciones por "Fecha:"
    evoluciones = re.split(r'(?=Fecha:)', full_text)

    # Filtrar secciones muy cortas (por ejemplo, menos de 10 palabras)
    evoluciones_filtradas = [sec.strip() for sec in evoluciones if len(sec.split()) >= 10]

    partes: List[str] = []
    for i, sec in enumerate(evoluciones_filtradas, start=1):
        partes.append(f"--- Sección {i} ---\n{sec.strip()}\n")

    return "\n\n".join(partes).strip()

# ---- Llamada a Ollama (streaming) ----
def ollama_generate(
    model: str,
    prompt: str,
    endpoint: str = OLLAMA_ENDPOINT,
    temperature: float = TEMPERATURE,
    options: Optional[dict] = None,
) -> str:
    base_opts = {
        "temperature":      temperature,
        "num_ctx":          NUM_CTX,
        "num_predict":      NUM_PREDICT,
        "top_p":            TOP_P,
        "top_k":            TOP_K,
        "repeat_penalty":   REPEAT_PENALTY,   # >1 penaliza repetir tokens
        "repeat_last_n":    REPEAT_LAST_N,    # mira las últimas N tokens para penalizar
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

def maybe_strip_think(text: str) -> str:
    """
    Aplica strip_think_segment solo si USE_THINK_TRUNC es True.
    Si no, simplemente hace strip().
    """
    if text is None:
        text = ""
    if USE_THINK_TRUNC:
        return strip_think_segment(text)
    return text.strip()

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
        --- Sección 1 ---
        --- Seccion 2 ---
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

# ---- extraer fecha de una sección ----
def extract_date_from_section_text(text: str) -> Optional[str]:
    """
    Busca una fecha con formato dd/mm/aaaa en el texto de la sección.
    Devuelve un string 'dd/mm/aaaa' o None si no se encuentra.
    """
    m = re.search(r"\b(\d{1,2}/\d{1,2}/\d{4})\b", text)
    if m:
        return m.group(1).strip()
    return None

def map_section_ids_to_dates(sections: List[Tuple[int, str]]):
    """
    Devuelve:
      - dict {id_seccion -> date | None}
      - lista con todas las fechas (tipo date) encontradas
    """
    section_dates = {}
    all_dates: List[date] = []

    for sid, sec_text in sections:
        if sid >= 0:
            d_str = extract_date_from_section_text(sec_text)
            if d_str:
                try:
                    d = datetime.strptime(d_str, "%d/%m/%Y").date()
                except ValueError:
                    d = None
            else:
                d = None
            section_dates[sid] = d
            if d is not None:
                all_dates.append(d)

    return section_dates, all_dates

# ================ NUEVO: normalizar signos de puntuación en cada elemento =================
def normalize_punctuation_in_list(seq: List[str]) -> List[str]:
    """
    Para cada elemento:
      - Reemplaza , . : ; por un espacio.
      - Colapsa espacios múltiples.
      - Aplica strip().
      - Descarta strings vacíos.

    Ejemplo:
      ["Sebastian, celesia", ",", "Montevideo; 123"]
        → ["Sebastian Celesia", "Montevideo 123"]
    """
    cleaned: List[str] = []
    for elem in seq:
        s = str(elem)
        # Reemplazar , . : ; por espacio
        s = re.sub(r"[,\.;:]", " ", s)
        # Colapsar espacios múltiples
        s = re.sub(r"\s+", " ", s)
        s = s.strip()
        if s:
            cleaned.append(s)
    return cleaned

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
    # IMPORTANTE: Prompt 1 se devuelve sin maybe_strip_think para no romper el parseo de la lista
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
    Ejemplos de entrada ideal:
      ["BENGELO RAKOTO MOHAMUD", "AA 6723.098-0", "ABUBAKAR 431"]

    Asume que raw_list ya viene con la puntuación normalizada
    (sin comas/puntos/;/: sueltos, y con espacios bien colocados).
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
      - normaliza signos de puntuación en cada elemento,
      - la postprocesa.
    """
    result_prompt_1 = run_prompt1_on_first_pages(pages_text, n_pages=2)

    # 1) String → lista python (cruda)
    raw_list = parse_llm_list(result_prompt_1)

    # 2) 🔧 Normalizar puntuación dentro de cada elemento
    raw_list_clean = normalize_punctuation_in_list(raw_list)

    # 3) Postproceso (usa la lista limpia)
    processed_list = postprocess_patient_data(raw_list_clean)

    # Devolvemos la lista limpia para mostrar en UI
    return result_prompt_1, raw_list_clean, processed_list

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
            out = maybe_strip_think(out)
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
        out = maybe_strip_think(out)
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
            out = maybe_strip_think(out)
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
        out = maybe_strip_think(out)
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
            out = maybe_strip_think(out)
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
        out = maybe_strip_think(out)
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
# ---- CORE: PDF → PDF final usando 1 etapa por bloque (filtrado por RANGO DE FECHAS) ----
def full_pipeline_pdf_pages_to_merged_pdf(
    pages_text: List[str],
    patient_data_list: List[str],
    sections_per_block: int = SECTIONS_PER_BLOCK,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    tempdir_callback: Optional[Callable[[str], None]] = None,
    warnings_collector: Optional[List[str]] = None,
) -> Tuple[bytes, List[str]]:
    """
    Nuevo comportamiento:
    - Se construye un texto seccionado a partir de las páginas (por 'Fecha:', secciones 1..N).
    - Se divide en secciones usando encabezados tipo '--- Sección N ---'.
    - Se extrae la fecha de cada sección.
    - Se procesan SOLO las secciones cuya fecha esté entre [start_date, end_date] (inclusive),
      y SIEMPRE se incluye la sección 1 si existe.
    - Si ninguna sección tiene fecha detectable, se ignora el filtro y se procesan todas.

    NUEVO:
    - Si a nivel de bloque se pierde más de 150 caracteres respecto al original, se reintenta
      procesar ese bloque sección por sección.
    - A nivel de sección, si se pierden más de 250 caracteres, se deja la sección original
      y se agrega un warning al collector.
    """
    if warnings_collector is None:
        warnings_collector = []

    full_text = build_sectioned_text_from_pages(pages_text)
    if not full_text:
        return b"", warnings_collector

    sections = split_text_into_sections(full_text)  # List[(sec_id, sec_text)]
    section_ids = [sid for sid, _ in sections if sid >= 0]

    # Mapear sección -> fecha
    section_dates, all_dates = map_section_ids_to_dates(sections)

    # ¿Hay al menos una fecha válida en alguna sección?
    has_any_date = len(all_dates) > 0

    if not has_any_date or start_date is None or end_date is None:
        # Sin fechas o sin rango: procesar todas las secciones
        sections_to_process_ids = set(section_ids)
    else:
        # Filtrar por rango
        sections_to_process_ids = {
            sid
            for sid in section_ids
            if (section_dates.get(sid) is not None
                and start_date <= section_dates[sid] <= end_date)
        }

    # SIEMPRE incluir sección 1 si existe
    if 1 in section_ids:
        sections_to_process_ids.add(1)

    # Filtrar solo las secciones dentro del set (las otras se descartan)
    sections_to_use: List[Tuple[int, str]] = []
    for sid, sec_text in sections:
        if sid >= 0 and sid in sections_to_process_ids:
            txt = sec_text.strip()
            if txt:
                sections_to_use.append((sid, txt))

    # Si no hay secciones para procesar, devolvemos vacío
    if not sections_to_use:
        return b"", warnings_collector

    # Construimos bloques de hasta `sections_per_block` secciones consecutivas (en orden lógico)
    # Ahora guardamos también el id de la sección para poder loguear warnings.
    chunks: List[List[Tuple[int, str]]] = []
    buffer: List[Tuple[int, str]] = []

    for sid, sec_text in sections_to_use:
        buffer.append((sid, sec_text))
        if len(buffer) >= sections_per_block:
            chunks.append(buffer)
            buffer = []

    # Bloque final si quedó algo pendiente
    if buffer:
        chunks.append(buffer)

    if not chunks:
        return b"", warnings_collector

    total_blocks_to_process = len(chunks)

    temp_dir = tempfile.mkdtemp(prefix="anon_sections_")
    if tempdir_callback is not None:
        tempdir_callback(temp_dir)

    final_block_paths: List[str] = []

    try:
        processed_blocks = 0

        for idx, section_block in enumerate(chunks, start=1):
            # Texto original del bloque: concatenación de las secciones
            block_text = "\n".join(sec_text for _, sec_text in section_block).strip()
            if DEBUG_PIPELINE:
                print(f"\n### Procesando bloque de secciones (chunk #{idx}) ###")
                print("LEN chunk original:", len(block_text))

            # Procesar bloque completo
            block_result_text = process_block_full(block_text, patient_data_list)

            # CONTROL DE LONGITUD A NIVEL BLOQUE
            len_orig_block = len(block_text)
            len_res_block  = len(block_result_text)
            diff_block     = len_orig_block - len_res_block

            if DEBUG_PIPELINE:
                print(
                    f"[BLOQUE #{idx}] len_orig={len_orig_block}, "
                    f"len_res={len_res_block}, diff={diff_block}"
                )

            # Si el resultado recorta MÁS de 150 caracteres → fallback por sección
            if diff_block > 150:
                if DEBUG_PIPELINE:
                    print(
                        f"[BLOQUE #{idx}] Diferencia > 150 caracteres. "
                        "Reprocesando sección por sección."
                    )

                per_section_outputs: List[str] = []
                for sid, sec_text in section_block:
                    sec_orig_len = len(sec_text)
                    sec_result   = process_block_full(sec_text, patient_data_list)
                    sec_result_stripped = sec_result.strip()

                    if not sec_result_stripped:
                        # Si queda vacío, claramente es riesgoso → mantener original
                        diff_sec = sec_orig_len
                    else:
                        diff_sec = sec_orig_len - len(sec_result_stripped)

                    if DEBUG_PIPELINE:
                        print(
                            f"  - Sección {sid}: len_orig={sec_orig_len}, "
                            f"len_res={len(sec_result_stripped)}, diff={diff_sec}"
                        )

                    # Si se pierden más de 250 chars a nivel sección → warning y usar original
                    if diff_sec > 250:
                        msg = (
                            f"Sección {sid}: no se pudo anonimizar de forma segura "
                            f"(diferencia de longitud {diff_sec}). "
                            "Se dejó el texto original."
                        )
                        warnings_collector.append(msg)
                        per_section_outputs.append(sec_text)
                    else:
                        # Si no hay diferencia excesiva y hay algo de texto, usamos el resultado
                        if sec_result_stripped:
                            per_section_outputs.append(sec_result_stripped)
                        else:
                            # En caso de duda, conservar original
                            per_section_outputs.append(sec_text)

                # Texto final del bloque después del fallback por sección
                block_result_text = "\n".join(per_section_outputs).strip()

            # Convertir texto final del bloque (sea del camino normal o del fallback) a PDF
            block_pdf_bytes = text_to_pdf_bytes(block_result_text)

            processed_blocks += 1
            if progress_callback is not None and total_blocks_to_process > 0:
                progress_callback(processed_blocks, total_blocks_to_process)

            block_filename = f"block_{idx:04d}.pdf"
            block_path = os.path.join(temp_dir, block_filename)
            with open(block_path, "wb") as f:
                f.write(block_pdf_bytes)

            final_block_paths.append(block_path)

            del block_pdf_bytes
            del block_text
            del block_result_text

        merged_pdf_bytes = merge_pdfs(sorted(final_block_paths))
        return merged_pdf_bytes, warnings_collector

    finally:
        try:
            shutil.rmtree(temp_dir)
        except Exception:
            pass

# =======================================
# ---- UI Streamlit ----
def main():
    global MODEL_NAME, TEMPERATURE, NUM_CTX, NUM_PREDICT
    global TOP_P, TOP_K, REPEAT_PENALTY, REPEAT_LAST_N, PRESENCE_PENALTY, USE_THINK_TRUNC

    st.set_page_config(page_title="PDF → 🧠 Anonimización con LLM (Ollama)", layout="centered")
    st.title("📄 PDF → 🧠 Anonimización con LLM (Ollama)")

    # ==== Selector de modelo ====
    model_label = st.selectbox(
        "Elegí el modelo a utilizar",
        list(MODEL_CONFIGS.keys()),
        index=0,
    )
    cfg = MODEL_CONFIGS[model_label]

    # Actualizar parámetros globales en base al modelo elegido (TODO en código, no en UI)
    MODEL_NAME       = cfg["model_name"]
    TEMPERATURE      = cfg["temperature"]
    NUM_CTX          = cfg["num_ctx"]
    NUM_PREDICT      = cfg["num_predict"]
    TOP_P            = cfg["top_p"]
    TOP_K            = cfg["top_k"]
    REPEAT_PENALTY   = cfg["repeat_penalty"]
    REPEAT_LAST_N    = cfg["repeat_last_n"]
    PRESENCE_PENALTY = cfg["presence_penalty"]
    USE_THINK_TRUNC  = cfg["use_think_trunc"]

    st.caption(
        "Modelo seleccionado: **{label}** (id: `{name}`, temp: {temp}, top_p: {top_p}, "
        "top_k: {top_k}, num_ctx: {num_ctx}, num_predict: {num_pred}, "
        "repeat_penalty: {rp}, presence_penalty: {pp}, think_trunc: {tt})".format(
            label=model_label,
            name=MODEL_NAME,
            temp=TEMPERATURE,
            top_p=TOP_P,
            top_k=TOP_K,
            num_ctx=NUM_CTX,
            num_pred=NUM_PREDICT,
            rp=REPEAT_PENALTY,
            pp=PRESENCE_PENALTY,
            tt="ON" if USE_THINK_TRUNC else "OFF",
        )
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

        # Construir texto seccionado
        sectioned_text = build_sectioned_text_from_pages(pages_text)

        # Dividir en secciones para info e identificar rango de fechas
        sections = split_text_into_sections(sectioned_text)
        section_ids = [sid for sid, _ in sections if sid >= 0]

        section_dates_map, all_dates = map_section_ids_to_dates(sections)

        st.success(f"PDF leído correctamente. Páginas detectadas: {num_pages}")
        st.caption(
            f"Caracteres extraídos (texto seccionado): {len(sectioned_text)} | "
            f"chunk: {MAX_CHARS_PER_CHUNK} | overlap: {OVERLAP} | "
            f"bloque de secciones: {SECTIONS_PER_BLOCK}"
        )
        st.caption(
            f"Secciones detectadas: {len(section_ids)}. "
            f"La Sección 1 se incluirá SIEMPRE en el resultado."
        )

        if all_dates:
            min_date_found = min(all_dates)
            max_date_found = max(all_dates)
            st.caption(
                f"Rango de fechas detectadas en las evoluciones: "
                f"{min_date_found.strftime('%d/%m/%Y')} – {max_date_found.strftime('%d/%m/%Y')}"
            )
        else:
            min_date_found = max_date_found = None
            st.warning(
                "No se detectaron fechas con formato dd/mm/aaaa en las secciones. "
                "Si no especificás rango, se procesarán todas las secciones."
            )

        st.text_area(
            "Vista previa del texto seccionado",
            value=sectioned_text[:2000] + ("..." if len(sectioned_text) > 2000 else ""),
            height=200,
        )

        # === Rango de fechas en la UI ===
        hoy = datetime.today().date()
        default_start = min_date_found or hoy
        default_end = max_date_found or hoy

        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Fecha inicial (inclusive)",
                value=default_start,
                format="DD/MM/YYYY",
            )
        with col2:
            end_date = st.date_input(
                "Fecha final (inclusive)",
                value=default_end,
                format="DD/MM/YYYY",
            )

        if end_date < start_date:
            st.error("La fecha final no puede ser anterior a la fecha inicial.")
            st.stop()

        st.caption(
            "Se procesarán todas las secciones cuya fecha esté dentro del rango indicado "
            "y, adicionalmente, la Sección 1 si existe. "
            "Si ninguna sección tiene fecha detectable, se ignora el filtro y se procesan todas."
        )

        if st.button("🚀 Ejecutar modelo (P1 + (P2+P5+P3-redondeo-CV) filtrando por rango de fechas)"):
            # Paso 1: Prompt 1 (encabezado)
            with st.spinner("Paso 1: ejecutando Prompt 1 (extraer datos del encabezado)..."):
                try:
                    result_prompt_1, raw_list, patient_data_list = extract_patient_data_chain(
                        pages_text
                    )
                except Exception as e:
                    st.error(f"Error en Prompt 1: {e}")
                    st.stop()

            # raw_list ya viene con "Sebastian, celesia" → "Sebastian Celesia", etc.
            st.session_state["prompt1_raw_response"] = result_prompt_1
            st.session_state["prompt1_raw_list"] = raw_list
            st.session_state["patient_data_list"] = patient_data_list

            st.success("Prompt 1 completado. Datos detectados:")
            st.write("Lista cruda normalizada:", raw_list)
            st.write("Lista procesada (nombres, doc, dirección):", patient_data_list)

            progress_bar = st.progress(0)
            status_placeholder = st.empty()
            tempdir_placeholder = st.empty()

            def progress_cb(current_block: int, total_blocks: int) -> None:
                pct = int(current_block / total_blocks * 100)
                progress_bar.progress(pct)
                status_placeholder.write(
                    f"Procesando bloque {current_block} de {total_blocks} "
                    f"(1 etapa por bloque, filtrado por rango de fechas)..."
                )

            def tempdir_cb(path: str) -> None:
                tempdir_placeholder.caption(f"Carpeta temporal usada: {path}")

            with st.spinner(
                "Procesando bloques en 1 etapa: (P2+P5+P3 con redondeo de carga viral, filtrando por rango de fechas)..."
            ):
                try:
                    final_pdf_bytes, warnings = full_pipeline_pdf_pages_to_merged_pdf(
                        pages_text,
                        patient_data_list=patient_data_list,
                        sections_per_block=SECTIONS_PER_BLOCK,
                        start_date=start_date,
                        end_date=end_date,
                        progress_callback=progress_cb,
                        tempdir_callback=tempdir_cb,
                        warnings_collector=[],
                    )
                except Exception as e:
                    st.error(f"Error durante el procesamiento por bloques: {e}")
                    st.stop()

            progress_bar.progress(100)
            status_placeholder.write("Procesamiento completado ✅")

            # Mostrar warnings de secciones no anonimizada de forma segura
            if warnings:
                st.warning(
                    "Algunas secciones no se pudieron anonimizar de forma segura y se dejaron "
                    "en su versión original:\n\n" +
                    "\n".join(f"- {msg}" for msg in warnings)
                )

            if not final_pdf_bytes:
                st.warning(
                    "La salida está vacía. Revisá el PDF original o el rango de fechas indicado."
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
                    "¡Listo! Se ejecutó el pipeline completo filtrando por rango de fechas y se generó el PDF anonimizado ✅"
                )

    else:
        st.info("Subí un PDF para comenzar.")


if __name__ == "__main__":
    main()