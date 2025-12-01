# app.py — Pipeline completo en 1 etapa por bloque:
# Etapa única: Prompt 2 + Prompt 5 + Prompt 3 (carga viral)

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
        "num_ctx":     25000,
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

# Procesar de a N páginas de PDF por bloque lógico
PAGES_PER_BLOCK       = 5               # páginas por bloque

# Importante para evitar cortes por contexto/salida en Ollama:
NUM_CTX               = MODEL_CONFIGS["Qwen 8B (Ollama)"]["num_ctx"]      # contexto (tokens del modelo) en Ollama
NUM_PREDICT           = MODEL_CONFIGS["Qwen 8B (Ollama)"]["num_predict"]  # tokens de salida máximos

# Flag de debug para ver longitudes en consola
DEBUG_PIPELINE        = True

# Marcador de nueva sección por bloque
SECTION_MARKER        = "---Sección Intermedia---"

# Umbral para detectar salidas "demasiado cortas"
LENGTH_DIFF_THRESHOLD = 250

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
3) Siempre que detectes la dirección de la lista en el documento cambialo por [CENSURADO]. Esta dirección de la lista no tiene porque ser idéntica a la que encuentres en el documento. Por dar algunos ejemplos, si en la lista está "Avenida Libertad 123" y en el documento aparece "Av. Libertad 123" esto debes cambiar por [CENSURADO] también. Si en la lista está "Mateo Cortez 2395" y en el documento está "M. Cortez 2395", "Cortez 2395" o incluso "Cortez numero 2395", esto debes cambiar por [CENSURADO] también.
4) Por favor manten el formato (no utilices negritas ni aumentes el tamaño de la letra).
5) MUY IMPORTANTE: si en el texto NO encuentras ninguno de los datos de la lista, devuelve el texto ORIGINAL sin ningún cambio y sin añadir frases como "no hay nada que censurar" ni otros comentarios.
6) Devuelve ÚNICAMENTE el documento censurado (o el original si no hay cambios), sin explicaciones ni notas adicionales.

A continuación te muestro la lista:

{lista}

A continuación te comparto el documento:
{text}
"""

# ====== PROMPT 3: marcar carga viral y redondear a decenas de millar ======
PROMPT3_TEMPLATE = """
Eres un especialista en análisis de textos clínicos en español.

Tu tarea es identificar TODAS las menciones de carga viral en el texto y reemplazar el número por el valor redondeado a las decenas de millar más cercanas.

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
) -> str:
    base_opts = {
        "temperature":    temperature,
        "num_ctx":        NUM_CTX,
        "num_predict":    NUM_PREDICT,
        # ==== Parámetros para penalizar repetición ====
        "repeat_penalty": 1.1,   # >1 penaliza repetir tokens
        "repeat_last_n":  256,   # mira las últimas N tokens para penalizar
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

# ---- Helper: cortar en el primer /think ----
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
    """Elimina TODAS las ocurrencias exactas de SECTION_MARKER del texto."""
    cleaned = text.replace(SECTION_MARKER, "")
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
    # Prompt 1 se devuelve sin strip_think_segment
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
        out = strip_think_segment(out)
        return out

# =======================================
# ---- Etapa única por bloque (ULTRA ESTRICTA) ----
def process_block_full(
    block_text_with_marker: str,
    patient_data_list: List[str],
    orig_len: int,
    length_diff_threshold: int = LENGTH_DIFF_THRESHOLD,
) -> Tuple[str, dict, bool, str, int]:
    """
    Etapa única por bloque (ULTRA ESTRICTA):
      - Prompt 2: censurar datos del paciente (lista).
      - Prompt 5: tratar sección 'Responsables del registro'.
      - Prompt 3: marcar carga viral y redondear.

    Lógica ultra estricta:
      - Después de CADA prompt, se mide len(salida).
      - Si len == 0 o (orig_len - len_salida) > length_diff_threshold:
            -> se marca fallback_needed = True
            -> NO se ejecutan los prompts siguientes.
            -> se devuelve inmediatamente.

    Devuelve:
      - final_text (lo que haya hasta el momento),
      - stats (con longitudes),
      - fallback_needed (bool),
      - bad_stage_label (str, si hubo problema),
      - bad_raw_len (int, len de la salida problemática).
    """
    stats = {"orig_len": orig_len}
    fallback_needed = False
    bad_stage_label = ""
    bad_raw_len = 0

    if not block_text_with_marker.strip():
        return "", stats, False, "", 0

    if DEBUG_PIPELINE:
        print("\n===== Nuevo bloque — ETAPA ÚNICA (estricto) =====")
        print("LEN original bloque (con marker):", orig_len)

    # -------- Prompt 2 --------
    p2_out = censor_block_text(block_text_with_marker, patient_data_list)
    p2_len = len(p2_out.strip())
    stats["p2_raw_len"] = p2_len

    if DEBUG_PIPELINE:
        print("LEN después de Prompt 2:", p2_len)

    if p2_len == 0 or (orig_len - p2_len) > length_diff_threshold:
        # Señal de que hay que reprocesar por página
        fallback_needed = True
        bad_stage_label = "Prompt 2"
        bad_raw_len = p2_len
        return p2_out, stats, fallback_needed, bad_stage_label, bad_raw_len

    # -------- Prompt 5 --------
    p5_out = responsables_block_text(p2_out)
    p5_len = len(p5_out.strip())
    stats["p5_raw_len"] = p5_len

    if DEBUG_PIPELINE:
        print("LEN después de Prompt 5:", p5_len)

    if p5_len == 0 or (orig_len - p5_len) > length_diff_threshold:
        fallback_needed = True
        bad_stage_label = "Prompt 5"
        bad_raw_len = p5_len
        return p5_out, stats, fallback_needed, bad_stage_label, bad_raw_len

    # -------- Prompt 3 --------
    p3_out = tag_cv_in_block_text(p5_out)
    p3_len = len(p3_out.strip())
    stats["p3_raw_len"] = p3_len

    if DEBUG_PIPELINE:
        print("LEN después de Prompt 3:", p3_len)

    if p3_len == 0 or (orig_len - p3_len) > length_diff_threshold:
        fallback_needed = True
        bad_stage_label = "Prompt 3"
        bad_raw_len = p3_len
        return p3_out, stats, fallback_needed, bad_stage_label, bad_raw_len

    # Si llegaste hasta acá, ningún prompt disparó el fallback
    return p3_out, stats, False, "", 0

# =======================================
# ---- CORE: PDF → PDF final usando 1 etapa por bloque ----
def full_pipeline_pdf_pages_to_merged_pdf(
    pages_text: List[str],
    patient_data_list: List[str],
    pages_per_block: int = PAGES_PER_BLOCK,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    tempdir_callback: Optional[Callable[[str], None]] = None,
) -> Tuple[bytes, List[str]]:
    """
    Para cada bloque de páginas:
      1) Une las páginas en texto.
      2) Agrega el marcador ---Sección Intermedia--- al inicio del bloque.
      3) Etapa única (Prompt 2 + Prompt 5 + Prompt 3) con chequeo ultra estricto.
      4) Si algún prompt devuelve salida vacía o mucho más corta:
           - Si hay >1 página en el bloque: reprocesar página por página.
           - Si hay 1 sola página: se usa lo que haya, pero se registra el evento.
      5) Elimina todas las ocurrencias de ---Sección Intermedia---.
      6) texto_final limpio → PDF de bloque (guardado en carpeta temporal).

    Devuelve:
      - merged_pdf_bytes
      - fallback_events: lista de mensajes de advertencia para mostrar en Streamlit.
    """
    num_pages = len(pages_text)
    if num_pages == 0:
        return b"", []

    temp_dir = tempfile.mkdtemp(prefix="anon_blocks_")

    if tempdir_callback is not None:
        tempdir_callback(temp_dir)

    final_block_paths: List[str] = []
    fallback_events: List[str] = []

    try:
        blocks = []
        for start in range(0, num_pages, pages_per_block):
            end = min(start + pages_per_block, num_pages)
            blocks.append((start, end))

        total_blocks = len(blocks)

        for block_idx, (start_idx, end_idx) in enumerate(blocks, start=1):
            block_pages = pages_text[start_idx:end_idx]
            block_text = "\n".join(block_pages).strip()

            # 1) Agregar marcador de nueva sección al inicio del bloque
            block_text_with_marker = add_section_marker(block_text)
            orig_len = len(block_text_with_marker)

            # 2) Procesar el bloque completo (P2 + P5 + P3) con control ultra estricto
            block_result_text, stats, need_page_fallback, bad_stage_label, bad_raw_len = process_block_full(
                block_text_with_marker,
                patient_data_list,
                orig_len=orig_len,
                length_diff_threshold=LENGTH_DIFF_THRESHOLD,
            )

            # Si se detectó problema y el bloque tiene más de una página,
            # reprocesar página por página
            if need_page_fallback and len(block_pages) > 1:
                fallback_events.append(
                    f"Bloque {block_idx} (páginas {start_idx+1}-{end_idx}) "
                    f"reprocesado página por página por salida anómala en {bad_stage_label} "
                    f"(len_salida={bad_raw_len}, len_original={orig_len})."
                )

                page_results: List[str] = []

                for page_text in block_pages:
                    page_text_with_marker = add_section_marker(page_text)
                    page_orig_len = len(page_text_with_marker)

                    page_result_text, page_stats, page_fallback, page_bad_stage, page_bad_len = process_block_full(
                        page_text_with_marker,
                        patient_data_list,
                        orig_len=page_orig_len,
                        length_diff_threshold=LENGTH_DIFF_THRESHOLD,
                    )

                    if page_fallback:
                        # No hay más nivel de fallback (ya es 1 página),
                        # pero lo registramos igual.
                        fallback_events.append(
                            f"  - Página individual en bloque {block_idx} "
                            f"tuvo salida anómala en {page_bad_stage} "
                            f"(len_salida={page_bad_len}, len_original={page_orig_len})."
                        )

                    page_result_text_clean = remove_section_marker(page_result_text)
                    page_results.append(page_result_text_clean)

                block_result_text_clean = "\n".join(page_results).strip()

            else:
                # No hace falta fallback por página (o solo había 1 página)
                if need_page_fallback and len(block_pages) == 1:
                    fallback_events.append(
                        f"Bloque {block_idx} (página {start_idx+1}) tuvo salida anómala en {bad_stage_label} "
                        f"(len_salida={bad_raw_len}, len_original={orig_len}), "
                        f"pero no se pudo dividir más (solo una página)."
                    )
                block_result_text_clean = remove_section_marker(block_result_text)

            # 4) Convertir a PDF
            block_pdf_bytes = text_to_pdf_bytes(block_result_text_clean)

            block_filename = f"block_{block_idx:04d}.pdf"
            block_path = os.path.join(temp_dir, block_filename)
            with open(block_path, "wb") as f:
                f.write(block_pdf_bytes)

            final_block_paths.append(block_path)

            # Liberar memoria de este bloque
            del block_pages
            del block_text
            del block_text_with_marker
            del block_result_text
            del block_result_text_clean
            del block_pdf_bytes

            if progress_callback is not None:
                progress_callback(block_idx, total_blocks)

        merged_pdf_bytes = merge_pdfs(sorted(final_block_paths))
        return merged_pdf_bytes, fallback_events

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

        st.success(f"PDF leído correctamente. Páginas detectadas: {num_pages}")
        st.caption(
            f"Caracteres extraídos (total): {len(full_text)} | "
            f"chunk: {MAX_CHARS_PER_CHUNK} | overlap: {OVERLAP} | "
            f"bloque de páginas: {PAGES_PER_BLOCK}"
        )
        st.text_area(
            "Vista previa del texto (primeras páginas)",
            value=full_text[:2000] + ("..." if len(full_text) > 2000 else ""),
            height=200,
        )

        if st.button("🚀 Ejecutar modelo (P1 + (P2+P5+P3) en 1 etapa estricta)"):
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
            fallback_placeholder = st.empty()

            def progress_cb(current_block: int, total_blocks: int) -> None:
                pct = int(current_block / total_blocks * 100)
                progress_bar.progress(pct)
                status_placeholder.write(
                    f"Procesando bloque {current_block} de {total_blocks} (1 etapa por bloque, modo estricto)..."
                )

            def tempdir_cb(path: str) -> None:
                tempdir_placeholder.caption(f"Carpeta temporal usada: {path}")

            with st.spinner(
                "Procesando bloques en 1 etapa (P2+P5+P3, con reintento página por página si alguna salida es anómala)..."
            ):
                try:
                    final_pdf_bytes, fallback_events = full_pipeline_pdf_pages_to_merged_pdf(
                        pages_text,
                        patient_data_list=patient_data_list,
                        pages_per_block=PAGES_PER_BLOCK,
                        progress_callback=progress_cb,
                        tempdir_callback=tempdir_cb,
                    )
                except Exception as e:
                    st.error(f"Error durante el procesamiento por bloques: {e}")
                    st.stop()

            progress_bar.progress(100)
            status_placeholder.write("Procesamiento completado ✅")

            # Mostrar advertencias si hubo re-procesos por página
            if fallback_events:
                fallback_placeholder.warning("Algunos bloques/páginas se reprocesaron página por página por salidas anómalas:")
                for ev in fallback_events:
                    fallback_placeholder.write(f"- {ev}")

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
                    "¡Listo! Se ejecutó el pipeline completo en 1 etapa por bloque (modo estricto) y se generó el PDF anonimizado ✅"
                )

    else:
        st.info("Subí un PDF para comenzar.")


if __name__ == "__main__":
    main()