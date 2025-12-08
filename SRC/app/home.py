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
import fitz  
import ast   


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
        "use_think_trunc":  True,
    },
    "Qwen 4B (Ollama)": {
        "model_name":       "qwen3:4b",
        "temperature":      0.0,
        "num_ctx":          12000,
        "num_predict":      6000,
        "top_p":            0.95,
        "top_k":            20,
        "repeat_penalty":   1.1,
        "repeat_last_n":    256,
        "presence_penalty": 0.0,
        "use_think_trunc":  True,
    },
}


OLLAMA_ENDPOINT = "http://localhost:11434"


_DEFAULT_CFG = MODEL_CONFIGS["Qwen 8B (Ollama)"]
MODEL_NAME      = _DEFAULT_CFG["model_name"]
TEMPERATURE     = _DEFAULT_CFG["temperature"]

USE_CHUNKING          = True            
MAX_CHARS_PER_CHUNK   = 15000           
OVERLAP               = 0               


SECTIONS_PER_BLOCK    = 1              


NUM_CTX               = _DEFAULT_CFG["num_ctx"]      
NUM_PREDICT           = _DEFAULT_CFG["num_predict"]  


DEBUG_PIPELINE        = True


TOP_P              = _DEFAULT_CFG["top_p"]
TOP_K              = _DEFAULT_CFG["top_k"]
REPEAT_PENALTY     = _DEFAULT_CFG["repeat_penalty"]
REPEAT_LAST_N      = _DEFAULT_CFG["repeat_last_n"]
PRESENCE_PENALTY   = _DEFAULT_CFG["presence_penalty"]
USE_THINK_TRUNC    = _DEFAULT_CFG["use_think_trunc"]


SECTION_HEADER_REGEX = re.compile(
    r"(?m)^---\s*Secci[oó]n\s+(\d+)\s*---\s*$"
)


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


PROMPT5_TEMPLATE = """
Eres un asistente especializado en anonimizar historias clínicas en español.

Tu ÚNICA tarea en este paso es tratar apellidos.

Instrucciones obligatorias:
1) Si en el texto aparece un encabezado "Responsables del registro:" (o una variante muy similar, sin importar mayúsculas/minúsculas), debes conservar el encabezado tal cual.
    Luego debes anonimizar los nombres y apellidos que se encuentren a continuacion. 
    
    Por ejemplo:

    Responsables del registro:
    AE. GARCIA
    LIC. DEL PUERTO

    Pasaria a ser:

    Responsables del registro:
    [CENSURADO]
    [CENSURADO]

2) Si identificas cualquier nombre o apellido en el texto, cambialo por [CENSURADO]. Ejemplos: Juan Perez → [CENSURADO], Rodriguez → [CENSURADO], Dr. Benitez → [CENSURADO].
3) Devuelve ÚNICAMENTE el documento censurado (o el original si no hay cambios), sin explicaciones ni notas adicionales.

Texto a procesar:
{text}
"""


def pdf_bytes_to_pages(pdf_bytes: bytes) -> List[str]:
    """
    Convierte un PDF en memoria a una lista de textos por página.
    Abre el PDF con PyMuPDF, recorre cada página y extrae el texto plano
    con get_text(), devolviendo una lista donde cada elemento corresponde
    al contenido de una página.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []
    for page in doc:
        pages.append(page.get_text().strip())
    doc.close()
    return pages


def build_sectioned_text_from_pages(pages_text: List[str]) -> str:
    """
    Construye un texto único seccionado a partir de las páginas del PDF.

    Une el texto de todas las páginas, lo divide por el patrón 'Fecha:'
    (cada aparición se interpreta como una evolución clínica) y genera
    secciones numeradas del tipo '--- Sección i ---' con el contenido
    correspondiente a cada evolución.
    """
    full_text = "\n".join(pages_text).strip()
    if not full_text:
        return ""

   
    evoluciones = re.split(r'(?=Fecha:)', full_text)

   
    evoluciones_filtradas = [sec.strip() for sec in evoluciones if len(sec.split()) >= 10]

    partes: List[str] = []
    for i, sec in enumerate(evoluciones_filtradas, start=1):
        partes.append(f"--- Sección {i} ---\n{sec.strip()}\n")

    return "\n\n".join(partes).strip()


def ollama_generate(
    model: str,
    prompt: str,
    endpoint: str = OLLAMA_ENDPOINT,
    temperature: float = TEMPERATURE,
    options: Optional[dict] = None,
) -> str:
    """
    Envía un prompt a Ollama y devuelve el texto generado por el modelo.

    Construye el payload con el modelo y los hiperparámetros de decodificación,
    realiza una llamada HTTP al endpoint /api/generate en modo streaming y
    concatena los fragmentos de texto recibidos en un único string de salida.
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

def strip_think_segment(text: str) -> str:
    """
    Elimina cualquier contenido posterior al marcador '/think' en el texto.

    Si encuentra la cadena '/think', devuelve solo la parte anterior a ese
    marcador; si no aparece, devuelve el texto original aplicando strip().
    Se utiliza como filtro para descartar bloques de razonamiento internos.
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

def chunk_text_by_chars(text: str, max_chars: int, overlap: int) -> List[str]:
    """
    Divide un texto largo en fragmentos de hasta max_chars caracteres.
    Recorre el string tomando bloques sucesivos de longitud máxima max_chars
    y, opcionalmente, solapa los fragmentos en overlap caracteres. Devuelve
    una lista con los chunks resultantes para facilitar el procesamiento
    de textos extensos.
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


def text_to_pdf_bytes(
    text: str,
    paper: str = "A4",
    fontname: str = "Courier",
    fontsize: int = 10,
    margin: int = 36,
    line_spacing: float = 1.4,
) -> bytes:
    """
    Convierte un texto plano en un PDF y devuelve sus bytes en memoria.

    Ajusta el texto a líneas y páginas según el tamaño de papel, fuente,
    márgenes y espaciado, crea el documento con PyMuPDF y genera un PDF
    simple y legible a partir del contenido anonimizado.
    """
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

def merge_pdfs(pdf_paths: List[str]) -> bytes:
    """
    Fusiona varios archivos PDF en un solo documento.

    Abre cada PDF indicado en la lista de rutas, inserta todas sus páginas
    en un documento de salida y devuelve los bytes del PDF combinado,
    listo para ser guardado o descargado por el usuario.
    """
    out_doc = fitz.open()
    for path in pdf_paths:
        src = fitz.open(path)
        out_doc.insert_pdf(src)
        src.close()
    merged_bytes = out_doc.tobytes()
    out_doc.close()
    return merged_bytes

def split_text_into_sections(text: str) -> List[Tuple[int, str]]:
    """
    Divide el texto completo en una lista de tuplas (numero_seccion, texto_de_la_seccion).

    Los encabezados esperados son líneas tipo:
        --- Sección 1 ---
        --- Seccion 2 ---

    Si no se encuentra ninguna sección, devuelve una sola "sección" 0 con todo el texto.
    También, si hay texto antes de la primera sección, se agrega como sección -1 (si no está vacío).
    """
    matches = list(SECTION_HEADER_REGEX.finditer(text))
    if not matches:
        
        return [(0, text)]

    sections: List[Tuple[int, str]] = []

  
    first_start = matches[0].start()
    preamble = text[:first_start].strip()
    if preamble:
        sections.append((-1, preamble)) 

    for i, m in enumerate(matches):
        sec_num_str = m.group(1)
        sec_num = int(sec_num_str)
        start = m.start()
        
        if i + 1 < len(matches):
            end = matches[i + 1].start()
        else:
            end = len(text)

        full_section_text = text[start:end].strip("\n")
        sections.append((sec_num, full_section_text))

    return sections

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
    Asocia cada sección con una fecha detectada en su contenido.

    Recorre la lista de secciones (id_sección, texto), extrae en cada una
    la primera fecha con formato dd/mm/aaaa y la convierte a objeto date.
    Devuelve un diccionario {id_sección -> date | None} y una lista con
    todas las fechas válidas encontradas, sirve para filtrar qué secciones
    procesar según un rango temporal.
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

def normalize_punctuation_in_list(seq: List[str]) -> List[str]:
    """
    Limpia la puntuación y espacios en una lista de strings.

    Para cada elemento reemplaza , . : ; por espacios, colapsa espacios
    múltiples y aplica strip(), descartando los elementos vacíos. Se usa
    para dejar más uniforme la lista cruda devuelta por el LLM.
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

def run_prompt1_on_first_pages(
    pages_text: List[str],
    n_pages: int = 2,
) -> str:
    """
    Ejecuta el Prompt 1 sobre las primeras páginas del documento.

    Toma el texto de las primeras n_pages, arma el prompt de encabezado y
    llama al modelo vía ollama_generate, devolviendo la respuesta en bruto
    (como string) para su posterior parseo.
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

def parse_llm_list(text: str) -> List[str]:
    """
    Convierte la salida del LLM en una lista de Python.

    Localiza el primer bloque entre corchetes [...], intenta parsearlo
    como JSON y, si falla, recurre a ast.literal_eval. Devuelve la lista
    con los elementos detectados (nombre, documento, dirección, etc.).
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

def postprocess_patient_data(raw_list: List[str]) -> List[str]:
    """
    Estandariza los datos del paciente extraídos por el modelo.

    A partir de una lista [nombre_completo, documento, dirección], separa
    el nombre en componentes, limpia el documento de signos no alfanuméricos
    y devuelve una lista con las partes del nombre, el documento limpio
    y la dirección original.
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

    Corre el modelo sobre las primeras páginas, parsea la lista devuelta,
    normaliza la puntuación de cada elemento y aplica el postprocesado
    específico. Devuelve la respuesta cruda del LLM, la lista normalizada
    y la lista final procesada para usar en la censura.
    """
    result_prompt_1 = run_prompt1_on_first_pages(pages_text, n_pages=2)


    raw_list = parse_llm_list(result_prompt_1)


    raw_list_clean = normalize_punctuation_in_list(raw_list)

   
    processed_list = postprocess_patient_data(raw_list_clean)

    
    return result_prompt_1, raw_list_clean, processed_list


def censor_block_text(block_text: str, patient_data_list: List[str]) -> str:
    """
    Censura datos del paciente en un bloque de texto usando Prompt 2.

    Construye una lista JSON con los datos del paciente, arma el prompt
    correspondiente y llama al modelo (con soporte opcional de chunking
    por longitud). Aplica maybe_strip_think y devuelve el bloque censurado
    o el original si no hubo cambios.
    """
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


def tag_cv_in_block_text(block_text: str) -> str:
    """
    Detecta y transforma menciones de carga viral en un bloque de texto.

    Aplica Prompt 3 sobre el bloque (con chunking si es necesario) para
    localizar expresiones de carga viral y redondear los valores numéricos
    al millar más cercano, devolviendo el texto resultante procesado.
    """
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


def responsables_block_text(block_text: str) -> str:
    """
    Anonimiza la sección 'Responsables del registro' y nombres asociados en un bloque.

    Aplica el Prompt 5 sobre el texto del bloque (con soporte opcional de chunking)
    para censurar nombres y apellidos de profesionales, en especial bajo el encabezado
    'Responsables del registro'. Utiliza ollama_generate y maybe_strip_think, y
    devuelve el bloque censurado o el original si no hay cambios.
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



def process_block_full(
    block_text: str,
    patient_data_list: List[str],
    warnings_collector: Optional[List[str]] = None,
    section_id: Optional[int] = None,
) -> str:
    """
    Ejecuta la etapa completa de anonimización sobre un bloque de texto.

    Encadena tres prompts en orden:
      1) Prompt 2 (censor_block_text): censura datos del paciente usando patient_data_list.
      2) Prompt 5 (responsables_block_text): anonimiza 'Responsables del registro' y nombres.
      3) Prompt 3 (tag_cv_in_block_text): detecta cargas virales y redondea sus valores.

    Si algún paso devuelve texto vacío, conserva el resultado del paso anterior para no
    perder contenido. Si se está procesando una sección individual (section_id no es None)
    y un paso devuelve salida vacía, se registra un warning indicando que se mantuvo el
    texto de entrada o el texto previo para esa sección.
    """
    if not block_text.strip():
        return ""

    if DEBUG_PIPELINE:
        print("\n===== Nuevo bloque — ETAPA ÚNICA =====")
        print("LEN original bloque:", len(block_text))

   
    censored_text = censor_block_text(block_text, patient_data_list)
    if not censored_text.strip():
        if DEBUG_PIPELINE:
            print("Prompt 2 devolvió vacío, se conserva el texto original del bloque.")
        if warnings_collector is not None and section_id is not None:
            warnings_collector.append(
                f"Sección {section_id}: Prompt 2 devolvió salida vacía; "
                f"se mantuvo el texto original de entrada para esa sección."
            )
        censored_text = block_text

    if DEBUG_PIPELINE:
        print("LEN después de Prompt 2 (censura específica):", len(censored_text))

    
    text_resp = responsables_block_text(censored_text)
    if not text_resp.strip():
        if DEBUG_PIPELINE:
            print("Prompt 5 devolvió vacío, se conserva el resultado de Prompt 2.")
        if warnings_collector is not None and section_id is not None:
            warnings_collector.append(
                f"Sección {section_id}: Prompt 5 devolvió salida vacía; "
                f"se mantuvo el texto previo para esa sección."
            )
        text_resp = censored_text

    if DEBUG_PIPELINE:
        print("LEN después de Prompt 5 (responsables):", len(text_resp))

   
    final_text = tag_cv_in_block_text(text_resp)
    if not final_text.strip():
        if DEBUG_PIPELINE:
            print("Prompt 3 devolvió vacío, se conserva el resultado de Prompt 5.")
        if warnings_collector is not None and section_id is not None:
            warnings_collector.append(
                f"Sección {section_id}: Prompt 3 devolvió salida vacía; "
                f"se mantuvo el texto previo para esa sección."
            )
        final_text = text_resp

    if DEBUG_PIPELINE:
        print("LEN después de Prompt 3 (redondeo CV):", len(final_text))

    return final_text


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
    Ejecuta el pipeline completo de anonimización y genera un PDF final.

    Construye el texto seccionado a partir de las páginas, identifica las secciones
    válidas y sus fechas, y procesa solo aquellas dentro del rango [start_date, end_date]
    (incluyendo siempre la sección 1 si existe). Agrupa las secciones en bloques de hasta
    sections_per_block, aplica process_block_full a cada bloque y controla la diferencia
    de longitud entre texto original y anonimizado: si el recorte supera un umbral,
    reprocesa sección por sección y, en caso extremo, conserva el texto original y añade
    un warning. Cada bloque procesado se convierte a PDF y, al final, todos los PDFs se
    fusionan en un único documento, que se devuelve en bytes junto con la lista de avisos.
    """
    if warnings_collector is None:
        warnings_collector = []

    full_text = build_sectioned_text_from_pages(pages_text)
    if not full_text:
        return b"", warnings_collector

    sections = split_text_into_sections(full_text)
    section_ids = [sid for sid, _ in sections if sid >= 0]


    section_dates, all_dates = map_section_ids_to_dates(sections)

 
    has_any_date = len(all_dates) > 0

    if not has_any_date or start_date is None or end_date is None:
      
        sections_to_process_ids = set(section_ids)
    else:
     
        sections_to_process_ids = {
            sid
            for sid in section_ids
            if (section_dates.get(sid) is not None
                and start_date <= section_dates[sid] <= end_date)
        }

 
    if 1 in section_ids:
        sections_to_process_ids.add(1)

 
    sections_to_use: List[Tuple[int, str]] = []
    for sid, sec_text in sections:
        if sid >= 0 and sid in sections_to_process_ids:
            txt = sec_text.strip()
            if txt:
                sections_to_use.append((sid, txt))

   
    if not sections_to_use:
        return b"", warnings_collector


    chunks: List[List[Tuple[int, str]]] = []
    buffer: List[Tuple[int, str]] = []

    for sid, sec_text in sections_to_use:
        buffer.append((sid, sec_text))
        if len(buffer) >= sections_per_block:
            chunks.append(buffer)
            buffer = []


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
           
            block_text = "\n".join(sec_text for _, sec_text in section_block).strip()
            if DEBUG_PIPELINE:
                print(f"\n### Procesando bloque de secciones (chunk #{idx}) ###")
                print("LEN chunk original:", len(block_text))

      
            block_result_text = process_block_full(
                block_text,
                patient_data_list,
                warnings_collector=None,
                section_id=None,
            )

           
            len_orig_block = len(block_text)
            len_res_block  = len(block_result_text)
            diff_block     = len_orig_block - len_res_block

            if DEBUG_PIPELINE:
                print(
                    f"[BLOQUE #{idx}] len_orig={len_orig_block}, "
                    f"len_res={len_res_block}, diff={diff_block}"
                )


            if diff_block > 150:
                if DEBUG_PIPELINE:
                    print(
                        f"[BLOQUE #{idx}] Diferencia > 150 caracteres. "
                        "Reprocesando sección por sección."
                    )

                per_section_outputs: List[str] = []
                for sid, sec_text in section_block:
                    sec_orig_len = len(sec_text)
         
                    sec_result = process_block_full(
                        sec_text,
                        patient_data_list,
                        warnings_collector=warnings_collector,
                        section_id=sid,
                    )
                    sec_result_stripped = sec_result.strip()

                    if not sec_result_stripped:
          
                        diff_sec = sec_orig_len
                    else:
                        diff_sec = sec_orig_len - len(sec_result_stripped)

                    if DEBUG_PIPELINE:
                        print(
                            f"  - Sección {sid}: len_orig={sec_orig_len}, "
                            f"len_res={len(sec_result_stripped)}, diff={diff_sec}"
                        )

     
                    if diff_sec > 250:
                        msg = (
                            f"Sección {sid}: no se pudo anonimizar de forma segura "
                            f"(diferencia de longitud {diff_sec}). "
                            "Se dejó el texto original."
                        )
                        warnings_collector.append(msg)
                        per_section_outputs.append(sec_text)
                    else:
                 
                        if sec_result_stripped:
                            per_section_outputs.append(sec_result_stripped)
                        else:
         
                            per_section_outputs.append(sec_text)

       
                block_result_text = "\n".join(per_section_outputs).strip()

      
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

def main():
    global MODEL_NAME, TEMPERATURE, NUM_CTX, NUM_PREDICT
    global TOP_P, TOP_K, REPEAT_PENALTY, REPEAT_LAST_N, PRESENCE_PENALTY, USE_THINK_TRUNC

    st.set_page_config(page_title="Anonimización de PDFs", layout="centered")
    st.title("Herramienta de anonimización de documentos")


    model_label = st.selectbox(
        "Elegir modelo",
        list(MODEL_CONFIGS.keys()),
        index=0,
    )
    cfg = MODEL_CONFIGS[model_label]


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
        "Modelo seleccionado: **{label}**".format(
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

    uploaded = st.file_uploader("Cargar PDF", type=["pdf"])

    if uploaded is not None:
        pdf_bytes = uploaded.read()
        with st.spinner("Extrayendo texto del PDF..."):
            try:
                pages_text = pdf_bytes_to_pages(pdf_bytes)
            except Exception as e:
                st.error(f"Error al leer el PDF: {e}")
                st.stop()

        num_pages = len(pages_text)


        sectioned_text = build_sectioned_text_from_pages(pages_text)


        sections = split_text_into_sections(sectioned_text)
        section_ids = [sid for sid, _ in sections if sid >= 0]

        section_dates_map, all_dates = map_section_ids_to_dates(sections)

        st.success(f"PDF leído correctamente. Páginas detectadas: {num_pages}")
        st.caption(
            f"Caracteres extraídos: {len(sectioned_text)} | "
            f"Secciones por bloque: {SECTIONS_PER_BLOCK}"
        )
        st.caption(
            f"Secciones detectadas: {len(section_ids)}. "
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
            )

        st.text_area(
            "Vista previa del documento",
            value=sectioned_text[:2000] + ("..." if len(sectioned_text) > 2000 else ""),
            height=200,
        )

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
            "Cargar historia clínica completa. "
            "Se procesarán todas las secciones cuya fecha esté dentro del rango indicado. "
        )

        if st.button("Procesar PDF"):
 
            with st.spinner("Extrayendo datos del encabezado..."):
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

            st.success("Prompt 1 completado.")
            st.write("Datos detectados del paciente:", patient_data_list)

            progress_bar = st.progress(0)
            status_placeholder = st.empty()
            tempdir_placeholder = st.empty()

            def progress_cb(current_block: int, total_blocks: int) -> None:
                pct = int(current_block / total_blocks * 100)
                progress_bar.progress(pct)
                status_placeholder.write(
                    f"Procesando bloque {current_block} de {total_blocks} "
                    f"(Filtrado por rango de fechas)..."
                )

            def tempdir_cb(path: str) -> None:
                tempdir_placeholder.caption(f"Carpeta temporal: {path}")

            with st.spinner(
                "Procesando bloques..."
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
            status_placeholder.write("Procesamiento completado")

          
            if warnings:
                st.warning(
                    "Algunas secciones no se pudieron anonimizar de forma segura o "
                    "presentaron salidas vacías de los prompts, por lo que se dejó "
                    "su versión original o se mantuvo el texto previo:\n\n" +
                    "\n".join(f"- {msg}" for msg in warnings)
                )

            if not final_pdf_bytes:
                st.warning(
                    "La salida está vacía. Revisá el PDF original o el rango de fechas indicado."
                )
            else:
                st.subheader("Descarga")
                st.download_button(
                    "Descargar PDF anonimizado",
                    data=io.BytesIO(final_pdf_bytes),
                    file_name="salida_ollama_anonimizada.pdf",
                    mime="application/pdf",
                )
                st.success(
                    "¡Listo! Se ejecutó el pipeline completo filtrando por rango de fechas y se generó el PDF anonimizado"
                )

    else:
        st.info("Subí un PDF para comenzar.")


if __name__ == "__main__":
    main()