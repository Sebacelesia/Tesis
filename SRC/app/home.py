# app.py — versión v2 con procesamiento por bloques y PDFs intermedios
import io
import os
import json
import shutil
import tempfile
import textwrap
import re           # para regex (carga viral, etc.)
import random       # para el ±50%
from typing import Optional, List

import requests
import streamlit as st

# ====== REGEX PARA EXTRAER DATOS (nombre, números, dirección) ======

regex_numeros = r"""
    (?<!\d)
    (?: 
        # ---- CÉDULAS URUGUAY ----
        \d{1,2}\.?\d{3}\.?\d{3}[-/]\d       # 1.234.567-8 / 1234567-8 / 1.234.567/8
        |
        \d{7,8}                             # 7 u 8 dígitos seguidos (cédula sin guion)
        |

        # ---- TELÉFONOS ----
        \+?\d[\d ]{6,}                      # internacionales o secuencias largas con +
    )
    (?!\d)
"""

# REGEX para DIRECCIÓN (solo si está declarada)
regex_direccion_sin_tilde = r"""
    (?i)                       # case insensitive
    direccion\s*:\s*           # Direccion: (con o sin espacio)
   (.+)                        # capturar lo que sigue
"""
regex_direccion_con_tilde = r"""
    (?i)                       # case insensitive
    dirección\s*:\s*           # Dirección: (con o sin espacio)
   (.+)                        # capturar lo que sigue
"""


def extraer_datos(texto: str) -> str:
    """
    Extrae:
      - Nombre a partir de una línea que empieza con 'Nombre:'
      - Números relevantes (teléfonos, cédulas, etc.)
      - Dirección si aparece como 'Direccion:' o 'Dirección:'

    Devuelve un string con el formato:
    "Juan, Perez, Herrera, 12345678, +598 99010203, Av. Italia 3333"

    Si no encuentra nada, devuelve "".
    """
    if not texto:
        return ""

    # 1) Números (teléfonos + cédulas + internacionales)
    numeros = re.findall(regex_numeros, texto, flags=re.VERBOSE)
    numeros = [n.strip() for n in numeros]
    numeros = list(dict.fromkeys(numeros))   # quitar duplicados

    # 2) Dirección SOLO si aparece como "Dirección:" o "Direccion:"
    match_dir = re.search(regex_direccion_con_tilde, texto, flags=re.VERBOSE)
    direccion = match_dir.group(1).strip() if match_dir else None

    if direccion is None:
        match_dir = re.search(regex_direccion_sin_tilde, texto, flags=re.VERBOSE)
        direccion = match_dir.group(1).strip() if match_dir else None

    # 3) Nombre
    match = re.search(r'Nombre:\s*(.+)', texto, flags=re.IGNORECASE)
    if not match:
        # No hay nombre, armamos igual con lo que tengamos
        elementos: List[str] = []
        if numeros:
            elementos.extend(numeros)
        if direccion:
            elementos.append(direccion)
        return '"' + ", ".join(elementos) + '"' if elementos else ""

    nombre_raw = match.group(1).strip()

    # Normalizar espacios y quitar signos que estorban
    nombre_raw = re.sub(r'[\.,]', ' ', nombre_raw)  # reemplazar comas y puntos por espacios
    nombre_raw = re.sub(r'\s+', ' ', nombre_raw)    # colapsar espacios múltiples

    # Caso especial: "Apellido(s) ..., Nombre"
    if ',' in match.group(1):
        partes = [p.strip() for p in match.group(1).replace('.', '').split(',')]
        if len(partes) >= 2:
            apellidos = partes[0]
            nombres = partes[1]
            nombre_raw = nombres + " " + apellidos

    # Separar por espacios y formatear con comas
    palabras = [p for p in nombre_raw.split() if p]
    nombre_formateado = ", ".join(palabras)  # "Juan, Perez, Herrera"

    elementos: List[str] = []
    if nombre_formateado:
        elementos.append(nombre_formateado)
    if numeros:
        elementos.extend(numeros)
    if direccion:
        elementos.append(direccion)

    if not elementos:
        return ""

    resultado = '"' + ", ".join(elementos) + '"'
    return resultado


# ====== PARÁMETROS FIJOS ======
OLLAMA_ENDPOINT = "http://localhost:11434"
MODEL_NAME      = "qwen3:8b"
TEMPERATURE     = 0.2

USE_CHUNKING          = True            # si el texto supera MAX_CHARS_PER_CHUNK, se parte
MAX_CHARS_PER_CHUNK   = 15000          # caracteres por chunk de texto
OVERLAP               = 10             # solapamiento entre chunks (en caracteres)

# Procesar de a N páginas de PDF por bloque lógico
PAGES_PER_BLOCK       = 10             # <-- controlás "cada 10 páginas"

# Importante para evitar cortes por contexto/salida en Ollama:
NUM_CTX               = 16384          # contexto (tokens) del modelo en Ollama
NUM_PREDICT           = 9000           # tokens de salida máximos


# =======================================
# ==== REGLA DE CARGA VIRAL (regex + ±50%) ====
# Formatos buscados:
#   - CV: valor
#   - Carga Viral: valor
#   - Carga viral: valor

CV_PATTERN = re.compile(
    r"\b(CV|Carga\s+Viral)\s*:\s*([-+]?\d[\d\.,]*)",
    flags=re.IGNORECASE,
)


def _parse_number_preserving_sign(num_str: str) -> float:
    """
    Convierte un string numérico con posibles separadores (., ,) a float,
    preservando el signo.
    """
    s = num_str.strip()
    if not s:
        return 0.0

    sign = -1.0 if s.startswith("-") else 1.0

    # quitar signo explícito si está
    if s[0] in "+-":
        s = s[1:].strip()

    # Heurística simple para coma/punto
    if "," in s and "." in s:
        # asumo puntos como miles, coma como decimal
        s_clean = s.replace(".", "")
        s_clean = s_clean.replace(",", ".")
    elif "," in s:
        # solo coma -> decimal
        s_clean = s.replace(",", ".")
    elif s.count(".") > 1:
        # muchos puntos -> todos como miles
        s_clean = s.replace(".", "")
    else:
        s_clean = s

    # dejar solo dígitos y punto decimal
    s_clean = re.sub(r"[^0-9.]", "", s_clean)
    if not s_clean:
        return 0.0

    try:
        val = float(s_clean)
    except ValueError:
        return 0.0

    return sign * val


def _format_number_like_original(original: str, value: float) -> str:
    """
    Intenta formatear 'value' con un estilo similar al de 'original':
    - respeta si usaba coma o punto como separador decimal
    - respeta cantidad de decimales si los hay
    - aplica signo según el valor final
    No reintroduce separadores de miles para simplificar.
    """
    s = original.strip()
    if not s:
        return str(int(round(value)))

    # ¿había decimales y con qué separador?
    if "," in s:
        dec_sep = ","
        parts = s.split(",")
        decs = len(parts[1]) if len(parts) > 1 else 0
    elif "." in s:
        dec_sep = "."
        parts = s.split(".")
        decs = len(parts[1]) if len(parts) > 1 else 0
    else:
        dec_sep = None
        decs = 0

    val = float(value)
    is_neg = val < 0
    val_abs = abs(val)

    if dec_sep is None or decs == 0:
        formatted = str(int(round(val_abs)))
    else:
        formatted = f"{val_abs:.{decs}f}"
        if dec_sep == ",":
            formatted = formatted.replace(".", ",")

    if is_neg:
        formatted = "-" + formatted

    return formatted


def perturb_cv_value(num_str: str) -> str:
    """
    Toma el string del valor numérico y lo modifica ±50%,
    manteniendo el signo original.
    """
    original_val = _parse_number_preserving_sign(num_str)

    # si no pudimos parsear nada, devolvemos tal cual
    if original_val == 0 and re.sub(r"[^0-9]", "", num_str) == "":
        return num_str

    # factor aleatorio entre 0.5 y 1.5 (±50%)
    factor = random.uniform(0.5, 1.5)
    new_val = original_val * factor

    # devolvemos string con formato similar al original
    return _format_number_like_original(num_str, new_val)


def perturb_cv_in_text(text: str) -> str:
    """
    Busca patrones de carga viral en el texto:
      - CV: valor
      - Carga Viral: valor
      - Carga viral: valor
    y reemplaza SOLO el valor por una versión perturbada ±50%.
    """

    def _repl(match: re.Match) -> str:
        label = match.group(1)    # "CV" o "Carga Viral"
        num_str = match.group(2)  # el valor como string

        new_num_str = perturb_cv_value(num_str)

        # reconstruimos el patrón conservando el label tal como vino
        return f"{label}: {new_num_str}"

    return CV_PATTERN.sub(_repl, text)


# =======================================
# ---- PDF → lista de páginas (PyMuPDF) ----
def pdf_bytes_to_pages(pdf_bytes: bytes) -> List[str]:
    """
    Devuelve una lista con el texto de cada página del PDF.
    pages_text[0] -> página 1
    pages_text[1] -> página 2
    ...
    """
    import fitz  # PyMuPDF
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []
    for page in doc:
        pages.append(page.get_text().strip())
    doc.close()
    return pages


def extraer_datos_desde_paginas(pages_text: List[str]) -> str:
    """
    Usa las primeras 1–2 páginas para extraer nombre, números y dirección,
    y devuelve el string resultado de extraer_datos().
    """
    if not pages_text:
        return ""

    # Tomar primeras 2 páginas si existen, sino todas las que haya
    subset = pages_text[:2]
    full_text = "\n".join(subset).strip()

    resultado = extraer_datos(full_text)
    return resultado


def build_prompt(resultado: str, text: str) -> str:
    """
    Construye el prompt para Ollama, incluyendo la lista de palabras/frases
    que queremos censurar explícitamente.
    """
    return f"""Eres un asistente especializado en anonimizar historias clínicas en español.

        INSTRUCCIONES OBLIGATORIAS
        1) Siempre que aparezcan las siguientes palabras o frases en el documento debes cambiarlas por [CENSURADO]. Estas son: {resultado}
        2) Sustituye SOLO datos personales por estos placeholders exactos:
        - Nombres y apellidos de personas de cualquier origen y en cualquier parte del documento (pacientes, familiares, médicos) → [CENSURADO]
        - Teléfonos (cualquier formato, nacional o internacional) → [CENSURADO]
        - Cédulas de identidad / documentos → [CENSURADO]
        - Direcciones postales/domicilios (calle/avenida + número, esquinas, apto, barrio) → [CENSURADO]
        3) Conserva TODO lo demás sin cambios: síntomas, diagnósticos, dosis, resultados, unidades, abreviaturas, signos de puntuación, mayúsculas/minúsculas.
        4) Si ya hay placeholders ([NOMBRE], [TELEFONO], [CI], [DIRECCIÓN], [CENSURADO]), NO los modifiques.
        5) Títulos y roles: conserva el título y reemplaza solo el nombre. Ej.: “Dr. [CENSURADO]”, “Lic. [CENSURADO]”.
        6) Teléfonos: reemplaza secuencias de 7+ dígitos o con separadores (+598, -, espacios, paréntesis).
        7) Direcciones: incluye referencias claras de domicilio (calle/esquina/número/apto/barrio).
        8) No inventes datos, no agregues comentarios, no cambies el formato. Respeta saltos de línea y espacios originales.
        9) Devuelve ÚNICAMENTE el texto anonimizado, sin explicaciones ni encabezados.
        10) NUNCA anonimices lo que aparece como Ciudad, Sexo o Edad. Es importante conservar esta información.

        Texto a anonimizar:
        {text}"""


# ---- Llamada a Ollama (streaming) ----
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

    text_parts = []
    for line in resp.iter_lines():
        if not line:
            continue
        chunk = json.loads(line)
        part = chunk.get("response", "")
        if part:
            text_parts.append(part)
    return "".join(text_parts).strip()


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
    fontname: str = "Courier",   # monoespaciada para envolver simple
    fontsize: int = 10,
    margin: int = 36,            # 0.5" en puntos
    line_spacing: float = 1.4,
) -> bytes:
    """
    Genera un PDF simple en memoria con PyMuPDF, multi-página.
    Si el texto está vacío, genera una página en blanco para evitar
    'cannot save with zero pages'.
    """
    import fitz
    doc = fitz.open()
    # Tamaños: A4 o Letter
    if paper.upper() == "A4":
        width, height = 595, 842
    else:
        width, height = 612, 792

    # 👇 Si no hay texto, devolvemos un PDF con UNA página en blanco
    if not text or not text.strip():
        doc.new_page(width=width, height=height)
        pdf_bytes = doc.tobytes()
        doc.close()
        return pdf_bytes

    usable_w = width - 2 * margin
    usable_h = height - 2 * margin

    # Estimación de ancho por carácter para monoespaciada
    char_w = fontsize * 0.6
    max_chars_per_line = max(20, int(usable_w / char_w))

    line_h = int(fontsize * line_spacing)
    max_lines_per_page = max(10, int(usable_h / line_h))

    # Envolver respetando saltos de párrafo
    all_lines: List[str] = []
    for para in text.splitlines():
        if not para.strip():
            all_lines.append("")  # línea en blanco
            continue
        wrapped = textwrap.wrap(para, width=max_chars_per_line, break_long_words=False)
        if not wrapped:
            all_lines.append("")
        else:
            all_lines.extend(wrapped)

    # Escribir líneas en páginas
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
    """
    Une una lista de PDFs (en orden) y devuelve los bytes del PDF final.
    """
    import fitz
    out_doc = fitz.open()
    for path in pdf_paths:
        src = fitz.open(path)
        out_doc.insert_pdf(src)
        src.close()
    merged_bytes = out_doc.tobytes()
    out_doc.close()
    return merged_bytes


# ---- Procesar UN bloque de texto (texto plano) → texto anonimizado ----
def anonymize_block_text(block_text: str, resultado: str) -> str:
    """
    Recibe el texto de un bloque de páginas, lo trocea si hace falta,
    llama al modelo y devuelve el texto anonimizado de TODO el bloque.
    Luego aplica la perturbación de carga viral sobre los patrones CV/Carga Viral.
    """
    if not block_text.strip():
        return ""

    # Si es muy grande, usamos chunking por caracteres
    if USE_CHUNKING and len(block_text) > MAX_CHARS_PER_CHUNK:
        chunks = chunk_text_by_chars(
            block_text,
            max_chars=MAX_CHARS_PER_CHUNK,
            overlap=OVERLAP,
        )
        block_out_parts: List[str] = []
        for ch in chunks:
            prompt = build_prompt(resultado, ch)
            out = ollama_generate(
                model=MODEL_NAME,
                prompt=prompt,
                endpoint=OLLAMA_ENDPOINT,
                temperature=TEMPERATURE,
            )
            block_out_parts.append(out.strip())
        block_result = "\n\n".join([p for p in block_out_parts if p]).strip()
    else:
        # Bloque suficientemente chico: va en una sola llamada
        prompt = build_prompt(resultado, block_text)
        block_result = ollama_generate(
            model=MODEL_NAME,
            prompt=prompt,
            endpoint=OLLAMA_ENDPOINT,
            temperature=TEMPERATURE,
        ).strip()

    # 🔧 APLICAR REGLA DE CARGA VIRAL (regex + ±50%)
    block_result = perturb_cv_in_text(block_result)

    return block_result


# ---- CORE: PDF (páginas en texto) → PDF final anonimizado usando carpeta temporal ----
def anonymize_pdf_pages_to_merged_pdf(
    pages_text: List[str],
    resultado: str,
    pages_per_block: int = PAGES_PER_BLOCK,
) -> bytes:
    """
    Recibe la lista de texto por página del PDF original y:
      1) arma bloques de páginas,
      2) anonimiza bloque por bloque usando 'resultado' en el prompt,
      3) genera un PDF por cada bloque en una carpeta temporal,
      4) une todos los PDFs en uno solo,
      5) borra la carpeta temporal,
      6) devuelve los bytes del PDF final.

    No guarda todos los resultados en RAM, solo procesa un bloque a la vez.
    """
    num_pages = len(pages_text)
    if num_pages == 0:
        return b""

    # Carpeta temporal para PDFs intermedios
    temp_dir = tempfile.mkdtemp(prefix="anon_blocks_")
    block_pdf_paths: List[str] = []

    # 👀 Mostrar en la UI dónde se está creando la carpeta temporal
    st.caption(f"Carpeta temporal para PDFs intermedios: {temp_dir}")

    try:
        # 1) Armar bloques de páginas
        blocks = []  # lista de tuplas: (start_page_idx, end_page_idx)
        for start in range(0, num_pages, pages_per_block):
            end = min(start + pages_per_block, num_pages)
            blocks.append((start, end))

        total_blocks = len(blocks)

        # UI de progreso en Streamlit
        progress_bar = st.progress(0)
        status_placeholder = st.empty()

        # 2) Procesar bloque por bloque
        for block_idx, (start_idx, end_idx) in enumerate(blocks, start=1):
            # Mostrar en la UI qué bloque se está procesando
            status_placeholder.text(
                f"Anonimizando bloque {block_idx}/{total_blocks} "
                f"(páginas {start_idx + 1}–{end_idx})..."
            )

            block_pages = pages_text[start_idx:end_idx]
            block_text = "\n".join(block_pages).strip()

            # Anonimizar texto del bloque (en memoria solo este bloque)
            block_result_text = anonymize_block_text(block_text, resultado)

            # Convertir el resultado del bloque a PDF
            block_pdf_bytes = text_to_pdf_bytes(block_result_text)

            # Guardar PDF del bloque en carpeta temporal
            block_filename = f"block_{block_idx:04d}.pdf"
            block_path = os.path.join(temp_dir, block_filename)
            with open(block_path, "wb") as f:
                f.write(block_pdf_bytes)

            block_pdf_paths.append(block_path)

            # Actualizar barra de progreso
            progress_bar.progress(block_idx / total_blocks)

            # Liberar referencias grandes explícitamente
            del block_pages
            del block_text
            del block_result_text
            del block_pdf_bytes

        # Mensaje final
        status_placeholder.text("✅ Anonimización completada para todos los bloques.")

        # 3) Unir todos los PDFs intermedios en uno solo
        merged_pdf_bytes = merge_pdfs(sorted(block_pdf_paths))
        return merged_pdf_bytes

    finally:
        # 4) Limpiar carpeta temporal (PDFs intermedios)
        try:
            shutil.rmtree(temp_dir)
        except Exception:
            # Si falla la limpieza no rompemos el flujo principal
            pass


# ---- UI Streamlit ----
def main():
    st.set_page_config(page_title="PDF → Texto → Ollama (Qwen 8B)", layout="centered")
    st.title("📄 PDF → 🧠 Qwen 8B (Ollama)")

    uploaded = st.file_uploader("Subí un PDF", type=["pdf"])

    if uploaded is not None:
        pdf_bytes = uploaded.read()
        with st.spinner("Extrayendo texto del PDF..."):
            try:
                pages_text = pdf_bytes_to_pages(pdf_bytes)  # lista de texto por página
                resultado = extraer_datos_desde_paginas(pages_text)
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

        if st.button("🚀 Ejecutar anonimización completa"):
            with st.spinner("Procesando bloques y generando PDF anonimizado..."):
                try:
                    final_pdf_bytes = anonymize_pdf_pages_to_merged_pdf(
                        pages_text,
                        resultado=resultado,
                        pages_per_block=PAGES_PER_BLOCK,
                    )
                except Exception as e:
                    st.error(f"Error durante la anonimización: {e}")
                    st.stop()

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
                st.success("¡Listo! Se procesaron todos los bloques y se generó el PDF anonimizado ✅")
    else:
        st.info("Subí un PDF para comenzar.")


if __name__ == "__main__":
    main()
