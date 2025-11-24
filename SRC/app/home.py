# app.py — versión v2 con procesamiento por bloques y PDFs intermedios
import io
import os
import json
import shutil
import tempfile
import textwrap
import re           # <-- NUEVO: para buscar tags
import random       # <-- NUEVO: para el ±50%
from typing import Optional, List

import requests
import streamlit as st

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

DEFAULT_TEMPLATE = (
"""Eres un asistente especializado en anonimizar historias clínicas en español.

        INSTRUCCIONES OBLIGATORIAS
        1) Sustituye SOLO datos personales por estos placeholders exactos:
        - Nombres y apellidos de personas de cualquier origen y en cualquier parte del documento (pacientes, familiares, médicos) → [CENSURADO]
        - Teléfonos (cualquier formato, nacional o internacional) → [CENSURADO]
        - Cédulas de identidad / documentos → [CENSURADO]
        - Direcciones postales/domicilios (calle/avenida + número, esquinas, apto, barrio) → [CENSURADO]
        2) Conserva TODO lo demás sin cambios: síntomas, diagnósticos, dosis, resultados, unidades, abreviaturas, signos de puntuación, mayúsculas/minúsculas.
        3) Si ya hay placeholders ([NOMBRE], [TELEFONO], [CI], [DIRECCIÓN], [CENSURADO]), NO los modifiques.
        4) Títulos y roles: conserva el título y reemplaza solo el nombre. Ej.: “Dr. [CENSURADO]”, “Lic. [CENSURADO]”.
        5) Teléfonos: reemplaza secuencias de 7+ dígitos o con separadores (+598, -, espacios, paréntesis).
        6) Direcciones: incluye referencias claras de domicilio (calle/esquina/número/apto/barrio).
        7) No inventes datos, no agregues comentarios, no cambies el formato. Respeta saltos de línea y espacios originales.
        8) Devuelve ÚNICAMENTE el texto anonimizado, sin explicaciones ni encabezados.
        9) NUNCA anonimices lo que aparece como Ciudad, Sexo o Edad. Es importante conservar esta información.
        10) Cuando encuentres una mención de carga viral (por ejemplo "carga viral: 120000" o "CV: 120000" o "cv: 120000"),
        NO modifiques el número tú mismo. En su lugar, reemplaza SOLO el número por una marca con este formato EXACTO:
           - "cv: [[CV_TAG: 120000]]"
           - "CV: [[CV_TAG: -34000]]"
           - "Carga viral: [[CV_TAG: 45000]]"
        Es muy importante que:
           - Conserves el texto que rodea al número (por ejemplo "cv:", "Carga viral:", "copias/ml", etc.).
           - Dentro de [[CV_TAG: ...]] coloques el valor numérico original tal como aparece (con signo si es negativo).
        11) No uses la marca [[CV_TAG: ...]] para otros datos que no sean carga viral.

        Texto a anonimizar:
        {text}"""
)

# =======================================
# ==== TOOL CASERA: perturbar valores marcados con [[CV_TAG: ...]] ±50% ====

def _parse_number_preserving_sign(num_str: str) -> float:
    """
    Convierte un string numérico con posibles separadores (., ,) a float,
    preservando el signo. No intenta ser perfecto, pero sirve para carga viral.
    """
    s = num_str.strip()
    if not s:
        return 0.0

    sign = -1.0 if s.startswith("-") else 1.0
    # quitar signo explícito
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

    # quitar cualquier cosa que no sea dígito o punto
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
    - respeta signo (ya viene aplicado en value)
    - respeta si usaba coma o punto como separador decimal
    - respeta cantidad de decimales si los hay
    No reintroduce separadores de miles para simplificar.
    """
    s = original.strip()
    if not s:
        # si original está vacío, devolvemos entero simple
        return str(int(round(value)))

    # detectar si había signo explícito
    had_plus = s.startswith("+")
    had_minus = s.startswith("-")

    # parte sin signo
    if s[0] in "+-":
        s_body = s[1:].strip()
    else:
        s_body = s

    # determinar separador decimal y decimales
    if "," in s_body:
        dec_sep = ","
        parts = s_body.split(",")
        decs = len(parts[1]) if len(parts) > 1 else 0
    elif "." in s_body:
        dec_sep = "."
        parts = s_body.split(".")
        decs = len(parts[1]) if len(parts) > 1 else 0
    else:
        dec_sep = None
        decs = 0

    val = float(value)
    is_neg = val < 0
    val_abs = abs(val)

    if dec_sep is None or decs == 0:
        # entero
        formatted = str(int(round(val_abs)))
    else:
        # decimal con misma cantidad de decimales
        formatted = f"{val_abs:.{decs}f}"
        if dec_sep == ",":
            formatted = formatted.replace(".", ",")

    # volver a aplicar signo
    if is_neg:
        formatted = "-" + formatted
    elif had_plus:
        formatted = "+" + formatted

    return formatted


def perturb_cv_tags(text: str) -> str:
    """
    Busca marcas [[CV_TAG: valor]] en el texto y sustituye cada valor
    por una versión perturbada ±50%, manteniendo el signo.
    Cada aparición se perturba de forma independiente.
    """

    # Patrón para [[CV_TAG: 12345]] (con espacios opcionales)
    pattern = re.compile(
        r"\[\[\s*CV_TAG\s*:\s*([-+]?\d[\d\.,]*)\s*\]\]",
        flags=re.IGNORECASE,
    )

    def _repl(match: re.Match) -> str:
        num_str = match.group(1)

        original_val = _parse_number_preserving_sign(num_str)

        # si no pudimos parsear nada, dejamos el tag tal cual
        if original_val == 0 and re.sub(r"[^0-9]", "", num_str) == "":
            return match.group(0)

        # factor aleatorio entre 0.5 y 1.5 (±50%), positivo
        factor = random.uniform(0.5, 1.5)
        new_val = original_val * factor  # mantiene signo

        # formatear con estilo similar
        new_num_str = _format_number_like_original(num_str, new_val)

        # devolvemos solo el número, sin el tag
        return new_num_str

    return pattern.sub(_repl, text)

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
    """
    import fitz
    doc = fitz.open()
    # Tamaños: A4 o Letter
    if paper.upper() == "A4":
        width, height = 595, 842
    else:
        width, height = 612, 792

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
def anonymize_block_text(block_text: str) -> str:
    """
    Recibe el texto de un bloque de páginas, lo trocea si hace falta,
    llama al modelo y devuelve el texto anonimizado de TODO el bloque.
    Luego aplica la perturbación de carga viral sobre los tags [[CV_TAG: ...]].
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
            prompt = (
                DEFAULT_TEMPLATE.replace("{text}", ch)
                if "{text}" in DEFAULT_TEMPLATE
                else f"{DEFAULT_TEMPLATE.strip()}\n\n{ch}"
            )
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
        prompt = (
            DEFAULT_TEMPLATE.replace("{text}", block_text)
            if "{text}" in DEFAULT_TEMPLATE
            else f"{DEFAULT_TEMPLATE.strip()}\n\n{block_text}"
        )
        block_result = ollama_generate(
            model=MODEL_NAME,
            prompt=prompt,
            endpoint=OLLAMA_ENDPOINT,
            temperature=TEMPERATURE,
        ).strip()

    # 🔧 APLICAR TOOL CASERA SOBRE TAGS DE CARGA VIRAL
    block_result = perturb_cv_tags(block_result)

    return block_result

# ---- CORE: PDF (páginas en texto) → PDF final anonimizado usando carpeta temporal ----
def anonymize_pdf_pages_to_merged_pdf(
    pages_text: List[str],
    pages_per_block: int = PAGES_PER_BLOCK,
) -> bytes:
    """
    Recibe la lista de texto por página del PDF original y:
      1) arma bloques de páginas,
      2) anonimiza bloque por bloque,
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

    try:
        # 1) Armar bloques de páginas
        blocks = []  # lista de tuplas: (start_page_idx, end_page_idx)
        for start in range(0, num_pages, pages_per_block):
            end = min(start + pages_per_block, num_pages)
            blocks.append((start, end))

        # 2) Procesar bloque por bloque
        for block_idx, (start_idx, end_idx) in enumerate(blocks, start=1):
            block_pages = pages_text[start_idx:end_idx]
            block_text = "\n".join(block_pages).strip()

            # Anonimizar texto del bloque (en memoria solo este bloque)
            block_result_text = anonymize_block_text(block_text)

            # Convertir el resultado del bloque a PDF
            block_pdf_bytes = text_to_pdf_bytes(block_result_text)

            # Guardar PDF del bloque en carpeta temporal
            block_filename = f"block_{block_idx:04d}.pdf"
            block_path = os.path.join(temp_dir, block_filename)
            with open(block_path, "wb") as f:
                f.write(block_pdf_bytes)

            block_pdf_paths.append(block_path)

            # Liberar referencias grandes explícitamente
            del block_pages
            del block_text
            del block_result_text
            del block_pdf_bytes

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
