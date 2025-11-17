# app.py — parámetros fijos en código
import io
import json
import requests
import streamlit as st
from typing import Optional, List
import textwrap

# ====== PARÁMETROS FIJOS ======
OLLAMA_ENDPOINT = "http://localhost:11434"
MODEL_NAME      = "qwen3:8b"
TEMPERATURE     = 0.2

USE_CHUNKING          = True            # si el texto supera MAX_CHARS_PER_CHUNK, se parte
MAX_CHARS_PER_CHUNK   = 15000          # caracteres por chunk de texto
OVERLAP               = 10             # solapamiento entre chunks (en caracteres)

# Procesar de a N páginas de PDF por bloque lógico
PAGES_PER_BLOCK       = 10             # <-- acá controlás "cada 10 páginas"

# Importante para evitar cortes por contexto/salida en Ollama:
NUM_CTX               = 16384          # contexto (tokens) del modelo en Ollama
NUM_PREDICT           = 9000           # tokens de salida máximos

DEFAULT_TEMPLATE = (
    """Eres un asistente especializado en anonimizar historias clínicas en español.

        INSTRUCCIONES OBLIGATORIAS
        1) Sustituye SOLO datos personales por estos placeholders exactos:
        - Nombres y apellidos de personas (pacientes, familiares, médicos) → [NOMBRE]
        - Teléfonos (cualquier formato, nacional o internacional) → [TELEFONO]
        - Cédulas de identidad / documentos → [CI]
        - Direcciones postales/domicilios (calle/avenida + número, esquinas, apto, barrio) → [DIRECCIÓN]
        2) Conserva TODO lo demás sin cambios: síntomas, diagnósticos, dosis, resultados, unidades, abreviaturas, signos de puntuación, mayúsculas/minúsculas.
        3) Si ya hay placeholders ([NOMBRE], [TELEFONO], [CI], [DIRECCIÓN]), NO los modifiques.
        4) Títulos y roles: conserva el título y reemplaza solo el nombre. Ej.: “Dr. [NOMBRE]”, “Lic. [NOMBRE]”.
        5) Teléfonos: reemplaza secuencias de 7+ dígitos o con separadores (+598, -, espacios, paréntesis).
        6) Direcciones: incluye referencias claras de domicilio (calle/esquina/número/apto/barrio).
        7) No inventes datos, no agregues comentarios, no cambies el formato. Respeta saltos de línea y espacios originales.
        8) Devuelve ÚNICAMENTE el texto anonimizado, sin explicaciones ni encabezados.

        Texto a anonimizar:
        {text}"""
)

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

    text = []
    for line in resp.iter_lines():
        if not line:
            continue
        chunk = json.loads(line)
        part = chunk.get("response", "")
        if part:
            text.append(part)
    return "".join(text).strip()

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

    return doc.tobytes()

# ---- UI mínima (sin parámetros) ----
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
        # 1) Armar bloques de páginas de a PAGES_PER_BLOCK
        blocks = []  # lista de tuplas: (start_page, end_page, block_text)
        for start in range(0, num_pages, PAGES_PER_BLOCK):
            end = min(start + PAGES_PER_BLOCK, num_pages)
            block_pages = pages_text[start:end]
            block_text = "\n".join(block_pages).strip()
            blocks.append((start + 1, end, block_text))  # páginas 1-based

        st.caption(f"Bloques de páginas generados: {len(blocks)} (de a {PAGES_PER_BLOCK} páginas)")

        all_results = []

        # 2) Procesar cada bloque de páginas INDEPENDIENTE
        for block_idx, (start_page, end_page, block_text) in enumerate(blocks, start=1):
            st.write(f"🔹 Procesando bloque {block_idx}/{len(blocks)} (págs {start_page}–{end_page})")

            # Dentro del bloque, seguimos usando chunking por caracteres si hace falta
            if USE_CHUNKING and len(block_text) > MAX_CHARS_PER_CHUNK:
                chunks = chunk_text_by_chars(block_text, max_chars=MAX_CHARS_PER_CHUNK, overlap=OVERLAP)
                st.write(f"   → Chunks en este bloque: {len(chunks)}")

                block_out_parts = []
                for i, ch in enumerate(chunks, start=1):
                    st.write(f"      → Chunk {i}/{len(chunks)} del bloque {block_idx}")
                    prompt = (
                        DEFAULT_TEMPLATE.replace("{text}", ch)
                        if "{text}" in DEFAULT_TEMPLATE
                        else f"{DEFAULT_TEMPLATE.strip()}\n\n{ch}"
                    )
                    try:
                        out = ollama_generate(
                            model=MODEL_NAME,
                            prompt=prompt,
                            endpoint=OLLAMA_ENDPOINT,
                            temperature=TEMPERATURE,
                        )
                    except Exception as e:
                        st.error(f"Error en chunk {i} del bloque {block_idx}: {e}")
                        st.stop()
                    block_out_parts.append(out.strip())

                block_result = "\n\n".join([p for p in block_out_parts if p]).strip()
            else:
                # Bloque suficientemente chico: va en una sola llamada
                prompt = (
                    DEFAULT_TEMPLATE.replace("{text}", block_text)
                    if "{text}" in DEFAULT_TEMPLATE
                    else f"{DEFAULT_TEMPLATE.strip()}\n\n{block_text}"
                )
                try:
                    block_result = ollama_generate(
                        model=MODEL_NAME,
                        prompt=prompt,
                        endpoint=OLLAMA_ENDPOINT,
                        temperature=TEMPERATURE,
                    )
                except Exception as e:
                    st.error(f"Error llamando a Ollama en bloque {block_idx}: {e}")
                    st.stop()

            all_results.append(block_result)

        # 3) Juntar todos los bloques anonimizados
        result = "\n\n".join([r for r in all_results if r]).strip()

        if not result:
            st.warning("La salida está vacía. Ajustá MAX_CHARS_PER_CHUNK o verificá NUM_CTX/NUM_PREDICT.")
        else:
            st.subheader("🧾 Salida del modelo (todos los bloques)")
            st.text_area("Texto generado", value=result, height=300)

            # 4) Generar un único PDF con TODO el texto anonimizado
            with st.spinner("Generando PDF…"):
                try:
                    pdf_out_bytes = text_to_pdf_bytes(
                        result,
                        paper="A4",
                        fontname="Courier",
                        fontsize=10,
                        margin=36,
                        line_spacing=1.4,
                    )
                except Exception as e:
                    st.error(f"No se pudo generar el PDF: {e}")
                    st.stop()

            # 5) Botón de descarga del PDF anonimizado completo
            st.download_button(
                "📄 Descargar salida (PDF anonimizado)",
                data=io.BytesIO(pdf_out_bytes),
                file_name="salida_ollama_anonimizada.pdf",
                mime="application/pdf",
            )

            st.success("¡Listo! Se procesaron todos los bloques y se generó el PDF anonimizado ✅")
else:
    st.info("Subí un PDF para comenzar.")
