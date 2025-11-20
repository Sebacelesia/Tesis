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

USE_CHUNKING          = True           # si el texto supera MAX_CHARS_PER_CHUNK, se parte
MAX_CHARS_PER_CHUNK   = 15000         # <- cambiá acá (p.ej., 10k)
OVERLAP               = 10           # solapamiento entre chunks

# Importante para evitar cortes por contexto/salida en Ollama:

NUM_CTX               = 16384          # contexto (tokens) del modelo en Ollama
NUM_PREDICT           = 9000           # tokens de salida máximos (subí si necesitás más)

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

        Texto a anonimizar:
        {text}"""
)

# =======================================

# ---- PDF → Texto (PyMuPDF) ----
def pdf_bytes_to_text(pdf_bytes: bytes) -> str:
    import fitz  # PyMuPDF
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    parts: List[str] = []
    for page in doc:
        parts.append(page.get_text())
    return "\n".join(parts).strip()

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
            text = pdf_bytes_to_text(pdf_bytes)
        except Exception as e:
            st.error(f"Error al leer el PDF: {e}")
            st.stop()

    st.success("PDF leído correctamente.")
    st.caption(f"Caracteres extraídos: {len(text)} (chunk: {MAX_CHARS_PER_CHUNK}, overlap: {OVERLAP})")
    st.text_area("Vista previa del texto", value=text[:2000] + ("..." if len(text) > 2000 else ""), height=200)

    if st.button("🚀 Ejecutar en Ollama (parámetros fijos)"):
        # CHUNKING sobre el TEXTO (inyectamos template por chunk)
        if USE_CHUNKING and len(text) > MAX_CHARS_PER_CHUNK:
            chunks = chunk_text_by_chars(text, max_chars=MAX_CHARS_PER_CHUNK, overlap=OVERLAP)
            st.caption(f"Chunks generados: {len(chunks)}")

            out_parts = []
            for i, ch in enumerate(chunks, start=1):
                st.write(f"→ Procesando chunk {i}/{len(chunks)}…")
                prompt = DEFAULT_TEMPLATE.replace("{text}", ch) if "{text}" in DEFAULT_TEMPLATE else f"{DEFAULT_TEMPLATE.strip()}\n\n{ch}"
                try:
                    out = ollama_generate(model=MODEL_NAME, prompt=prompt, endpoint=OLLAMA_ENDPOINT, temperature=TEMPERATURE)
                except Exception as e:
                    st.error(f"Error en chunk {i}: {e}")
                    st.stop()
                out_parts.append(out.strip())
            result = "\n\n".join([p for p in out_parts if p]).strip()
        else:
            # TODO el texto en una sola llamada
            prompt = DEFAULT_TEMPLATE.replace("{text}", text) if "{text}" in DEFAULT_TEMPLATE else f"{DEFAULT_TEMPLATE.strip()}\n\n{text}"
            with st.spinner("Consultando Ollama…"):
                try:
                    result = ollama_generate(model=MODEL_NAME, prompt=prompt, endpoint=OLLAMA_ENDPOINT, temperature=TEMPERATURE)
                except Exception as e:
                    st.error(f"Error llamando a Ollama: {e}")
                    st.stop()

        if not result:
            st.warning("La salida está vacía. Ajustá MAX_CHARS_PER_CHUNK o verificá NUM_CTX/NUM_PREDICT.")
        else:
            st.subheader("🧾 Salida del modelo")
            st.text_area("Texto generado", value=result, height=300)

            # Descarga en PDF
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

            st.download_button(
                "📄 Descargar salida (PDF)",
                data=io.BytesIO(pdf_out_bytes),
                file_name="salida_ollama.pdf",
                mime="application/pdf",
            )

            st.success("¡Listo! El modelo respondió y el PDF fue generado ✅")
else:
    st.info("Subí un PDF para comenzar.")
