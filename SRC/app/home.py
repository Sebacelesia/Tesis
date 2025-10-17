# app.py (parámetros predeterminados)
import io
import json
import requests
import streamlit as st
from typing import Optional, List

# ====== PARÁMETROS PREDETERMINADOS ======
OLLAMA_ENDPOINT = "http://localhost:11434"
MODEL_NAME      = "qwen3:8b"
TEMPERATURE     = 0.2

USE_CHUNKING = True
MAX_CHARS    = 8000
OVERLAP      = 200

DEFAULT_TEMPLATE = (
    "Resumí el siguiente texto en 5 puntos claros y concisos:\n\n{text}"
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
    Llama al endpoint /api/generate de Ollama en modo stream y acumula la respuesta.
    Si Ollama está con GPU y drivers OK, el modelo correrá en GPU.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": {
            "temperature": temperature,
            **(options or {}),
        },
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

# ---- Chunking simple por caracteres ----
def chunk_text_by_chars(text: str, max_chars: int = MAX_CHARS, overlap: int = OVERLAP) -> List[str]:
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

# ---- UI ----
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
    st.caption(f"Caracteres extraídos: {len(text)}")
    st.text_area("Vista previa del texto", value=text[:2000] + ("..." if len(text) > 2000 else ""), height=200)

    if st.button("🚀 Ejecutar en Ollama (parámetros predeterminados)"):
        # Construir prompt final
        template = DEFAULT_TEMPLATE
        if "{text}" in template:
            final_input = template.replace("{text}", text)
        else:
            final_input = f"{template.strip()}\n\n{text}"

        # Chunking (si corresponde)
        if USE_CHUNKING and len(final_input) > MAX_CHARS:
            with st.spinner("Dividiendo en chunks y ejecutando..."):
                chunks = chunk_text_by_chars(final_input, max_chars=MAX_CHARS, overlap=OVERLAP)
                st.caption(f"Chunks generados: {len(chunks)}")

                partials = []
                for i, ch in enumerate(chunks, start=1):
                    st.write(f"→ Procesando chunk {i}/{len(chunks)}…")
                    try:
                        out = ollama_generate(model=MODEL_NAME, prompt=ch, endpoint=OLLAMA_ENDPOINT, temperature=TEMPERATURE)
                    except Exception as e:
                        st.error(f"Error en chunk {i}: {e}")
                        st.stop()
                    partials.append(out.strip())

                result = "\n\n".join([p for p in partials if p]).strip()
        else:
            with st.spinner("Consultando Ollama…"):
                try:
                    result = ollama_generate(model=MODEL_NAME, prompt=final_input, endpoint=OLLAMA_ENDPOINT, temperature=TEMPERATURE)
                except Exception as e:
                    st.error(f"Error llamando a Ollama: {e}")
                    st.stop()

        if not result:
            st.warning("La salida está vacía. Probá ajustar el template o desactivar chunking.")
        else:
            st.subheader("🧾 Salida del modelo")
            st.text_area("Texto generado", value=result, height=300)

            st.download_button(
                "📥 Descargar salida (.txt)",
                data=io.BytesIO(result.encode("utf-8")),
                file_name="salida_ollama.txt",
                mime="text/plain",
            )

            st.success("¡Listo! El modelo respondió correctamente.")
else:
    st.info("Subí un PDF para comenzar.")

