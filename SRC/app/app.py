# app/app.py
import os, sys, io
from datetime import date
import streamlit as st

from slm.client import load_flan_small, run_prompt_flan
from slm.client import load_qwen_hf, run_prompt_qwen_hf  #


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from services.widget import pdf_uploader
from services.text_processing import (
    pdf_bytes_to_text,
    strip_header_block,
    split_evoluciones,
    filter_evoluciones_by_date_range,
    build_output_text,
    chunk_text_by_chars,
    text_to_pdf_bytes,
)
from slm.client import load_flan_small, run_prompt_flan

# Prompt fijo en código (mejor con prefijo de tarea para T5)
PROMPT_1 = (
    "resumir en español: "  # prefijo tipo tarea para T5/FLAN
    "por favor haz un resumen breve (1-2 oraciones) manteniendo el sentido clínico."
)

@st.cache_resource(show_spinner=False)
def get_flan_pipe():
    return load_flan_small()

@st.cache_resource(show_spinner=False)
def get_qwen_bundle(mode: str = "4bit"):
    # Elegí "fp16" o "bf16" si tenés GPU con suficiente VRAM; "4bit" ahorra memoria
    return load_qwen_hf(mode=mode)

st.set_page_config(page_title="Anonimizador Historias Clínicas", layout="centered")
st.title("📑 Anonimizador PDF → TXT → FLAN → PDF")

with st.sidebar:
    st.subheader("⚙️ Parámetros del modelo")
    max_chars = st.number_input("Tamaño del chunk (caracteres)", 200, 4000, 800, 50)
    overlap   = st.number_input("Solapamiento", 0, 1000, 80, 10)
    max_new   = st.number_input("max_new_tokens (salida)", 8, 1024, 256, 8)

    engine = st.selectbox("Motor", ["FLAN-T5 small (HF)", "Qwen 7B/8B (HF)"], index=1)
    qwen_mode = "4bit"
    if engine == "Qwen 7B/8B (HF)":
        qwen_mode = st.selectbox("Modo de Qwen", ["4bit", "fp16", "bf16", "auto", "cpu"], index=0)

    st.caption("PROMPT_1 (fijo en código):")
    st.code(PROMPT_1, language="text")


modo = st.radio("Selecciona el modo de procesamiento:",
                ["PDF completo", "Filtrar por rango de fechas"])

uploaded_file = pdf_uploader("Sube un archivo PDF")

# ------ Botón de prueba aislada del modelo ------
with st.expander("🔬 Probar modelo sin PDF (sanity check)"):
    if st.button("Probar FLAN (debería decir 'OK')"):
        pipe = get_flan_pipe()
        prueba = run_prompt_flan(
            pipe,
            prompt="Di 'OK' en mayúsculas.",
            max_new_tokens=8
        )
        st.write("Salida de prueba:", repr(prueba))

if uploaded_file is not None:
    pdf_bytes = uploaded_file.read()

    # 1) PDF → texto
    texto = pdf_bytes_to_text(pdf_bytes)

    # 2) Encabezado (opcional) + cuerpo sin encabezado
    encabezado, texto_sin_encabezado = strip_header_block(texto)

    # 3) Evoluciones y limpieza mínima
    evoluciones = split_evoluciones(texto_sin_encabezado)

    # 4) (Opcional) filtrar por fecha
    if modo == "Filtrar por rango de fechas":
        col1, col2 = st.columns(2)
        with col1:
            fecha_inicio = st.date_input("Fecha inicio", value=date(2000, 1, 1))
        with col2:
            fecha_fin = st.date_input("Fecha fin", value=date.today())
        evoluciones = filter_evoluciones_by_date_range(evoluciones, fecha_inicio, fecha_fin)

    # 5) Construir TXT final (PROMPT_2)
    texto_final = build_output_text(encabezado, evoluciones)

    # Diagnóstico de entrada
    st.caption(f"Caracteres en TXT procesado (PROMPT_2): {len(texto_final)}")
    st.caption(f"Primeros 200 chars del TXT: {repr(texto_final[:200])}")

    # 6) Vista previa + descarga TXT
    st.subheader("Vista previa del TXT procesado")
    preview = texto_final[:2000] + ("..." if len(texto_final) > 2000 else "")
    st.text_area("Texto", preview, height=300)

    st.download_button(
        label="📥 Descargar TXT procesado",
        data=io.BytesIO(texto_final.encode("utf-8")),
        file_name="procesado.txt",
        mime="text/plain",
    )

    st.divider()
    st.subheader("🤖 Ejecutar FLAN-T5 sobre (PROMPT_1 + TXT) y descargar PDF")

    if st.button("Ejecutar modelo (FLAN-T5 small)"):
        pipe = get_flan_pipe()

        # 7) Chunkear
        chunks = chunk_text_by_chars(texto_final, max_chars=int(max_chars), overlap=int(overlap))
        st.caption(f"Chunks generados: {len(chunks)}")
        if chunks:
            st.caption(f"Len primer chunk: {len(chunks[0])}")

        # 8) Ejecutar por chunk
        outputs = []
    if engine == "Qwen 7B/8B (HF)":
        qwen_bundle = get_qwen_bundle(qwen_mode)

    for i, ch in enumerate(chunks, start=1):
        if engine == "FLAN-T5 small (HF)":
            combined_prompt = f"{PROMPT_1}\n\nTexto {i}:\n{ch}"
            out = run_prompt_flan(
                get_flan_pipe(),
                prompt=combined_prompt,
                max_new_tokens=int(max_new),
                num_beams=4,
                do_sample=False,
                temperature=0.0,
            )
        else:
            # En Qwen usamos plantillas de chat:
            # - system = PROMPT_1 (reglas)
            # - user   = el chunk
            out = run_prompt_qwen_hf(
                qwen_bundle,
                system=PROMPT_1,
                user=f"Texto {i}:\n{ch}",
                max_new_tokens=int(max_new),
                temperature=0.2,
                do_sample=False,
            )

        outputs.append(out.strip())


        # 9) Unir salida
        salida_modelo = "\n\n".join([o for o in outputs if o]).strip()

        st.caption(f"Caracteres en salida del modelo: {len(salida_modelo)}")
        if not salida_modelo:
            st.warning("La salida del modelo está vacía. Probá bajar el tamaño del chunk, cambiar PROMPT_1 o aumentar max_new_tokens.")
        else:
            st.text_area("Salida del modelo", salida_modelo, height=250)

            # 10) PDF de la salida
            pdf_out_bytes = text_to_pdf_bytes(
                salida_modelo,
                paper="A4",
                fontname="courier",
                fontsize=11,
                margin=36,
            )
            st.download_button(
                label="📄 Descargar PDF con salida del modelo",
                data=pdf_out_bytes,
                file_name="salida_modelo.pdf",
                mime="application/pdf",
            )
