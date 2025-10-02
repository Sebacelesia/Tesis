# app/app.py
import os, sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))  # carpeta SRC
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

import io
from datetime import date
import streamlit as st

from services.widget import pdf_uploader
from services.text_processing import (
    pdf_bytes_to_text,
    strip_header_block,
    split_evoluciones,
    filter_evoluciones_by_date_range,
    build_output_text,
)

st.set_page_config(page_title="Anonimizador Historias Clínicas", layout="centered")
st.title("📑 Anonimizador de historias clínicas PDF → TXT")

modo = st.radio("Selecciona el modo de procesamiento:",
                ["PDF completo", "Filtrar por rango de fechas"])

uploaded_file = pdf_uploader("Sube un archivo PDF")

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

    # 5) Construir texto final
    texto_final = build_output_text(encabezado, evoluciones)

    # 6) Vista previa y descarga
    st.subheader("Vista previa del texto procesado")
    preview = texto_final[:2000] + ("..." if len(texto_final) > 2000 else "")
    st.text_area("Texto", preview, height=400)

    st.download_button(
        label="📥 Descargar TXT procesado",
        data=io.BytesIO(texto_final.encode("utf-8")),
        file_name="procesado.txt",
        mime="text/plain",
    )
