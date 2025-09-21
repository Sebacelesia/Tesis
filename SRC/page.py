#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import fitz  # PyMuPDF
import re
import io

st.set_page_config(page_title="Anonimizador Historias Clinicas", layout="centered")
st.title("📑 Anonimizador de historias clinicas PDF → TXT")

# Opción de procesamiento
modo = st.radio("Selecciona el modo de procesamiento:", 
                ["PDF completo", "Filtrar por rango de fechas"])

# Subir archivo
uploaded_file = st.file_uploader("Sube un archivo PDF", type=["pdf"])

if uploaded_file is not None:
    # Abrir PDF
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    
    # Extraer texto completo
    texto = ""
    for pagina in doc:
        texto += pagina.get_text()

    # Buscar encabezado
    patron_encabezado = re.search(r'Nombre:.*?Registro:\s*\d+', texto, re.DOTALL | re.IGNORECASE)
    if patron_encabezado:
        encabezado = patron_encabezado.group(0).strip()
        texto_sin_encabezado = texto.replace(encabezado, '').strip()
    else:
        encabezado = ''
        texto_sin_encabezado = texto

    # Dividir evoluciones por "Fecha:"
    evoluciones = re.split(r'(?=Fecha:)', texto_sin_encabezado)
    evoluciones_filtradas = [sec.strip() for sec in evoluciones if len(sec.split()) >= 10]

    # Filtrado por rango de fechas
    if modo == "Filtrar por rango de fechas":
        fecha_inicio = st.date_input("Fecha inicio")
        fecha_fin = st.date_input("Fecha fin")

        def extraer_fecha(texto_sec):
            # Busca formato Fecha: dd/mm/yyyy
            match = re.search(r'Fecha:\s*(\d{2}/\d{2}/\d{4})', texto_sec)
            if match:
                return match.group(1)
            return None

        from datetime import datetime
        evoluciones_filtradas_rango = []
        for sec in evoluciones_filtradas:
            fecha_str = extraer_fecha(sec)
            if fecha_str:
                fecha_obj = datetime.strptime(fecha_str, "%d/%m/%Y").date()
                if fecha_inicio <= fecha_obj <= fecha_fin:
                    evoluciones_filtradas_rango.append(sec)
        evoluciones_filtradas = evoluciones_filtradas_rango

    # Construir lista final de secciones
    secciones = [encabezado] + evoluciones_filtradas if encabezado else evoluciones_filtradas

    # Armar texto final
    texto_final = ""
    for i, sec in enumerate(secciones):
        if i == 0 and encabezado:
            texto_final += f"--- Sección 0 (Encabezado) ---\n{sec.strip()}\n\n"
        else:
            texto_final += f"--- Sección {i} ---\n{sec.strip()}\n\n"

    # Mostrar preview
    st.subheader("Vista previa del texto procesado")
    st.text_area("Texto", texto_final[:2000] + ("..." if len(texto_final) > 2000 else ""), height=400)

    # Generar TXT descargable
    txt_bytes = io.BytesIO(texto_final.encode("utf-8"))
    st.download_button(
        label="📥 Descargar TXT procesado",
        data=txt_bytes,
        file_name="procesado.txt",
        mime="text/plain"
    )

