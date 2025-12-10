import io
from datetime import datetime

import streamlit as st

import sys
from pathlib import Path


SRC_PATH = Path(__file__).resolve().parents[1] 
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


from llm.model_config import (
    MODEL_CONFIGS,
    set_active_model,
    SECTIONS_PER_BLOCK,
)
from utils.pdf_utils import pdf_bytes_to_pages
from utils.section_utils import (
    build_sectioned_text_from_pages,
    split_text_into_sections,
    map_section_ids_to_dates,
)
from services.patient_extraction import extract_patient_data_chain
from services.pipeline import full_pipeline_pdf_pages_to_merged_pdf


def main():
    st.set_page_config(page_title="Anonimización de PDFs", layout="centered")
    st.title("Herramienta de anonimización de documentos")

    # -------- Selector de modelo --------
    model_label = st.selectbox(
        "Elegir modelo",
        list(MODEL_CONFIGS.keys()),
        index=0,
    )

    cfg = set_active_model(model_label)

    st.caption(f"Modelo seleccionado: **{model_label}**")

    uploaded = st.file_uploader("Cargar PDF", type=["pdf"])

    if uploaded is None:
        st.info("Subí un PDF para comenzar.")
        return

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
    st.caption(f"Secciones detectadas: {len(section_ids)}.")

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
            "No se detectaron fechas con formato dd/mm/aaaa en las secciones."
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
        "Se procesarán todas las secciones cuya fecha esté dentro del rango indicado."
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

        with st.spinner("Procesando bloques..."):
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
                "presentaron salidas vacías de los prompts:\n\n" +
                "\n".join(f"- {msg}" for msg in warnings)
            )

        if not final_pdf_bytes:
            st.warning(
                "La salida está vacía. Revisá el PDF original o el rango de fechas indicado."
            )
            return

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


if __name__ == "__main__":
    main()
