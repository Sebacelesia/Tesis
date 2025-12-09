import os
import io
import json
import shutil
import tempfile
import textwrap
from typing import Optional, List, Callable, Tuple
from datetime import datetime, date

from llm.prompts import PROMPT2_TEMPLATE, PROMPT3_TEMPLATE, PROMPT5_TEMPLATE
from llm.ollama_client import ollama_generate
from llm.model_config import (
    MODEL_NAME,
    OLLAMA_ENDPOINT,
    TEMPERATURE,
    USE_CHUNKING,
    MAX_CHARS_PER_CHUNK,
    OVERLAP,
    SECTIONS_PER_BLOCK,
    DEBUG_PIPELINE,
)

from utils.text_utils import maybe_strip_think, chunk_text_by_chars
from utils.pdf_utils import text_to_pdf_bytes, merge_pdfs
from utils.section_utils import (
    build_sectioned_text_from_pages,
    split_text_into_sections,
    map_section_ids_to_dates,
)


def censor_block_text(block_text: str, patient_data_list: List[str]) -> str:
    """
    Censura datos del paciente en un bloque de texto usando Prompt 2.
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
