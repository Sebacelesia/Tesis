import re
from datetime import datetime, date
from typing import List, Tuple, Optional

from llm.model_config import SECTION_HEADER_REGEX


def build_sectioned_text_from_pages(pages_text: List[str]) -> str:
    """
    Construye un texto único seccionado a partir de las páginas del PDF.
    """
    full_text = "\n".join(pages_text).strip()
    if not full_text:
        return ""

    evoluciones = re.split(r'(?=Fecha:)', full_text)
    evoluciones_filtradas = [sec.strip() for sec in evoluciones if len(sec.split()) >= 10]

    partes: List[str] = []
    for i, sec in enumerate(evoluciones_filtradas, start=1):
        partes.append(f"--- Sección {i} ---\n{sec.strip()}\n")

    return "\n\n".join(partes).strip()


def split_text_into_sections(text: str) -> List[Tuple[int, str]]:
    """
    Divide el texto completo en una lista de tuplas (numero_seccion, texto_de_la_seccion).
    """
    matches = list(SECTION_HEADER_REGEX.finditer(text))
    if not matches:
        return [(0, text)]

    sections: List[Tuple[int, str]] = []

    first_start = matches[0].start()
    preamble = text[:first_start].strip()
    if preamble:
        sections.append((-1, preamble))

    for i, m in enumerate(matches):
        sec_num_str = m.group(1)
        sec_num = int(sec_num_str)
        start = m.start()

        if i + 1 < len(matches):
            end = matches[i + 1].start()
        else:
            end = len(text)

        full_section_text = text[start:end].strip("\n")
        sections.append((sec_num, full_section_text))

    return sections


def extract_date_from_section_text(text: str) -> Optional[str]:
    """
    Busca una fecha con formato dd/mm/aaaa en el texto de la sección.
    """
    m = re.search(r"\b(\d{1,2}/\d{1,2}/\d{4})\b", text)
    if m:
        return m.group(1).strip()
    return None


def map_section_ids_to_dates(sections: List[Tuple[int, str]]):
    """
    Asocia cada sección con una fecha detectada en su contenido.
    """
    section_dates = {}
    all_dates: List[date] = []

    for sid, sec_text in sections:
        if sid >= 0:
            d_str = extract_date_from_section_text(sec_text)
            if d_str:
                try:
                    d = datetime.strptime(d_str, "%d/%m/%Y").date()
                except ValueError:
                    d = None
            else:
                d = None

            section_dates[sid] = d
            if d is not None:
                all_dates.append(d)

    return section_dates, all_dates
