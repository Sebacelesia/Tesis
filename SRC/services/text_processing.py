from __future__ import annotations
from typing import List, Tuple, Optional
from datetime import date, datetime
import re
import fitz  # PyMuPDF

# ---------- 1) PDF → Texto ----------
def pdf_bytes_to_text(pdf_bytes: bytes) -> str:
    """Convierte bytes de un PDF en texto concatenado (una cadena)."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    return "".join(page.get_text() for page in doc)




# ---------- 2) Encabezado ----------
_HEADER_PAT = re.compile(r'Nombre:.*?Registro:\s*\d+', re.DOTALL | re.IGNORECASE)

def strip_header_block(texto: str) -> Tuple[Optional[str], str]:
    """
    Extrae (si existe) bloque de encabezado y retorna (encabezado, texto_sin_encabezado).
    Si no hay encabezado, devuelve (None, texto_original).
    """
    m = _HEADER_PAT.search(texto)
    if m:
        encabezado = m.group(0).strip()
        cuerpo = texto.replace(m.group(0), '').strip()
        return encabezado, cuerpo
    return None, texto

# ---------- 3) Evoluciones ----------
_SPLIT_EVOL = re.compile(r'(?=Fecha:)')  # mantiene "Fecha:" al inicio de cada sección

def split_evoluciones(texto_sin_encabezado: str, min_tokens: int = 10) -> List[str]:
    """
    Divide por 'Fecha:' y limpia secciones muy cortas/ruidosas.
    """
    crudos = _SPLIT_EVOL.split(texto_sin_encabezado)
    # limpiar y filtrar breves
    evoluciones = [s.strip() for s in crudos if len(s.split()) >= min_tokens]
    return evoluciones

# ---------- 4) Filtrado por rango de fechas ----------
_DATE_PAT = re.compile(r'Fecha:\s*(\d{2}/\d{2}/\d{4})')

def _extraer_fecha_ddmmyyyy(texto_sec: str) -> Optional[date]:
    m = _DATE_PAT.search(texto_sec)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%d/%m/%Y").date()
    except ValueError:
        return None

def filter_evoluciones_by_date_range(secciones: List[str],
                                     f_ini: date,
                                     f_fin: date) -> List[str]:
    """
    Mantiene solo secciones cuya fecha (en formato 'Fecha: dd/mm/yyyy') esté en [f_ini, f_fin].
    """
    keep = []
    for sec in secciones:
        f = _extraer_fecha_ddmmyyyy(sec)
        if f and f_ini <= f <= f_fin:
            keep.append(sec)
    return keep

# ---------- 5) Construcción de salida ----------
def build_output_text(encabezado: Optional[str], evoluciones: List[str]) -> str:
    """
    Ensambla texto final numerado. Si hay encabezado, se usa como Sección 0.
    """
    secciones = ([encabezado] + evoluciones) if encabezado else evoluciones
    out_lines = []
    for i, sec in enumerate(secciones):
        title = "--- Sección 0 (Encabezado) ---" if (i == 0 and encabezado) else f"--- Sección {i} ---"
        out_lines.append(f"{title}\n{sec.strip()}\n")
    return "\n".join(out_lines).strip()


# --- CHUNKING por caracteres (para no pasarse del contexto del modelo) ---
def chunk_text_by_chars(text: str, max_chars: int = 1200, overlap: int = 100) -> list[str]:
    """
    Parte el texto en trozos por cantidad de caracteres.
    - max_chars: tamaño del chunk.
    - overlap: solapamiento entre chunks para no cortar ideas.
    """
    if max_chars <= 0:
        raise ValueError("max_chars debe ser > 0")
    if overlap < 0 or overlap >= max_chars:
        overlap = 0

    chunks = []
    step = max_chars - overlap
    for i in range(0, len(text), step):
        chunks.append(text[i:i + max_chars])
    return chunks


# --- TEXTO → PDF (devuelve bytes, ideal para st.download_button) ---

import textwrap
from typing import List, Optional

def text_to_pdf_bytes(
    text: str,
    paper: str = "A4",
    fontname: str = "courier",   # monoespaciada → wrapping estable
    fontsize: float = 11,
    margin: float = 36,
    line_spacing: float = 1.25,
    wrap_chars: Optional[int] = None,
) -> bytes:
    """
    Genera un PDF multi-página dibujando línea por línea (sin insert_textbox).
    - Calcula el ancho aprox. por carácter y ajusta el wrap.
    - Fuerza el color negro (evita texto blanco “invisible”).
    """
    # 0) Normalizar entrada
    text = "" if text is None else str(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if not text.strip():
        text = "(sin contenido)"

    # 1) Preparar documento y geometría
    doc = fitz.open()
    try:
        page_rect = fitz.paper_rect(paper)
        usable_w  = page_rect.width  - 2 * margin
        usable_h  = page_rect.height - 2 * margin

        # Ancho aprox. por carácter
        try:
            ch_w = fitz.get_text_length("M", fontname=fontname, fontsize=fontsize)
            if ch_w <= 0:
                raise ValueError
        except Exception:
            ch_w = max(1.0, fontsize * 0.6)  # fallback

        if wrap_chars is None:
            wrap_chars = max(10, int(usable_w // ch_w))

        line_h    = fontsize * line_spacing
        max_lines = max(1, int(usable_h // line_h))

        # 2) Partir en líneas (respeta saltos y envuelve párrafos largos)
        lines: List[str] = []
        for para in text.split("\n"):
            if para.strip() == "":
                lines.append("")  # línea en blanco
            else:
                wrapped = textwrap.wrap(
                    para,
                    width=wrap_chars,
                    replace_whitespace=False,
                    drop_whitespace=False,
                    break_long_words=True,
                    break_on_hyphens=False,
                )
                lines.extend(wrapped if wrapped else [""])

        # 3) Paginar y dibujar
        i = 0
        while i < len(lines):
            page = doc.new_page(width=page_rect.width, height=page_rect.height)
            y = margin
            for j in range(i, min(i + max_lines, len(lines))):
                page.insert_text(
                    (margin, y),
                    lines[j],
                    fontname=fontname,
                    fontsize=fontsize,
                    color=(0, 0, 0),  # negro explícito
                )
                y += line_h
            i += max_lines

        # 4) Devolver bytes del PDF
        try:
            return doc.write()
        except Exception:
            return doc.tobytes()
    finally:
        doc.close()