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
