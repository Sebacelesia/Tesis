import fitz  # PyMuPDF
from typing import Optional, List
import textwrap


def pdf_bytes_to_text(pdf_bytes: bytes) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    parts: List[str] = []
    for page in doc:
        parts.append(page.get_text())
    return "\n".join(parts).strip()


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
