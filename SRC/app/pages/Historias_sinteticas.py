import io
import os
import json
import requests
import random
import textwrap
import streamlit as st
import google.generativeai as genai
import fitz 
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]

CASOS_PATH = ROOT / "data_partesintetica" / "casos.json"

with CASOS_PATH.open("r", encoding="utf-8") as f:
    CASOS = json.load(f)




OPENROUTER_KEY  = "API KEY"
genai.configure(api_key="API KEY")


PATOLOGIAS_VALIDAS = [
    "VIH", "Tuberculosis pulmonar", "Tuberculosis ganglionar",
    "Meningitis bacteriana", "Endocarditis",
    "Infección de prótesis osteoarticular", "Celulitis",
    "Fiebre prolongada", "Neumonía adquirida en la comunidad",
    "Infección urinaria", "Candidemia",
    "Infección de sitio quirúrgico",
    "Sepsis de origen desconocido"
]



def call_model_openrouter(prompt: str) -> str:
    url = "https://openrouter.ai/api/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {OPENROUTER_KEY}",
        "Content-Type": "application/json",
    }

    body = {
        "model": "openai/gpt-oss-20b:free",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2
    }

    resp = requests.post(url, headers=headers, json=body, timeout=120)
    resp.raise_for_status()
    data = resp.json()

    return data["choices"][0]["message"]["content"].strip()


def seleccionar_6_casos(patologia: str, casos: list):
    """
    Usa un modelo en OpenRouter para elegir los casos más similares.
    (Mantengo tu nombre de función original.)
    """
    casos_txt = "\n\n".join(
        f"CASO {i+1}:\n{c}" for i, c in enumerate(casos)
    )

    prompt = f"""
Sos un infectólogo experto.

Quiero que analices los siguientes 20 casos clínicos completos y elijas los
**8 casos más similares clínicamente** a la patología:

➡️ {patologia}

Instrucciones:
1) Razoná libremente como infectólogo.
2) Compará síntomas, evolución, localización, mecanismo fisiopatológico y foco.
3) NO uses coincidencias de palabras sino coincidencias clínicas reales.
4) Podés escribir todo el análisis que quieras.
5) En la ÚLTIMA línea devolvé SOLO:

CASOS_ELEGIDOS: n1, n2, n3, n4, n5, n6, n7, n8

──────────────────────────
CASOS:
{casos_txt}
"""

    respuesta = call_model_openrouter(prompt)

    lineas = [l.strip() for l in respuesta.split("\n") if l.strip() != ""]
    ultima = lineas[-1] if lineas else ""

    if not ultima.startswith("CASOS_ELEGIDOS:"):
        import re
        nums = re.findall(r"\b([1-9]|1[0-9]|20)\b", respuesta)
        indices = [int(x) for x in nums[:8]]
        return indices

    numeros = ultima.replace("CASOS_ELEGIDOS:", "").strip()
    indices = [int(x) for x in numeros.split(",") if x.strip().isdigit()]

    return indices


def elegir_3_random(indices_3):
    """
    Elige 3 al azar entre los índices elegidos por el modelo.
    Manejo defensivo por si vienen menos de 3.
    """
    uniq = list(dict.fromkeys(indices_3))  
    if len(uniq) <= 3:
        return uniq
    return random.sample(uniq, 3)


def generar_fewshot(indices_3, patologia, casos):
    salida = [
        "=====================",
        f"FEW SHOT – {patologia}",
        "=====================",
        ""
    ]

    for idx in indices_3:
       
        if 1 <= idx <= len(casos):
            texto = casos[idx - 1]
            salida.append(f"### CASO {idx}\n{texto}\n")

    return "\n".join(salida)


def text_to_pdf_bytes(
    text: str,
    paper: str = "A4",
    fontname: str = "Courier",
    fontsize: int = 10,
    margin: int = 36,
    line_spacing: float = 1.4,
) -> bytes:
    """
    Convierte texto plano a PDF en memoria.
    """
    doc = fitz.open()
    if paper.upper() == "A4":
        width, height = 595, 842
    else:
        width, height = 612, 792

    usable_w = width - 2 * margin
    usable_h = height - 2 * margin

    char_w = fontsize * 0.6
    max_chars_per_line = max(20, int(usable_w / char_w))

    line_h = int(fontsize * line_spacing)
    max_lines_per_page = max(10, int(usable_h / line_h))

    all_lines = []
    for para in text.splitlines():
        if not para.strip():
            all_lines.append("")
            continue
        wrapped = textwrap.wrap(para, width=max_chars_per_line, break_long_words=False)
        all_lines.extend(wrapped if wrapped else [""])

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

    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes

st.title("Generador de Historia Clínica Sintética – Infectología")

modo = st.radio(
    "Modo de generación:",
    [
        "Manual (edad, sexo, patología, motivo)",
        "Patología y motivo libres (infecc.)",
        "Totalmente libre",
    ]
)

patologia_usuario = None
motivo = None

if modo.startswith("Manual"):
    edad = st.number_input("Edad", min_value=0, max_value=120)
    sexo = st.selectbox("Sexo", ["Femenino", "Masculino", "Otro"])
    patologia_usuario = st.text_input("Patología")
    motivo = st.text_area(
        "Datos iniciales (comentarios libres: procedencia, motivo, convivencia, nacionalidad, etc.)",
        placeholder="Ej: Venezolano, radicado en Uruguay. Vive con pareja. Consulta por tos productiva."
    )

elif modo.startswith("Patología"):
    edad = None
    sexo = None
    patologia_usuario = None
    motivo = st.text_area(
        "Datos iniciales (comentarios libres: procedencia, motivo, convivencia, nacionalidad, etc.)",
        placeholder="Ej: Venezolano, radicado en Uruguay. Vive con pareja. Consulta por tos productiva."
    )

else:
    edad = None
    sexo = None
    patologia_usuario = None
    motivo = None
    st.write("(Modo totalmente libre: no se ingresa edad, sexo ni motivo.)")


if st.button("Generar historia clínica"):

    with st.spinner("Generando..."):


        if modo.startswith("Manual"):
            patologia_final = (patologia_usuario or "").strip()
            if not patologia_final:
                st.error("En modo Manual debés ingresar una patología.")
                st.stop()
        elif modo.startswith("Patología"):
            patologia_final = random.choice(PATOLOGIAS_VALIDAS)
        else:
            patologia_final = random.choice(PATOLOGIAS_VALIDAS)


        indices6 = seleccionar_6_casos(patologia_final, CASOS)
        indices3 = elegir_3_random(indices6)


        few_shot = generar_fewshot(indices3, patologia_final, CASOS)


        system_prompt = """
Sos un médico infectólogo uruguayo que trabaja en un hospital público (Maciel, Clínicas, Pasteur o INOT).
Redactás historias clínicas sintéticas indistinguibles del corpus real uruguayo (PDF + JSON).
El estilo debe ser telegráfico, fragmentado, con abreviaciones reales y jerga local auténtica. 

Regla N° 1 y FUNDAMENTAL:
La historia debe reproducir las pequeñas desprolijidades humanas presentes en los FEW-SHOTS seleccionados: variaciones reales en mayúsculas/minúsculas y nombres de secciones, orden irregular de sistemas, uso no uniforme de dos puntos y saltos de línea, puntuación inconsistente, frases entrecortadas u omitidas, repeticiones leves, conectores ausentes, cambios abruptos de ritmo y micro-omisiones típicas del corpus. Estas variaciones deben inspirarse en los FEW-SHOTS y mantenerse siempre con coherencia clínica básica y sin cambiar el sentido de EA y EF ni las abreviaciones reales del corpus.

La salida debe ser únicamente la historia clínica. Sin explicaciones.

════════════════════════════════════════════
1) SECCIONES PERMITIDAS
════════════════════════════════════════════
Usar exclusivamente secciones que existen en el corpus real:
SF, SM, MC, AP, AI, AQ, AEA, EA, EF.

Prohibido crear secciones nuevas (ej.: “Linfático”, “Cuello”, “Extremidades”, “Laboratorio”).
Prohibido escribir “Paciente F/M”.  
SF = femenina, SM = masculino.

El inicio debe copiar alguno de estos patrones reales, según lo que muestren los few-shots:
- “SF: …”
- “SM: …”
- “Paciente …”
- Inicio directo sin etiqueta

Edad y sexo SOLO pueden aparecer si aparecen en los few-shots seleccionados.
Hábitos tóxicos solo si aparecen en los few-shots o en el input del usuario.

════════════════════════════════════════════
2) ABREVIACIONES PERMITIDAS
════════════════════════════════════════════
Usar solamente abreviaciones del corpus:
RR, RBG, Tax, Sat, MAV, PyM, BF, LG, PP, PNM,
TU, TD, TDB, SP, VEA, entre otras que estén en el corpus.

════════════════════════════════════════════
3) EA — ENFERMEDAD ACTUAL
════════════════════════════════════════════
Estilo: telegráfico, coherente, no académico.

Inicios permitidos:
“Hace X días…”
“Comienza…”
“Desde entonces…”
“Refiere…”
“Consulta por…”

“Consulta por…” solo UNA SOLA VEZ al inicio del EA.

Prohibido:
- Diagnósticos
- Interpretaciones
- Laboratorio o imágenes

════════════════════════════════════════════
4) EF — EXAMEN FÍSICO
════════════════════════════════════════════
Imitar estilo real del corpus uruguayo.
No duplicar signos vitales en distintas secciones.
No inventar abreviaturas nuevas.

5) ESTILO GLOBAL
════════════════════════════════════════════
- Telegráfico, fragmentado, no académico.
- Jerga local del corpus.
- Sin diagnósticos ni laboratorio.
- Pequeñas imperfecciones naturales permitidas.
"""

        base_prompt = f"""
Actuás como un médico infectólogo uruguayo.

Debés redactar **una única historia clínica sintética**, verosímil y clínicamente coherente,
imitando el registro clínico hospitalario (telegráfico, abreviado, con jerga local).

**Modo:** {modo}
{f'Edad: {edad} años' if edad is not None else 'Edad: a definir'}
{f'Sexo: {sexo}' if sexo else 'Sexo: a definir'}
{f'Patología: {patologia_final}' if patologia_final else 'Patología: a definir dentro de Infectología'}
{f'Motivo: {motivo}' if motivo else 'Motivo: a definir'}

---

### Restricciones clínicas
1. Solo se permiten casos **de Infectología**.
   Si la patología o motivo **no pertenece a Infectología**, devolvé exactamente:
   "Error: el modelo está diseñado solo para historias clínicas de Infectología."
2. El caso debe ser **coherente**, sin contradicciones.
3. Si el cuadro incluye patologías no infecciosas (fractura, migraña, hipotiroidismo, etc.), devolvé el mismo mensaje de error.

---
"""

        prompt_total = (
            f"{system_prompt}\n\n"
            "### FEW SHOT (casos seleccionados por OSS)\n"
            f"{few_shot}\n\n"
            "---- FIN FEW SHOT ----\n\n"
            f"{base_prompt}"
        )

  
        model = genai.GenerativeModel("models/gemini-2.5-flash")

        response = model.generate_content(
            prompt_total,
            generation_config=genai.types.GenerationConfig(
                temperature=0.5,
                top_p=0.9,
                top_k=40,
            ),
        )

        historia = (response.text or "").strip()

        st.text_area("Historia clínica generada:", historia, height=420)
        st.text_area("Few shots tenidos en cuenta:", few_shot, height=420)

        if historia:
            pdf_bytes = text_to_pdf_bytes(historia)

            safe_pat = patologia_final.replace(" ", "_").replace("/", "_")
            st.download_button(
                "Descargar historia en PDF",
                data=pdf_bytes,
                file_name=f"historia_sintetica_{safe_pat}.pdf",
                mime="application/pdf",
            )
        else:
            st.warning("La historia generada está vacía.")
