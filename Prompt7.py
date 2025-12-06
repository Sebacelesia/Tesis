#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import google.generativeai as genai
import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import os
# ---------------------------------------------------------------------
# CONFIGURACIÓN
# ---------------------------------------------------------------------
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
st.title("🧫 Generador de Historia Clínica Sintética – Infectología (Gemini 2.5 Flash)")


EMB_PATH = r"C:\Users\59892\desktop\data\embeddings.npy"
CASES_PATH = r"C:\Users\59892\desktop\data\casos.json"

embeddings = np.load(EMB_PATH)
with open(CASES_PATH, "r", encoding="utf-8") as f:
    casos_index = json.load(f)

modelo_embed = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def buscar_casos_similares(texto_usuario, top_k=3):
    query_emb = modelo_embed.encode([texto_usuario], normalize_embeddings=True)
    sims = cosine_similarity(query_emb, embeddings)[0]
    idx_top = sims.argsort()[-top_k:][::-1]
    return [casos_index[i] for i in idx_top]

def casos_diversidad_media(embeddings, casos_index, top_k=3):
    sim_matrix = cosine_similarity(embeddings)
    mean_sim = sim_matrix.mean(axis=1)
    mean_total = mean_sim.mean()
    idx_sorted = np.argsort(np.abs(mean_sim - mean_total))
    return [casos_index[i] for i in idx_sorted[:top_k]]

# ---------------------------------------------------------------------
# INTERFAZ
# ---------------------------------------------------------------------
modo = st.radio(
    "Seleccioná el modo de generación:",
    [
        "1️⃣ Ingreso manual (Patología y Motivo determinados por el usuario)",
        "2️⃣ Patología y Motivo libres (el modelo los elige dentro de Infectología)",
        "3️⃣ Todo libre (edad, sexo, patología y motivo elegidos por el modelo)",
    ],
)

edad, sexo, patologia, motivo = None, None, None, None

if modo.startswith("1️⃣"):
    edad = st.number_input("Edad", min_value=0, max_value=125, step=1)
    sexo = st.selectbox("Sexo", ["Femenino", "Masculino", "Otro", "Prefiero no decir"])
    patologia = st.text_input("Patología", placeholder="Ej: VIH, tuberculosis, meningitis por H. influenzae…")
    motivo = st.text_area("Motivo de consulta", height=100, placeholder="Ej: fiebre prolongada, tos con expectoración…")
elif modo.startswith("2️⃣"):
    edad = st.number_input("Edad", min_value=0, max_value=125, step=1)
    sexo = st.selectbox("Sexo", ["Femenino", "Masculino", "Otro", "Prefiero no decir"])

# ---------------------------------------------------------------------
# PROMPT ACTUALIZADO – FLEXIBLE Y CON RESTRICCIÓN DE PATOLOGÍAS
# ---------------------------------------------------------------------

PATOLOGIAS_VALIDAS = [
    # VIH y complicaciones
    "VIH avanzado o controlado",
    "tuberculosis pulmonar o extrapulmonar",
    "neumonía adquirida en la comunidad",
    "neumonía intrahospitalaria",
    "neumonía por Pneumocystis jirovecii (PCP)",
    "meningitis criptocócica",
    "toxoplasmosis cerebral",
    "citomegalovirus (CMV)",
    "candidiasis mucocutánea o esofágica",
    "infección por herpes simple o zóster diseminado",

    # Bacterianas sistémicas
    "endocarditis infecciosa",
    "bacteriemia o sepsis",
    "fiebre prolongada o de origen infeccioso",
    "brucelosis",
    "leptospirosis",
    "salmonelosis",
    "listeriosis",
    "tétanos",
    "botulismo",

    # Infecciones respiratorias
    "bronquitis aguda",
    "neumonía bacteriana o viral",
    "absceso pulmonar",
    "tuberculosis pulmonar o miliar",
    "aspergilosis pulmonar",

    # Infecciones del sistema nervioso
    "meningitis bacteriana",
    "meningitis viral",
    "encefalitis viral",
    "neurotuberculosis",
    "absceso cerebral bacteriano o fúngico",

    # Infecciones gastrointestinales
    "hepatitis viral aguda o crónica",
    "fiebre tifoidea o paratifoidea",
    "diarrea infecciosa aguda",
    "amebiasis",
    "giardiasis",
    "colitis por Clostridium difficile",
    "absceso hepático amebiano o piógeno",

    # Infecciones urinarias
    "pielonefritis aguda o crónica",
    "infección urinaria complicada o asociada a catéter",

    # Infecciones cutáneas y osteoarticulares
    "celulitis",
    "erisipela",
    "absceso de partes blandas",
    "osteomielitis",
    "espondilodiscitis",
    "infección de sitio quirúrgico",
    "infección protésica o postoperatoria",

    # Infecciones tropicales / vectoriales
    "dengue",
    "chikungunya",
    "zika",
    "paludismo o malaria",
    "tripanosomiasis",
    "leishmaniasis cutánea o visceral",
    "filariasis",
    "esquistosomiasis",
    "fiebre amarilla",
    "hantavirosis",
    "tifus o rickettsiosis",

    # Infecciones zoonóticas o ambientales
    "leptospirosis",
    "brucelosis",
    "toxoplasmosis",
    "pasteurelosis",
    "antrax",
    "fiebre Q (Coxiella burnetii)",

    # Infecciones en inmunodeprimidos o hospitalarios
    "infección oportunista en paciente trasplantado",
    "bacteriemia por catéter venoso central",
    "infección urinaria nosocomial",
    "neumonía asociada a ventilación mecánica",
    "infección de herida quirúrgica",
    "infección fúngica invasiva (aspergilosis, candidemia)",
    "infección relacionada a dispositivos médicos",
    "infección por Pseudomonas aeruginosa",
    "infección por Acinetobacter baumannii"
]

# ---------------------------------------------------------------------
# SYSTEM PROMPT
# ---------------------------------------------------------------------

system_prompt = """
Sos un médico infectólogo uruguayo que trabaja en un hospital público (H. Maciel, Clínicas, Pasteur, INOT).
Redactás historias clínicas sintéticas con fines académicos, imitando el estilo real del registro hospitalario:
telegráfico, fragmentado, abreviado y con jerga local propia de Infectología.

Tu salida debe ser **solo texto clínico**, sin explicaciones ni markdown.
Debe sonar como una evolución real, escrita en sala o guardia.

---

La estructura es libre y adaptable.
Podés usar o no encabezados (AP, EA, EF…), fusionar secciones o escribir en texto continuo.
Usá los encabezados solo cuando resulten naturales; muchos médicos escriben sin ellos.
En algunos casos, el texto puede iniciar directamente con el motivo o el cuadro actual (“Ingresa por…” o “Consulta por…”).
Evitá formato de plantilla; buscá sonar natural, como nota clínica escrita con apuro en sala o guardia.

---

EA (Enfermedad Actual):
- Describí evolución y síntomas en orden temporal, frases cortas, sin hallazgos físicos.
- Puede iniciar con “Consulta por…” o “Ingresa por…”.
- Evitá repetir lo que luego aparezca en EF.

EF (Examen Físico):
- Describí solo lo constatado al examen físico.
- Incluí constantes vitales y hallazgos relevantes: piel, mucosas, CV, PP, Abd, MMII, SNM.
- Usá abreviaciones locales: EG, RR, FC, FR, Tax, Sat, MAV, SP, VEA, etc.
- “SP” se usa correctamente solo en contexto:
    • “Resto SP.” (para indicar que el resto del examen es normal)
    • “SP. Sin foco evidente.” 
    • “SP. VEA.”
  Nunca cerrar el texto solo con “SP.” sin aclaración.
- El EF debe cerrar con una frase realista de Infectología:
  “SP. Sin foco evidente.” o “SP. VEA.” o “SP. Sin foco evidente. En seg. por infecto.”

---

Estilo general:
- Telegráfico, impersonal, apurado.
- Frases cortas, puntuación discontinua, con omisiones o repeticiones leves.
- Repeticiones y desorden aceptables; evitar estructura narrativa fluida o literaria.
- La apertura (edad, sexo, procedencia) puede variar o incluso omitirse.
  Ejemplo: “F45a.”, “M40a VIH+…”, “Ingresa por sd febril…”.
  No usar siempre el mismo patrón inicial.
- Los apartados no deben tener todos la misma extensión: el EA suele ser más largo, AP y AEA más breves o ausentes.
- Simulá escritura hospitalaria real: abreviaciones sin patrón (“RR88cpm”, “Sat94%AA”, “Tax38°”, “FC98”), omisiones de artículos o verbos (“Sd febril 2m ev. Tos seca persist.”), frases cortadas (“Fiebre vespertina. Sudoración noct. Tos seca.”).
- Permití errores leves de formato, espaciado o unidades (“PA110/70”, “Sat95AA”, “GCS11/15”), incluso combinaciones dispares dentro del mismo texto.
- Alterná entre frases completas y fragmentos sueltos, y podés alterar ligeramente el orden temporal de los síntomas.
- Se permiten signos de puntuación inconsistentes (“;”, “-”, “/”) o mezclados (“RR88cpm RR normofon. FC98”).
- Aceptá variación entre médicos: algunos más detallados, otros más telegráficos o con abreviaciones idiosincráticas.
- Evitá texto limpio, equilibrado o académico; buscá aspecto mecanografiado o escrito en guardia, con leves irregularidades.
- Puede incluir pequeñas redundancias o incongruencias leves (“Niega fiebre. Consulta por fiebre.”) si son verosímiles clínicamente.
- Se permiten abreviaciones disparejas y jerga local: sd toxiinfeccioso, IO, PBC, OH, LT CD4, CV detectable, IRAB, VEA, SP, EG regular, sd febril prolongado.
- Sin diagnósticos, sin laboratorio, sin plan.



En general, el EA debe ser el segmento más extenso y detallado; los antecedentes y el examen físico son más concisos.
"""


few_shot = """
### 🧪 EJEMPLOS BASE (NO COPIAR TEXTUAL)
[Incluí aquí 3-4 ejemplos reales del corpus para calibrar estilo]
"""

# ---------------------------------------------------------------------
# GENERACIÓN
# ---------------------------------------------------------------------

if st.button("Generar historia clínica"):
    with st.spinner("Generando historia clínica..."):
        try:
            texto_usuario = " ".join([
                str(edad or ""), str(sexo or ""), str(patologia or ""), str(motivo or "")
            ]).strip()

            # Si el modo es libre, agregamos la restricción de patologías
            if "libre" in modo.lower():
                patologias_str = ", ".join(PATOLOGIAS_VALIDAS)
                restriccion_libre = f"En modo libre, el modelo debe elegir **solo una** patología del siguiente conjunto: {patologias_str}. No inventar patologías fuera de esta lista."
            else:
                restriccion_libre = ""

            # Selección según modo
            if modo.startswith("1️⃣") and any([patologia, motivo]):
                casos_relacionados = buscar_casos_similares(texto_usuario, top_k=3)
            else:
                casos_relacionados = casos_diversidad_media(embeddings, casos_index, top_k=3)

            few_shot_dinamico = "\n\n".join(
                [f"Ejemplo similar {i+1}:\n{txt}" for i, txt in enumerate(casos_relacionados)]
            )

            base_prompt = f"""
Actuás como un médico infectólogo uruguayo.

Debés redactar **una única historia clínica sintética**, verosímil y clínicamente coherente,
imitando el registro clínico hospitalario (telegráfico, abreviado, con jerga local y cierre propio de infectología).

**Modo:** {modo}
{f'Edad: {edad} años' if edad else 'Edad: a definir'}
{f'Sexo: {sexo}' if sexo else 'Sexo: a definir'}
{f'Patología: {patologia}' if patologia else 'Patología: a definir dentro de Infectología'}
{f'Motivo: {motivo}' if motivo else 'Motivo: a definir'}

---

### 🔒 Restricciones clínicas
1. Solo se permiten casos **de Infectología**.
   Si la patología o motivo **no pertenece a Infectología**, devolvé exactamente:
   "❌ Error: el modelo está diseñado solo para historias clínicas de Infectología."
2. El caso debe ser **coherente**, sin contradicciones.
3. Si el cuadro incluye patologías no infecciosas (fractura, migraña, hipotiroidismo, etc.), devolvé el mismo mensaje de error.
4. {restriccion_libre}

---

El texto debe sonar como una nota clínica auténtica de Infectología.
Puede usar o no encabezados, y no necesita que todas las secciones estén presentes.
El EA suele ocupar la mayor parte del texto.
Debe cerrar con hallazgos físicos abreviados y un cierre típico (“SP. VEA.” o “SP. Sin foco evidente. En seg. por infecto.”).
"""

# ---------------------------------------------------------------------
# FIN DEL CÓDIGO DEL PROMPT
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
# FIN DEL CÓDIGO DEL PROMPT
# ---------------------------------------------------------------------


            prompt_total = (
                f"{system_prompt}\n\n{few_shot}\n\n"
                f"### Casos del corpus recuperados por embeddings\n{few_shot_dinamico}\n\n"
                f"{base_prompt}"
            )

            model = genai.GenerativeModel(model_name="models/gemini-2.5-flash")
            response = model.generate_content(prompt_total)
            historia = response.text.strip()
            st.text_area("Historia clínica generada:", historia, height=420)

        except Exception as e:
            st.error(f"Error al generar la historia: {e}")

