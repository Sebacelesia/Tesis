#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# ---------------------------------------------------------------------
# 🧫 GENERADOR DE HISTORIAS CLÍNICAS SINTÉTICAS – INFECTOLOGÍA
# Versión optimizada para GEMINI 2.5 FLASH
# Con few-shot estático (ancla de estilo) + dinámico (por embeddings)
# ---------------------------------------------------------------------

import streamlit as st
import google.generativeai as genai
import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import os

# ---------------------------------------------------------------------
# CONFIGURACIÓN INICIAL
# ---------------------------------------------------------------------
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
st.title("🧫 Generador de Historia Clínica Sintética – Infectología (Gemini 2.5 Flash)")

# Rutas locales
EMB_PATH = r"C:\Users\59892\desktop\data\embeddings.npy"
CASES_PATH = r"C:\Users\59892\desktop\data\casos.json"

# Cargar embeddings y casos
embeddings = np.load(EMB_PATH)
with open(CASES_PATH, "r", encoding="utf-8") as f:
    casos_index = json.load(f)

modelo_embed = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ---------------------------------------------------------------------
# FUNCIONES AUXILIARES
# ---------------------------------------------------------------------
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
# INTERFAZ DE USUARIO
# ---------------------------------------------------------------------
modo = st.radio(
    "Seleccioná el modo de generación:",
    [
        "1️⃣ Ingreso manual (Patología y Motivo determinados por el usuario)",
        "2️⃣ Patología y Motivo libres (el modelo los elige dentro de Infectología)",
        "3️⃣ Todo libre (edad, sexo, patología y motivo elegidos por el modelo)",
    ],
)

# Control de temperatura / estilo
st.subheader("🧠 Estilo de redacción (Temperatura del modelo)")
temperatura = st.radio(
    "Seleccioná el estilo de escritura:",
    [
        "1️⃣ Por defecto (temperatura 0.5)",
        "2️⃣ Realista hospitalario (temperatura 0.7 - por defecto)",
        "3️⃣ Guardia caótica (temperatura 1.1)"
    ],
    index=1
)

if "Por defecto" in temperatura:
    temp_value = 0.7
elif "Guardia" in temperatura:
    temp_value = 1.1
else:
    temp_value = 0.9

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
# LISTA DE PATOLOGÍAS PERMITIDAS
# ---------------------------------------------------------------------
PATOLOGIAS_VALIDAS = [
    "VIH avanzado o controlado", "tuberculosis pulmonar o extrapulmonar",
    "neumonía adquirida en la comunidad", "meningitis criptocócica",
    "toxoplasmosis cerebral", "citomegalovirus (CMV)",
    "candidiasis mucocutánea o esofágica", "endocarditis infecciosa",
    "bacteriemia o sepsis", "fiebre prolongada o de origen infeccioso",
    "pielonefritis", "celulitis", "osteomielitis", "espondilodiscitis",
    "infección de sitio quirúrgico", "infección protésica o postoperatoria",
    "infección fúngica invasiva", "infección por Pseudomonas aeruginosa",
    "infección por Acinetobacter baumannii"
]

# ---------------------------------------------------------------------
# GENERACIÓN
# ---------------------------------------------------------------------
if st.button("Generar historia clínica"):
    with st.spinner("Generando historia clínica..."):
        try:
            # Texto base del usuario
            texto_usuario = " ".join([
                str(edad or ""), str(sexo or ""), str(patologia or ""), str(motivo or "")
            ]).strip()

            # Casos recuperados (embeddings o diversidad media)
            if modo.startswith("1️⃣") and any([patologia, motivo]):
                casos_relacionados = buscar_casos_similares(texto_usuario, top_k=3)
            else:
                casos_relacionados = casos_diversidad_media(embeddings, casos_index, top_k=3)

            # FEW-SHOT DINÁMICO
            few_shot_dinamico = "\n\n".join(casos_relacionados)

            # FEW-SHOT ESTÁTICO – ejemplos reales del PDF
            few_shot_estatico = """
M40a VIH+. Dx 10a, abandono TARV 2a. No LT CD4 CV. Tabaquista. HTA irreg.
Ingresa por sd febril prolongado 3s ev. Fiebre vespertina 39° sudoración noct. Tos seca persist. Disnea CF II-III.
EF: EG desmejorado, adelgazado. FR28 Sat88%AA MAV ↓ bil. Crepitantes bases. SP. Sin foco evidente. En seg.

F67a DM2 HTA sonda vesical. Sd febril 48h ev escalofríos disuria dolor lumbar. EF: EG conservado. CV RR FC102.
PP FR20 Sat96%AA Abd blando dolor lumbar +. SP. En seg.

SF entre 50 y 60 a. AP VIH Dx 10a abandono TARV 1a. Último control 30 CD4 CV>5M copias.
Ingresa por tos productiva 2m ev, dolor torácico puntada de lado, astenia y sudoración noct.
EF: Lucido apirético PyM piel fina desnutrición marcada PP MAV+/+ sibilancias y crepitantes difusos. SP. Sin foco evidente. En seg.
"""

            # SYSTEM PROMPT OPTIMIZADO
            system_prompt = """
Sos un médico infectólogo uruguayo de hospital público (H. Maciel, Clínicas, Pasteur, INOT).
Redactás historias clínicas sintéticas con fines académicos, imitando el registro real de sala:
telegráfico, fragmentado, con abreviaciones locales y tono de guardia.

Tu escritura debe parecer una evolución hospitalaria escrita apurada:
- Frases cortas o incompletas.
- Puntuación irregular o ausente.
- Abreviaciones locales (EG, RR, FC, FR, Tax, Sat, MAV, SP, VEA, IO, TARV, etc.).
- Alterná mayúsculas y minúsculas sin patrón.
- Errores menores, omisiones y repeticiones leves son correctos.
- No busques claridad ni limpieza: si parece desprolijo, está bien.
- No limpies ni corrijas gramática o abreviaciones.
- No uses markdown.
- Si dudás entre limpio o sucio, elegí sucio.
- Si hay contradicciones leves, no las corrijas.
- No agregues diagnóstico ni plan.
- No incluyas laboratorio.
- El texto debe cerrar con frase típica: "SP. Sin foco evidente." o "SP. VEA." o "SP. Sin foco evidente. En seg. por infecto."
Imitá exactamente el estilo, puntuación y formato de los textos previos.
"""

            # BASE PROMPT – datos del caso y restricciones
            base_prompt = f"""
Actuás como un médico infectólogo uruguayo y debés redactar **una única historia clínica sintética**
verosímil y clínicamente coherente, en tono hospitalario realista.

**Modo:** {modo}
{f'Edad: {edad} años' if edad else 'Edad: a definir'}
{f'Sexo: {sexo}' if sexo else 'Sexo: a definir'}
{f'Patología: {patologia}' if patologia else 'Patología: a definir dentro de Infectología'}
{f'Motivo: {motivo}' if motivo else 'Motivo: a definir'}

---

### 🔒 Restricciones clínicas
1. Solo se permiten casos de Infectología.
   Si el caso no pertenece a Infectología, devolvé:
   "❌ Error: el modelo está diseñado solo para historias clínicas de Infectología."
2. El texto debe ser clínicamente verosímil aunque pueda tener omisiones o desorden.
3. No generar más de una historia; no numerar ni usar viñetas.
4. Si el modo es libre, elegí la patología solo dentro de Infectología.
5. Cerrá con un EF abreviado y cierre típico ("SP. Sin foco evidente." o "SP. VEA.").

El texto final debe parecer una nota clínica escrita en guardia:
taquigráfica, irregular y con jerga local uruguaya.
"""

            # ---------------------------------------------------------------------
            # ORDEN ÓPTIMO DEL PROMPT
            # ---------------------------------------------------------------------
            prompt_total = (
                f"{few_shot_estatico}\n\n"
                f"{few_shot_dinamico}\n\n"
                f"{system_prompt}\n\n"
                f"{base_prompt}"
            )

            # ---------------------------------------------------------------------
            # LLAMADA AL MODELO
            # ---------------------------------------------------------------------
            model = genai.GenerativeModel(model_name="models/gemini-2.5-flash")
            response = model.generate_content(
                prompt_total,
                generation_config={"temperature": temp_value}
            )

            historia = response.text.strip()
            st.text_area("Historia clínica generada:", historia, height=420)

        except Exception as e:
            st.error(f"Error al generar la historia: {e}")

