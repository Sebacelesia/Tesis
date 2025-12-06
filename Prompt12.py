#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8
# ---------------------------------------------------------------------
# 🧫 GENERADOR DE HISTORIAS CLÍNICAS SINTÉTICAS – INFECTOLOGÍA
# Estilo hospitalario uruguayo real, calibrado para Gemini 2.5 Flash
# ---------------------------------------------------------------------

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
# PATOLOGÍAS PERMITIDAS
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
# GENERACIÓN Y EVALUACIÓN
# ---------------------------------------------------------------------
if st.button("Generar y evaluar historias clínicas"):
    with st.spinner("Generando historias clínicas..."):
        texto_usuario = " ".join([
            str(edad or ""), str(sexo or ""), str(patologia or ""), str(motivo or "")
        ]).strip()

        if modo.startswith("1️⃣") and any([patologia, motivo]):
            casos_relacionados = buscar_casos_similares(texto_usuario, top_k=3)
        else:
            casos_relacionados = casos_diversidad_media(embeddings, casos_index, top_k=3)

        few_shot_dinamico = "\n\n".join(casos_relacionados)

        # ---------------------------------------------------------------------
        # FEW-SHOT ESTÁTICO (tres ejemplos calibrados)
        # ---------------------------------------------------------------------
        few_shot_estatico = """
Pte masculino 48 años.
AP: VIH dx hace 12a, TARV c/Biktarvy ref abandono hace aprox 1a. Últ control CD4/CV desconocido. Tabaquista 20 p/y. Ex-udiv último uso hace 6 meses. Niega otros AP.
EA: Consulta por sd febril prolongado 3 semanas ev, picos hasta 39°C. Acompaña astenia marcada anorexia pérdida de peso no cuantificada. Desde 1 semana agrega tos seca no productiva, disnea de pequeños esfuerzos progresiva, ahora en reposo. Niega dolor torácico. Ref cefalea ocasional. Familiares notan confusión y desorientación espacial los últimos 2d. Niega otra sintomatología.
EF: Lúcido desorientado a predominio temporo-espacial, confuso. Mal EG. Pac hipoactivo, febril Tax 38.7°C. PyM hipocoloreadas delgadez extrema. Candidiasis oral extensa placa blanquecina no desprendible, afectando paladar y mucosa yugal. Adenopatias laterocervicales bilaterales móviles no dolorosas, de 1cm aprox. CV RR FC110lpm soplo sistólico 2/6 foco mitral. PA 90/60. PP FR28 taquipneico Sat88% AA. MAV↓ bases crepitantes finos bibasales. Abd blando depresible indoloro. PNM rigidez nuca dudosa sin foco motor. SP. En seg. por infecto.

SF 46 años ingresa por cuadro respiratorio.
AP: VIH Dx 2005, no adhiere TARV, ref múltiples abandonos, último control >1a sin CD4 ni CV. Internac previas por neumonía en Clínicas hace 3a, alta c/tto domic s/seguim. Ref otras patologías pero no recuerda, niega aq relev.
EA: Cuadro 2sem ev caracterizado x tos seca q progresa a tos productiva c/expectoración mucopurulenta abundante. Disnea esf prog actual CFIII-IV, tolera mínimos esf. Sudorac noct, astenia marcada, adinamia, pérdida peso no cuantif. Refiere registros febriles no cuantif. Niega dolor torácico.
EF: Pac lúcida orientada 3 esferas. Mal EG, adelgazada, palidez cut-muc, desnutrición proteico-calórica marcada. Apirética al examen. CV FC115 RR RBG s/soplos. PA100/60. PP FR28 tiraje intercostal leve MAV abolido base izq↓ base der estertores secos y crepit finos bibasales. Sat74% AA uso musculatura accesoria. PNM GCS15 s/foco. Abd blando depres indoloro s/visceromeg. MI s/edemas. PyM lesiones blanquecinas mucosa oral (candidiasis). SP. En seg x infecto.

Pcte F 58a VIH Dx 10a abandono TARV 1a. Últ control CD4 30 CV>5M cop. Niega oportunistas previas. Tabaq exOH.
EA: Tos productiva 2m ev inicio insidioso mucopurulenta ocas hemoptoica. Disnea prog inicio gr esf ahora mínimos. Dolor torácico tipo puntada Lado der aumenta c/tos e insp prof. Astenia marcada adinamia anorexia pérdida peso no cuantif ult meses. Sudorac noct profusa. Niega registros febriles pero refiere sens febril vespertina. Sin otra sintomatología.
EF: Lúcida Glasgow15 apirética Tax36.8 FR22 FC98 PA100/60 Sat90% AA. PyM piel fina deshidratación leve marcada desnutrición proteico-calórica. Mucosas secas. CV RR RBG s/soplos. PP taquipneica MAV global↓ base der crepit finos difusos sibilancias esp bilat. Abd blando depres indoloro H no palpable RHA+. PNM sin focalidad. Pupilas isoc reactivas. SP. En seg x infecto.
"""

        # ---------------------------------------------------------------------
        # SYSTEM PROMPT + EXTRA PROMPT
        # ---------------------------------------------------------------------
        system_prompt = """
Sos un médico infectólogo uruguayo de hospital público (Maciel, Clínicas, Pasteur, INOT).
Redactás historias clínicas hospitalarias con estilo telegráfico, continuo y realista.
Usá abreviaciones locales (EA, EF, AP, SP, EG, RR, FC, FR, Tax, Sat, MAV, TARV, etc.).
Permití irregularidades tipográficas y mezcla de mayúsculas, pero no dentro de palabras.
No inventes encabezados fuera de EA, EF, AP, SP.
No uses tono narrativo ni formato prolijo.
No incluyas laboratorio, diagnóstico ni plan.
Terminá con cierre típico: "SP. Sin foco evidente." o "SP. En seg. por infecto."
"""

        extra_prompt = """
El texto debe sonar a evolución real de guardia o ingreso hospitalario, escrito apurado.
Frases encadenadas, errores leves, abreviaciones acortadas, puntuación irregular.
Simulá el "ensucie hospitalario real": alterná densidad, errores, frases telegráficas.
No corrijas ni limpies: debe conservar la textura caótica pero legible de historias reales.
"""

        # ---------------------------------------------------------------------
        # PROMPT BASE
        # ---------------------------------------------------------------------
        base_prompt = f"""
Actuás como un médico infectólogo uruguayo y debés redactar una historia clínica hospitalaria completa,
verosímil y clínicamente coherente, con estilo telegráfico y tono hospitalario real.

Modo: {modo}
{f'Edad: {edad} años' if edad else 'Edad: a definir'}
{f'Sexo: {sexo}' if sexo else 'Sexo: a definir'}
{f'Patología: {patologia}' if patologia else 'Patología: a definir dentro de Infectología'}
{f'Motivo: {motivo}' if motivo else 'Motivo: a definir'}
"""

        # ---------------------------------------------------------------------
        # ENSAMBLAJE FINAL
        # ---------------------------------------------------------------------
        prompt_total = (
            f"{base_prompt}\n\n"
            f"{few_shot_estatico}\n\n"
            f"{few_shot_dinamico}\n\n"
            f"{system_prompt}\n\n"
            f"{extra_prompt}\n\n"
            "Recordá mantener el estilo hospitalario real con abreviaciones y errores humanos visibles."
        )

        model = genai.GenerativeModel(model_name="models/gemini-2.5-flash")

# ---------------------------------------------------------------------
# MULTITEMPERATURA + EVALUADOR (solo devuelve la mejor historia)
# ---------------------------------------------------------------------
temperaturas = [0.5, 0.8, 1.1]
historias = {}

with st.spinner("Generando historias clínicas con distintas temperaturas..."):
    for temp in temperaturas:
        resp = model.generate_content(
            prompt_total,
            generation_config={"temperature": temp}
        )
        historias[temp] = resp.text.strip()

# ---------------------------------------------------------------------
# EVALUADOR DE ESTILO (elige la mejor, sin mostrar comentarios)
# ---------------------------------------------------------------------
evaluador_prompt = f"""
Sos infectólogo uruguayo de hospital público.
Reconocés el dialecto hospitalario real: abreviaciones, frases apuradas,
errores leves y estructura AP-EA-EF-SP.

Ejemplos reales de ese estilo:

{few_shot_estatico}

{few_shot_dinamico}

Analizá las siguientes versiones generadas y elegí cuál reproduce mejor
el dialecto hospitalario uruguayo. 
Solo devolvé el texto completo de la mejor versión, sin comentarios,
sin puntajes y sin ninguna otra explicación.
"""

bloque_versiones = ""
for temp, texto in historias.items():
    bloque_versiones += f"\n\nVersión {temp}:\n{texto}\n"

with st.spinner("Seleccionando la versión más fiel al dialecto hospitalario..."):
    evaluacion = model.generate_content(
        f"{evaluador_prompt}\n{bloque_versiones}",
        generation_config={"temperature": 0.3}
    )

# ---------------------------------------------------------------------
# MOSTRAR SOLO LA HISTORIA FINAL ELEGIDA
# ---------------------------------------------------------------------
st.subheader("🩺 Historia clínica final (dialecto hospitalario más realista):")
st.text_area("", evaluacion.text.strip(), height=420)


