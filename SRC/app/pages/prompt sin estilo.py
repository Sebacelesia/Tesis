#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import json
import requests
import random
import streamlit as st
import google.generativeai as genai


# =========================================================
# CONFIG
# =========================================================

OPENROUTER_KEY  = "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"
genai.configure(api_key="AIzaSyB4M-JF00FgQfPpSf0v-NqpDdm4bOKtgxc")

with open(r"C:\Users\59892\desktop\data\casos.json", "r", encoding="utf-8") as f:
    CASOS = json.load(f)

PATOLOGIAS_VALIDAS = [
    "VIH", "Tuberculosis pulmonar", "Tuberculosis ganglionar",
    "Meningitis bacteriana", "Endocarditis",
    "Infección de prótesis osteoarticular", "Celulitis",
    "Fiebre prolongada", "Neumonía adquirida en la comunidad",
    "Infección urinaria", "Candidemia",
    "Infección de sitio quirúrgico",
    "Sepsis de origen desconocido"
]


# =========================================================
# 1) LLAMADA AL MODELO GPT-OSS-20B (OpenRouter)
# =========================================================
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

    resp = requests.post(url, headers=headers, json=body)
    resp.raise_for_status()
    data = resp.json()

    return data["choices"][0]["message"]["content"].strip()


# =========================================================
# 2) SELECCIONAR 6 CASOS SEGÚN PATOLOGÍA
# =========================================================
def seleccionar_6_casos(patologia: str, casos: list):
    casos_txt = "\n\n".join(
        f"CASO {i+1}:\n{c}" for i, c in enumerate(casos)
    )

    prompt = f"""
Sos un infectólogo experto.

Quiero que analices los siguientes 20 casos clínicos completos y elijas los
**6 casos más similares clínicamente** a la patología:

➡️ {patologia}

Instrucciones:
1) Razoná libremente como infectólogo.
2) Compará síntomas, evolución, localización, mecanismo fisiopatológico y foco.
3) NO uses coincidencias de palabras sino coincidencias clínicas reales.
4) Podés escribir todo el análisis que quieras.
5) En la ÚLTIMA línea devolvé SOLO:

CASOS_ELEGIDOS: n1, n2, n3, n4, n5, n6

──────────────────────────
CASOS:
{casos_txt}
"""

    respuesta = call_model_openrouter(prompt)

    # Recuperar solo la línea final
    lineas = [l.strip() for l in respuesta.split("\n") if l.strip() != ""]
    ultima = lineas[-1]

    if not ultima.startswith("CASOS_ELEGIDOS:"):
        import re
        nums = re.findall(r"\b([1-9]|1[0-9]|20)\b", respuesta)
        return [int(x) for x in nums[:6]]

    numeros = ultima.replace("CASOS_ELEGIDOS:", "").strip()
    indices = [int(x) for x in numeros.split(",") if x.strip().isdigit()]

    return indices


# =========================================================
# 3) Elegir 3 al azar entre los 6 elegidos
# =========================================================
def elegir_3_random(indices_6):
    return random.sample(indices_6, 4)


# =========================================================
# 4) Generar few-shot con los 3 casos seleccionados
# =========================================================
def generar_fewshot(indices_3, patologia, casos):
    salida = [
        f"=====================",
        f"FEW SHOT – {patologia}",
        f"=====================",
        ""
    ]

    for idx in indices_3:
        texto = casos[idx - 1]
        salida.append(f"### CASO {idx}\n{texto}\n")

    return "\n".join(salida)


# =========================================================
# STREAMLIT UI
# =========================================================
st.title("🧫 Generador de Historia Clínica Sintética – Infectología")

modo = st.radio(
    "Modo de generación:",
    [
        "1️⃣ Manual (edad, sexo, patología, motivo)",
        "2️⃣ Patología y motivo libres (infecc.)",
        "3️⃣ Totalmente libre"
    ]
)

# =========================================================
# INPUTS SEGÚN MODO
# =========================================================
patologia_usuario = None
motivo = None


if modo.startswith("1️⃣"):
    # MODO 1: usuario define todo
    edad = st.number_input("Edad", min_value=0, max_value=120)
    sexo = st.selectbox("Sexo", ["Femenino", "Masculino", "Otro"])
    patologia_usuario = st.text_input("Patología")
    motivo = st.text_area("Motivo de consulta")

elif modo.startswith("2️⃣"):
    # MODO 2: usuario escribe motivo, patología se asigna aleatoria
    edad = None
    sexo = None
    patologia_usuario = None
    motivo = st.text_area("Motivo (Gemini lo puede ajustar)")

else:
    # MODO 3: totalmente libre → NO edad, NO sexo, NO motivo
    edad = None
    sexo = None
    patologia_usuario = None
    motivo = None
    st.write("(Modo totalmente libre: no se ingresa edad, sexo ni motivo.)")



# =========================================================
# BOTÓN
# =========================================================
if st.button("Generar historia clínica"):
    with st.spinner("Generando..."):

        # 1) Determinar patología final
        if modo.startswith("1️⃣"):
            patologia_final = patologia_usuario
        elif modo.startswith("2️⃣"):
            patologia_final = random.choice(PATOLOGIAS_VALIDAS)
        else:
            patologia_final = random.choice(PATOLOGIAS_VALIDAS)

        # 2) Selección automática de casos (GPT-OSS-20B)
        indices6 = seleccionar_6_casos(patologia_final, CASOS)
        indices3 = elegir_3_random(indices6)

        # 3) Construir FEW-SHOT
        few_shot = generar_fewshot(indices3, patologia_final, CASOS)

        # 4) PROMPT COMPLETO PARA GEMINI
        system_prompt = """
Sos un médico clínico experto en Infectología.

Generá una historia clínica nueva basada clínicamente en los casos del few-shot, sin copiar su estilo.
Mantené coherencia médica estricta: no inventes datos imposibles y no agregues información irrelevante.
Usá lenguaje clínico claro y estándar (NO estilo telegrafiado). El estilo final se aplicará después.

Usá solo las secciones habituales: SF (si aporta), AP, AQ, AEA, EA y EF. No inventes secciones nuevas.

Guía mínima de qué va en cada sección:

- SF: datos de contexto si son relevantes (edad, procedencia, convivencia). Opcional.
- AP: antecedentes médicos relevantes, comorbilidades, hábitos, medicaciones.
- AQ: cirugías o procedimientos previos importantes.
- AEA: evolución o controles previos relacionados al cuadro, si los hubiera.
- EA: motivo de consulta y evolución cronológica de los síntomas relatados por el paciente.

- EF: hallazgos constatados en el examen físico.
    • Poné los signos vitales (T°, FC, TA, FR, SatO₂) juntos y una sola vez.  
    • CV: solo ritmo, ruidos cardíacos, soplos, perfusión. NO la TA.  
    • PP: solo murmullos y ruidos agregados. NO FR ni SatO₂.  
    • Abdomen, piel, neurológico o foco local según corresponda.  
    • No repitas en el EF lo que ya está en la EA.

Fuera de esta guía mínima, tenés libertad total para redactar la historia clínica.


"""

        base_prompt = f"""
Edad: {edad} años
Sexo: {sexo}
Patología: {patologia_final}
Motivo: {motivo if motivo else "A definir por cuadro"}

Redactá una única historia clínica sintética, basada conceptualmente en los casos del few-shot, pero SIN copiar su estilo.  
Usá lenguaje clínico claro, completo y no telegrafiado.
"""
        with st.expander("📚 Few-shot seleccionado (casos similares):"):
            st.write(few_shot)


        prompt_total = (
            f"{system_prompt}\n\n"
            "### FEW SHOT (casos seleccionados por OSS)\n"
            f"{few_shot}\n\n"
            "---- FIN FEW SHOT ----\n\n"
            f"{base_prompt}"
        )

        # 5) Llamada a Gemini
        model = genai.GenerativeModel("models/gemini-2.5-flash")

        response = model.generate_content(
            prompt_total,
            generation_config=genai.types.GenerationConfig(
                temperature=0.15,
                top_p=0.7,
                top_k=40,
            ),
        )

        historia = response.text.strip()
        st.text_area("Historia clínica generada:", historia, height=420)

