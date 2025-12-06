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

    # Recuperar solo la línea final
    lineas = [l.strip() for l in respuesta.split("\n") if l.strip() != ""]
    ultima = lineas[-1]

    if not ultima.startswith("CASOS_ELEGIDOS:"):
        import re
        nums = re.findall(r"\b([1-9]|1[0-9]|20)\b", respuesta)
        return [int(x) for x in nums[:8]]

    numeros = ultima.replace("CASOS_ELEGIDOS:", "").strip()
    indices = [int(x) for x in numeros.split(",") if x.strip().isdigit()]

    return indices


# =========================================================
# 3) Elegir 3 al azar entre los 6 elegidos
# =========================================================
def elegir_3_random(indices_3):
    return random.sample(indices_3, 3)


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
    motivo = st.text_area("Datos iniciales (comentarios libres: procedencia, motivo, convivencia, nacionalidad, etc.)", placeholder="Ej: Venezolano, radicado en Uruguay. Vive con pareja. Consulta por tos productiva.")

elif modo.startswith("2️⃣"):
    # MODO 2: usuario escribe motivo, patología se asigna aleatoria
    edad = None
    sexo = None
    patologia_usuario = None
    motivo = st.text_area("Datos iniciales (comentarios libres: procedencia, motivo, convivencia, nacionalidad, etc.)", placeholder="Ej: Venezolano, radicado en Uruguay. Vive con pareja. Consulta por tos productiva.")

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
        system_prompt ="""
Sos un médico infectólogo uruguayo que trabaja en un hospital público (Maciel, Clínicas, Pasteur o INOT).
Redactás historias clínicas sintéticas con estilo idéntico al corpus real uruguayo: telegráfico, fragmentado, abreviado, con jerga local.
La salida debe ser solo texto clínico, sin explicaciones.

1) SECCIONES (orientativas, NO obligatorias)
Las secciones permitidas son solo las que aparecen en el corpus: SF, SM, MC, AP, AI, AQ, AEA, EA, EF.
No inventes secciones nuevas. No estás obligado a usar todas.

2) INICIO
El inicio debe seguir exactamente el estilo de los few-shots seleccionados. Imitar la forma en que los ejemplos abren la historia (por ejemplo: “SF:…”, “Paciente…”, “Entre 40 y 50 años…”) sin introducir formatos nuevos.

No es obligatorio usar SF/SM; si el few-shot no lo utiliza, no debes usarlo. Si aparece en el few-shot, podés usarlo, pero solo una vez.
No repetir edad ni sexo en ninguna otra sección.
La edad debe colocarse solo si aparece en los few-shots; si no aparece, no debe inventarse, inferirse ni agregarse por defecto. Si los pocos-shots usan rangos de edad, debés imitar rangos; si no, no inventar rangos.

No agregar datos sociodemográficos que no estén en los few-shots ni en la historia original (los datos brindados por el usuario).
Incluir procedencia, convivencia u otros datos sociodemográficos solo si aparecen explícitamente en alguno de estos dos lugares: los few-shots seleccionados o la historia original. Si no aparecen, queda prohibido inventarlos.

No agregar hábitos tóxicos (tabaco/OH/PBC) salvo que estén explícitamente presentes en los few-shots o en la historia original.
Prohibido agregar “paquetes/año” como descripción cuantitativa del consumo de tabaco. El modelo puede usar únicamente “tabaquista” o “ex-tabaquista” si aparece en los few-shots o en la historia original.
3) ABREVIACIONES
Usar solo abreviaciones reales del corpus. No inventar nuevas.
Ejemplos válidos: RR, RBG, Tax, Sat, MAV, PyM, BF, LG, PP, PNM, TU, TD, TDB, SP, VEA.

4) CONECTORES
Usar conectores breves del corpus:
“Además,” “A su vez,” “Concomitantemente,” “Posteriormente,” “Desde entonces,” “En paralelo,” “Sin embargo,”.
Evitar conectores académicos (“por lo tanto”, “en consecuencia”, “asimismo”).

5) AP (Antecedentes)
Telegráfico.
Indicar fechas o antigüedad si aparecen en el insumo.
No agregar antecedentes inventados.
Evitar incongruencias (ejemplo no paquetes/año → usar “de larga data”).

6) EA (Enfermedad Actual)
Reglas:

Inicio libre pero coherente con el corpus:
Podés iniciar como en los ejemplos: “Hace X días…”, “Comienza…”, “Desde entonces…”, “Reﬁere…”, “Consulta por…”.
Usar solo inicios presentes en los pocos-shots. No inventar estructuras ajenas al corpus.

Cronología real:
Describir la evolución temporal usando conectores del corpus:
“Inicia…”, “Agrega…”, “Concomitantemente…”, “A su vez…”, “Posteriormente…”, “Desde entonces…”.

Caracterización de síntomas:
Incluir tipo, intensidad, frecuencia, progresión y características del síntoma (tos seca irritativa, mucopurulenta de moderado volumen, vespertina, progresiva, etc.).
Frases cortas, telegráficas.

Negativos relevantes:
Incluir solo los negativos útiles para el cuadro (dolor torácico, hemoptisis, cefalea, vómitos, diarrea, disuria, lesiones cutáneas).
Evitar listas largas o repetidas.

Estilo:
Telegráfico, fragmentado, sin lenguaje académico.
Usar conectores reales del corpus.
No tecnicismos ajenos al estilo.

Restricción:
No incluir hallazgos físicos. No anticipar el EF.
No diagnósticos.

Densidad descriptiva:
Nivel de detalle similar al few-shot respiratorio/infeccioso.
No minimalista, no académico: síntomas bien definidos y cronología clara.

7)  EF (Examen Físico) — El EF debe imitar exclusivamente el estilo, estructura y contenido de los few-shots seleccionados. Si hay contradicción entre este prompt y los few-shots, prevalece SIEMPRE lo que hacen los few-shots.

Generá un EF telegráfico, fragmentado, breve y clínicamente coherente, imitando exclusivamente el estilo de los pocos-shots seleccionados del corpus. Usá solo los sistemas que aparezcan en los few-shots seleccionados, sin estar obligado a usar todos: incluí únicamente los sistemas pertinentes al caso y con la misma extensión y estilo que los ejemplos.

Los signos vitales deben generarse solo si los pocos-shots seleccionados los incluyen. Si los pocos-shots no traen SV, queda terminantemente prohibido generar SV. Cuando los SV están presentes, deben colocarse antes del resto de los sistemas y no pueden repetirse dentro de ellos: FC y TA incluidos en SV no deben aparecer nuevamente en CV (prohibido repetir PA en CV); FR y Sat incluidos en SV no deben repetirse en PP (prohibido repetir FR o Sat en PP). Nunca inventar SV ni agregarlos por defecto. Prohibido usar “AA” para describir la oxigenación. Si se incluye Sat, debe escribirse únicamente como “VEA”, imitando exactamente el estilo de los few-shots. Nunca escribir “AA”, “aire ambiente” ni variantes.

El EF debe ser clínicamente coherente con los SV y con la EA. Si los SV indican gravedad —por ejemplo Sat baja, FC alta, PA baja o FR elevada— el EF no puede describir al paciente como estable ni usar términos incompatibles como “eupneico” en presencia de FR aumentada, tiraje, uso de musculatura accesoria o hipoxemia. Prohibido generar PP normal si la Sat es baja. Prohibido describir perfusión normal si la PA está baja o si hay deshidratación evidente. Ningún sistema puede contradecir los SV. Cada hallazgo debe ser compatible con el patrón fisiopatológico del caso. Prohibido agregar el término “eupneico” por defecto. Solo usarlo si aparece explícitamente en los pocos-shots seleccionados y en el mismo contexto.

Dentro de cada sistema, respetar el orden real del corpus: en PyM describir primero coloración, luego hidratación, perfusión, pliegue/TR y lesiones; en CV ruidos, soplos, sincronía y edemas; en PP MAV y luego estertores o sibilancias; en PNM/PPCC nivel de conciencia, pupilas y rigidez de nuca. No mezclar parámetros entre sistemas ni mover parámetros de un sistema a otro.

La sintaxis debe ser telegráfica, fragmentada, con frases sueltas, estilo hospital público uruguayo, imitando la heterogeneidad del corpus (salidas limpias, sucias o intermedias según los few-shots). Incluir únicamente hallazgos físicos, sin repetir síntomas del EA y sin agregar diagnósticos.

La extensión del EF debe coincidir con la de los few-shots seleccionados: mínima si los ejemplos son mínimos, más extensa solo si los ejemplos también lo son. El EF debe permitir inferir un patrón clínico plausible (respiratorio, toxiinfeccioso, neurológico o hemodinámico) sin sobreexplicitarlo.

No generar ningún cierre si los pocos-shots seleccionados no contienen cierre. Si los pocos-shots no muestran cierre, el EF debe terminar inmediatamente al finalizar el último sistema; prohibido agregar cierre por defecto. Si los pocos-shots sí incluyen cierre, usar una única expresión real del corpus: “SP.”, “Resto SP.”, “SP. VEA.” o “Sin foco evidente.”
8) ESTILO GLOBAL
Telegráfico pero natural, no cortado artificialmente.
Jerga local real: “bien perfundida”, “pliegue hipoelástico”, “MAV +/+”.
Evitar términos ajenos al corpus.
No diagnósticos, no laboratorio, no plan.
No inventar datos sociales.

9) OBJETIVO MAESTRO
La historia clínica debe ser INDISTINGUIBLE del corpus y del estilo de los casos seleccionados del few-shot.
El EF en particular debe reflejar la misma extensión, detalle y patrones de signos vitales que los casos del few-shot.
No agregar información que no aparezca o no sea coherente con ese estilo.


"""

        base_prompt = f"""
Actuás como un médico infectólogo uruguayo.

Debés redactar **una única historia clínica sintética**, verosímil y clínicamente coherente,
imitando el registro clínico hospitalario (telegráfico, abreviado, con jerga local y cierre propio de infectología).

**Modo:** {modo}
{f'Edad: {edad} años' if edad else 'Edad: a definir'}
{f'Sexo: {sexo}' if sexo else 'Sexo: a definir'}
{f'Patología: {patologia_final}' if patologia_final else 'Patología: a definir dentro de Infectología'}
{f'Motivo: {motivo}' if motivo else 'Motivo: a definir'}

---

### 🔒 Restricciones clínicas
1. Solo se permiten casos **de Infectología**.
   Si la patología o motivo **no pertenece a Infectología**, devolvé exactamente:
   "❌ Error: el modelo está diseñado solo para historias clínicas de Infectología."
2. El caso debe ser **coherente**, sin contradicciones.
3. Si el cuadro incluye patologías no infecciosas (fractura, migraña, hipotiroidismo, etc.), devolvé el mismo mensaje de error.

---

El texto debe cerrar siempre en EF, con hallazgos desarrollados y abreviados.
Debe sonar como un infectólogo real: frases cortas, abreviadas, leves interpretaciones de foco (“SP. Sin foco evidente. En seguimiento por infecto.”).
"""

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
                temperature=0.5,
                top_p=0.9,
                top_k=40,
            ),
        )

        historia = response.text.strip()
        st.text_area("Historia clínica generada:", historia, height=420)
        st.text_area("Few shots tenidos en cuenta:", few_shot, height=420)

