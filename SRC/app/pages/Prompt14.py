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
            patologia_final = motivo
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
El inicio debe seguir el estilo de los few-shots seleccionados, pero sin rigidez: se permite cualquiera de las formas presentes en ellos (“SF:…”, “SM:…”, “Paciente…”, o inicio directo si así aparece en los few-shots).
Queda prohibido generar estructuras de inicio que no aparezcan en los few-shots. Queda expresamente prohibido: abreviar sexo (p. ej. “Paciente F”, “Paciente M”)
inventar estructuras nuevas no presentes en los few-shots.

No es obligatorio usar SF/SM; usarlos solo si están en los few-shots. PROHIBIDO especificar el sexo si ya ponemos SM o SF, que significa sexo masculino o femenino.
Si se usa SF o SM, está prohibido repetir sexo o edad en otra sección.

Edad y sexo solo deben incluirse si aparecen en los few-shots seleccionados (número o rango).
No inventar formatos nuevos.

Datos sociales:
Si el input del usuario aporta datos sociales, deben incluirse.
Si el input NO trae datos sociales, el modelo puede agregar un dato social mínimo opcional, usando únicamente patrones presentes en los few-shots (p. ej.: “procedente de Mdeo”, “vive sola”, “vive con familiar”, “independiente para ABVD”, “añosa frágil”).
Este dato no es obligatorio: debe usarse solo si es consistente con los few-shots seleccionados.

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

Inicio:
Usar solo inicios presentes en el corpus: “Hace X días…”, “Comienza…”, “Desde entonces…”, “Refiere…”, “Consulta por…”.

Cronología real:
Describir evolución temporal con conectores del corpus: “Inicia…”, “Agrega…”, “Concomitantemente…”, “A su vez…”, “Posteriormente…”.

Caracterización:
Describir tipo, intensidad, frecuencia y progresión del síntoma (tos seca irritativa, productiva mucopurulenta, vespertina, progresiva).
Frases cortas, telegráficas.

Negativos relevantes:
Incluir solo los negativos útiles para orientar el cuadro, integrados en la narrativa (dolor torácico, hemoptisis, cefalea, vómitos, diarrea, disuria, lesiones cutáneas).
Evitar formato checklist.

Estilo:
Telegráfico, fragmentado.
Usar conectores y expresiones reales del corpus (“astenia”, “adinamia”, “chuchos”, “sensación febril”).
Sin tecnicismos que no aparezcan en los pocos-shots.

Restricciones:
No incluir hallazgos físicos ni diagnósticos.

Densidad:
Detalles suficientes, con síntomas caracterizados y cronología clara. No minimalista, no académico.

7)  EF (Examen Físico) 
El EF es telegráfico, fragmentado y clínicamente coherente, usando solo los sistemas que aparecen en los few-shots y con su misma extensión.

El EF no se divide en un bloque de “signos vitales” separado.
Los parámetros vitales se integran dentro del sistema correspondiente, sin duplicaciones.
Queda prohibido listar Tax/FC/PA/FR/Sat por separado cuando luego se describen los sistemas CV y PP.

Pantallazo inicial:
Opcional, solo si aparece en los few-shots seleccionados.
Debe ser breve y no puede contener valores numéricos ni parámetros que luego correspondan a CV o PP (ej.: “Lucido. Apirético. Eupneico.”).
No sustituye a ningún sistema del EF.

Sistema PyM:
Debe seguir estrictamente el orden:
coloración → hidratación → perfusión → pliegue/TR → lesiones.
Queda prohibido incluir dispositivos, catéteres o abordajes quirúrgicos en PyM.

Sistema Cardiovascular (CV):
Incluye exclusivamente: RR, ruidos, soplos, sincronía y los parámetros PA y FC.
Si PA o FC ya aparecieron fuera de CV en el pantallazo inicial, queda prohibido repetirlos.
Nunca colocar hallazgos cardiovasculares fuera de CV.

Sistema Pleuropulmonar (PP):
Incluye exclusivamente: FR, Sat/VEA, MAV y ruidos respiratorios (estertores o sibilancias).
Si FR o Sat ya aparecieron fuera de PP en el pantallazo inicial, queda prohibido repetirlos.
Nunca colocar FR, Sat ni hallazgos respiratorios fuera de PP.
No usar “AA”; la saturación debe escribirse únicamente como “VEA”.

Sistema Abdomen:
Debe describirse exclusivamente como “ABD” o “Abdomen”.
Queda prohibido usar “OA” para abdomen.
OA se reserva solo para hallazgos osteoarticulares o abordajes quirúrgicos en miembros.

PNM / PPCC:
Conciencia → pupilas → rigidez de nuca, siguiendo el orden del corpus.
No mezclar parámetros con otros sistemas.

Coherencia general:
Ningún sistema puede contradecir parámetros vitales o la EA.
No describir estabilidad si hay datos de gravedad.
No usar “eupneico” por defecto; solo si aparece en los few-shots en un contexto equivalente.
No repetir ningún parámetro numérico en más de un sistema.

Extensión y cierre:
La extensión del EF debe estar dentro del rango del few-shot seleccionado, no igualarlo.
Si los few-shots no incluyen cierre, queda PROHIBIDO generarlo.
Si lo incluyen, las únicas formas admitidas son:
“SP.”, “Resto SP.”, “SP. VEA.”, o “Sin foco evidente.”
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
{f'Datos del paciente: {motivo}' if motivo else 'Datos del paciente: a definir'}

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
                temperature=0.25,
                top_p=0.85,
                top_k=20,
            ),
        )

        historia = response.text.strip()
        st.text_area("Historia clínica generada:", historia, height=420)
        st.text_area("Few shots tenidos en cuenta:", few_shot, height=420)