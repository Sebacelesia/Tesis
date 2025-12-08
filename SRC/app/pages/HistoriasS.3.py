#!/usr/bin/env python
# coding: utf-8

# In[9]:


import streamlit as st
import google.generativeai as genai



genai.configure(api_key="AIzaSyB4M-JF00FgQfPpSf0v-NqpDdm4bOKtgxc")

st.title("🧫 Generador de Historia Clínica Sintética – Infectología (Gemini 2.5 Flash)")


modo = st.radio(
    "Seleccioná el modo de generación:",
    [
        "1️⃣ Ingreso manual (Patología y Motivo de consulta determinados por el usuario)",
        "2️⃣ Patología y Motivo libres (el modelo los elige con coherencia médica dentro de Infectología)",
        "3️⃣ Todo libre (edad, sexo, patología y motivo elegidos por el modelo)",
    ],
)

# --- ENTRADAS SEGÚN MODO ---
edad = None
sexo = None
patologia = None
motivo = None

if modo == "1️⃣ Ingreso manual (Patología y Motivo de consulta determinados por el usuario)":
    edad = st.number_input("Edad", min_value=0, max_value=125, step=1)
    sexo = st.selectbox("Sexo", ["Femenino", "Masculino", "Otro", "Prefiero no decir"])
    patologia = st.text_input("Patología (obligatoria)", placeholder="Ej: VIH, tuberculosis, meningitis por H. influenzae, etc.")
    motivo = st.text_area("Motivo de consulta (opcional)", height=100, placeholder="Ej: fiebre prolongada, tos con expectoración...")

elif modo == "2️⃣ Patología y Motivo libres (el modelo los elige con coherencia médica dentro de Infectología)":
    edad = st.number_input("Edad", min_value=0, max_value=125, step=1)
    sexo = st.selectbox("Sexo", ["Femenino", "Masculino", "Otro", "Prefiero no decir"])

system_prompt = """
Sos un médico infectólogo uruguayo que trabaja en un hospital público.
Redactás historias clínicas sintéticas con fines académicos, con sintaxis telegráfica,
fragmentada e impersonal, imitando fielmente el registro médico rioplatense.
No explicás tus decisiones ni agregás comentarios: solo generás la historia clínica.
Tu salida debe ser exclusivamente el texto clínico, sin encabezados, sin markdown,
sin títulos ni aclaraciones. Siempre cerrás en EF **con hallazgos desarrollados**:
nunca cerrar en la palabra “EF” sin contenido.
"""
    
few_shot = """
### 🧪 EJEMPLOS DE REFERENCIA (NO REPRODUCIR, SOLO IMITAR EL ESTILO)

Ejemplo 1:
SF: Entre 40 y 50 años. Procedente de Mdeo. Vive con madre e hijos. Primaria completa. -HIV diagnosticado en 2005 y 2015, no adhiere TARV. Estado inmune 04.2023: CD4 entre 130 y 180 cel/ml, CV: En torno a 300.000 -BK diseminado múltiples abandonos de ttos. Última vez que abandonó en enero de 2023. Tuberculosis miliar en 03/22 con compromiso ganglionar, biopsia Xpert y cultivo positivos para M. tuberculosis abandona tto. -Ingreso 07/22 por planteo de sífilis meningovascular, recibe tto 21 días con penicilina cristalina con VDRL al alta 1/32. -Ingreso el 01/23 a H de Clínicas por BK ganglionar + neumonitis a Sars Cov2. De dicha internación VDRL + que se trata con 3 dosis de benzetacil. Abandona tto anti BK al alta. -Meningitis a H. Influenzae 05/23 diagnosticado por filmarray, recibe Ceftriaxona 14 días. -Planteo por citología de LNH - sin enfermedad hematológica demostrada - Policonsumo, PBC, en abstinencia desde el ingreso. -No alergias a medicamentos. EA: Consulta por tos y expectoración mucopurulenta de 2 semanas, disnea de esfuerzo, anorexia. EF: Severa desnutrición proteico-calórica. BF: Muguet oral. CV: RR 90 cpm, sin soplos, sincronico. MAV abolido, estertores secos en bases pulmonares, SAT 74%.

Ejemplo 2:
Paciente, 70 a 75 años, procedente de Carmelo, vive sola, independiente para ABVD. Jubilada. AP Asma en tratamiento durante CBO. Arritmia en tratamiento con Diltiazem, seguimiento por cardiología. AQ Apendicectomía. Cirugía de túnel carpiano MSD. Intervenida el 12/08/24: liberación canal estrecho lumbar. AEA: Posoperatorio con buena evolución. EA: Fiebre 38°C y dolor abdominal difuso. EF: Pálidas, febril, hidratadas. Herida eritematosa con dehiscencia a nivel medial. PP: MAV+/+. CV: FC 110 cpm. Abd: Blando, dolor a hipogastrio.

Ejemplo 3:
Paciente entre 40 y 50 años. AP: VIH hace 15 años, TARV actual Biktarvy, buena adherencia, CV 1000 copias, CD4 entre 200-250 (Dic 2024). NAC previa hace 20 años. Toxoplasmosis encefálica 2020, PCP y eritema multiforme 2023. Ex tabaquista. Sin alergias. MC: Gonalgia derecha. EA: Gonalgia 2 semanas con edema, sin fiebre. EF: Lucido, buen EG, apirético. PyM normocoloreadas. CV RR 80 cpm, PP eupneico, MAV +/+, OA inflamación y rubor local.

Ejemplo 4:
SM: Entre 30 y 40 años. AP: Niega patologías crónicas. Tabaquista. Consumo de PBC. AEA: Fractura tibia-peroné izq. 2022 con osteosíntesis. EA: Dolor en pierna izq. y fiebre 4 días. EF: Exposición de placa con secreción seropurulenta. PP: MAV ++, no estertores.
"""

base_prompt = f"""
Actuás como un **médico especialista en Infectología** que trabaja en un hospital público de Uruguay.

Tu tarea es redactar **historias clínicas sintéticas** destinadas exclusivamente a fines académicos.  
Deben **imitar fielmente la sintaxis y el estilo del registro médico rioplatense**, tal como se usa en hospitales públicos de Uruguay y Argentina, basándote estrictamente en los ejemplos provistos en el few-shot anterior.

---

### 🔒 Restricciones clínicas
1. Solo se permiten casos **de Infectología**.  
   Si la patología ingresada o seleccionada **no pertenece a la especialidad de Infectología**, devolvé el mensaje exacto:  
   > "❌ Error: el modelo está diseñado solo para historias clínicas de Infectología."
2. El caso debe ser **médicamente coherente**, sin contradicciones ni repeticiones innecesarias.  
   Ejemplo: si menciona fiebre en EA, no repetirla en antecedentes.
3. Si el caso incluye una patología o motivo incompatibles (por ejemplo, “fractura”, “hipotiroidismo”, “migraña”), devolvé el mismo mensaje de error.

---

### 🧬 Reglas de estilo rioplatense
- **Sintaxis telegráfica**, impersonal y fragmentada.  
  ❌ No usar: “El paciente refiere fiebre...”  
  ✅ Usar: “Fiebre 3 días, tos seca, sin expectoración.”  
- **Abreviaciones clínicas locales:** SF, AP, AQ, AEA, EA, EF, PyM, CV, PP, MAV, RR, Tax, VEA, SP, MMII, PBC, IRAB, HTA, DM2, Tto, TARV, etc.  
- **Estructura sugerida (no obligatoria):** SF / AP / AQ / AEA / EA / EF  
- **Cierre obligatorio en EF.**
- No incluir “Impresión”, “Plan”, “Laboratorio”, “Evolución”.
- **Nunca** agregar texto fuera de la historia clínica (sin explicaciones ni encabezados).

---

### 🧾 Uso del few-shot
Debés **replicar el estilo, ritmo y sintaxis** de los ejemplos del few-shot.  
No copiar frases literales, pero sí su **estructura, tono y cadencia**.  
Las historias deben ser **verosímiles y con densidad clínica** comparable a los ejemplos.

---

### ⚙️ Parámetros del caso

**Modo de generación:** {modo}

{f'**Edad:** {edad} años' if edad else '**Edad:** no especificada (puede definirla el modelo)'}
{f'**Sexo:** {sexo}' if sexo else '**Sexo:** no especificado (puede definirlo el modelo)'}
{f'**Patología:** {patologia}' if patologia else '**Patología:** a definir por el modelo dentro de Infectología'}
{f'**Motivo de consulta:** {motivo}' if motivo else '**Motivo de consulta:** a definir por el modelo'}

- Si alguno de los campos no fue provisto, el modelo puede definirlo libremente dentro de la práctica infectológica.
- Si el modo es “libre”, el modelo debe generar edad, sexo, patología y motivo de consulta dentro del campo de Infectología, asegurando coherencia.

---

### 🩺 Instrucción final
Generá **una única historia clínica sintética**, en formato texto plano, siguiendo todas las reglas anteriores y los ejemplos del few-shot.  
Debe sonar **como una historia clínica real** escrita por un infectólogo rioplatense, sin explicaciones ni títulos, cerrando siempre en EF.
"""

if st.button("Generar historia clínica"):
    with st.spinner("Generando historia clínica..."):
        try:
            model = genai.GenerativeModel(model_name="models/gemini-2.5-flash")
            prompt_total = f"{system_prompt}\n\n{few_shot}\n\n{base_prompt}"
            response = model.generate_content(prompt_total)
            historia = response.text.strip()
            st.text_area("Historia clínica generada:", historia, height=420)
        except Exception as e:
            st.error(f"Error al generar la historia: {e}")

