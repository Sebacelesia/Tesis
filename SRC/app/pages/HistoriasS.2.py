#!/usr/bin/env python
# coding: utf-8

# In[8]:


import streamlit as st
from openai import OpenAI

# Inicializar cliente OpenRouter
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="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",  # ⚠️ usa secrets.toml o variable de entorno
)

# --- Interfaz Streamlit ---
st.title("Generador de Historia Clínica Sintética")

edad = st.number_input("Edad", min_value=0, max_value=125, step=1)
sexo = st.selectbox("Sexo", ["Femenino", "Masculino", "Otro", "Prefiero no decir"])
motivo = st.text_area("Motivo de consulta", height=120, placeholder="Ej: fiebre prolongada en paciente con VIH conocido")

# --- Prompt base ---
base_prompt = f"""
Actúa como un médico especialista en Infectología que trabaja en un hospital público de Uruguay.

Tu función es redactar historias clínicas sintéticas destinadas a fines académicos. Deben imitar fielmente el estilo y la sintaxis reales del registro médico rioplatense, tal como se utiliza en hospitales públicos de Uruguay y Argentina, basándote en los ejemplos provistos.

Los casos deben ser ficticios, pero clínicamente verosímiles y coherentes con el motivo de consulta y la edad/sexo proporcionados. El texto debe sonar como una historia clínica redactada por un médico de guardia o de sala.

**REGLAS OBLIGATORIAS DE ESTILO Y SINTAXIS (Estilo Rioplatense Fragmentado):**

1.  **Solo output clínico**: La respuesta debe contener *exclusivamente* la historia clínica. **Prohibido incluir** introducciones, comentarios, etiquetas, explicaciones ni salutaciones.
2.  **Sintaxis Telegráfica**: Usar frases fragmentadas, impersonal y nominal. **Evitar oraciones completas con sujeto explícito y verbo conjugado** (p. ej., no usar "El paciente refiere...", sino "Refiere...", o mejor aún, "Dolor...").
3.  **Encadenamiento**: Unir la mayoría de las frases descriptivas con **comas o guiones**, como se observa en los ejemplos. Usar punto solo al final de una idea o sección mayor.
4.  **Motivo Integrado**: El motivo de consulta proporcionado debe **incorporarse directamente y de forma telegráfica dentro de la sección EA** (Enfermedad Actual), nunca como un encabezado separado.

**REGLAS DE ESTRUCTURA Y FORMATO:**

5.  **Estructura Base**: Seguir el orden clínico estándar: **SF / AP / AQ / AEA / EA / EF**. Omitir las secciones que sean irrelevantes para el caso generado (p. ej., si no hay cirugía, omitir AQ).
6.  **Cierre Estricto**: La historia debe **cerrar obligatoriamente en la sección EF** (Examen Físico), que debe ser detallada y reflejar el estado actual del paciente.
7.  **Prohibido**: No incluir las secciones “Impresión”, “Plan”, “Laboratorio” ni “Evolución”.
8.  **Abreviaciones Locales**: Uso constante de abreviaturas rioplatenses, tal como en los modelos: **SF, SM, AP, AQ, AEA, EA, EF, PyM, CV, PP, MAV, RR, Tax, VEA, SP, MMII, S/P, PBC, IRAB, IRBA, HTA, DM2, CBO, Tto, TTM, etc.**

**REGLAS DE CONTENIDO:**

9.  **Datos de Identificación**: Usar la `Edad` y el `Sexo` provistos en la sección `SF` o `AP` (p. ej., "SF entre {edad} y {edad+5} años"). Nunca incluir datos identificatorios reales (nombre, documento).
10. **Densidad Clínica**: El texto debe ser compacto, pero rico en detalles clínicos creíbles (p. ej., títulos de patologías con sus años, valores de CD4/CV si aplica, detalles quirúrgicos, hallazgos de examen físico).
11. **Fechas**: Usar el formato de fecha dd/mm/yy o mm/yyyy según convenga al contexto, imitando el uso de los modelos (p. ej., "diagnóstico en 2005 y 2015", "Ingreso 07/22").

**EJEMPLOS DE REFERENCIA (ESTILO CLÍNICO RIOPLATENSE):**
* *Ejemplo 1:*
SF: Entre 40 y 50 años. Procedente de Mdeo. Vive con madre e hijos. Primaria completa. -HIV diagnosticado en 2005 y 2015, no adhiere TARV. Estado inmune 04.2023: CD4 entre 130 y 180 cel/ml, CV: En torno a 300.000 -BK diseminado múltiples abandonos de ttos. Última vez que abandonó en enero de 2023. Tuberculosis miliar en 03/22 con compromiso ganglionar , biopsia Xpert y cultivo positivos para M. tuberculosis abandona tto. -Ingreso 07/22 por planteo de sífilis meningovascular, recibe tto 21 días con penicilina cristalina con VDRL al alta 1/32. -Ingreso el 01/23 a H de Clínicas por BK ganglionar + neumonitis a Sars Cov2. De dicha internación VDRL + que se trata con 3 dosis de benzetacil. Abandona tto anti BK al alta. -Meningitis a H. Influenzae 05/23 diagnosticado por filmarray , recibe Ceftriaxona 14 días. - Planteo por citología de LNH - sin enfermedad hematológica demostrada - Policonsumo, PBC, en abstinencia desde el ingreso. -No alergias a medicamentos AEA: Ingreso previo por IRB subaguda con probable participación de PCP / BK diseminado bajo tto de 2da línea por toxicidad hepática y viremia por CMV sin compromiso oftálmico. EA: Consulta por sus propios medios en puerta de emergencia 16/08/2024 por cuadro filiatorio de tos y expectoración mucopurulenta de 2 semanas de evolución, disnea de esfuerzo, no DPN, no DD, no DS, sudoración nocturna y anorexia. Del EF en emergencia: Severa desnutrición proteico-calórica. Regular estado general, lúcida, Tax 37,4ª. BF: Muguet oral. CV: RR de 90 cpm, sin soplos, sincronico. No IY ni RHY, no edemas de MMII. PA 130/80 PP: FR 18 rpm MAV abolido, estertores secos en bases pulmonares, SAT 74% VEA PyM: normocoloreada.

* *Ejemplo 2:*
Paciente, 70 a 75 años, procedente de Carmelo, vive sola, independiente para ABVD. Jubilada. AP Asma en tratamiento durante CBO. Arritmia en tratamiento con Diltiazem, seguimiento por cardiología. Patología de columna dada por canal estrecho lumbar L3-4 y L4-5 Lumbociatalgia crónica de 5 años de evolución, irradiada bilateralmente hasta los dedos de ambos pies, controlada parcialmente con Tramadol y Diclofenac. Niega otras patologías médicas, medicación habitual y alergias. AQ Apendicectomía. Cirugía de túnel carpiano MSD. Intervenida el 12/08/24: Se realizó liberación del canal estrecho lumbar mediante abordaje medial, resección del arco de L4, resección de ligamentos flavum L3-4 y L4-5, artrodesis instrumentada con colocación de tornillos y barras, control radiográfico intraoperatorio, y colocación de drenaje aspirativo. El procedimiento transcurrió sin complicaciones. AEA En el posoperatorio se otorga alta con seguimiento y controles en Carmelo. Evolucionó bien por tres semanas. En la última semana antes del ingreso, presentó episodios febriles de hasta 38°C que cedían con AINES, y dolor abdominal difuso. Sin otros síntomas o déficits motores/sensitivos. EA Comienza hace 5 días con episodios febriles, de hasta 38°C que cede con AINES, niega otra sintomatología toxiinfecciosa. Refiere dolor abdominal difuso. Niega elementos deficitarios motores/sensitivos. Niega otra sintomatología. TU-TD s/p EF PYM: Pálidas, febril , hidratadas. Curación moja serohemático, herida cicatrizal, eritematosa, dehiscencia a nivel medial PP: MAV+/+. No estertores CV: RI, taquicárdicos, FC: 110 ABD: Blando, depresible, dolor a hipogastrio PNM: Lúcida sin ni déficit motor o sensitivo.

* *Ejemplo 3:*
Paciente añosa frágil. Dependiente para actividades de la vida diaria. Vive con familiar cercano. AP: Deterioro cognitivo leve. Insuficiencia de vitamina D. AQ: Osteosíntesis de fractura subtrocantérica izquierda realizada en septiembre. Procedimiento ECM gamma largo, sin complicaciones. AEA: En el posoperatorio se otorga alta con seguimiento ambulatorio. Al control a la semana de la cirugía, se constatan elementos fluxivos leves a nivel de la herida operatoria sin clínica infecciosa. Se indica TMP-SMX por vía oral. EA: A las dos semanas de la cirugía mantiene trasladada a emergencia de INOT por aumento de signos fluxivos y dolor. Niega fiebre y elementos de sd toxiinfeccioso. Mala rehabilitación postoperatoria. Niega elementos de sd toxiinfeccioso. Refiere dolor en zona de abordaje proximal. EF en emergencia: lucidez aceptable dado deterioro cognitivo, apirética PyM: normocoloreada CV Fc 80 cpm RR PP eupneica. OA edema y eritema a nivel de abordaje, dehiscencia de herida operatoria. No se evidencia salida de pus.

* *Ejemplo 4:*
AP: SM entre 50 y 60 años Patología psiquiátrica actualmente sin tratamiento. VIH diagnosticado hace 10 años, abandono de TARV hace un año. Último control en el mes del ingreso: 30 CD4 y CV más de 5 millones de copias. Sin internaciones previas por IO. Ex PPL Ex-tabaquista. No EPOC. AI: -CEV completo. 2 dosis COVID, Vacuna de la gripe. EA: Consulta en emergencia por tos productiva de 2 meses de evolución. No cianosante, no en accesos, emetizante. Expectoración mucopurulenta de gran volumen, no episodios rojos, no drenaje postural. Concomitantemente, dolor torácico tipo puntada de lado bilateral 7/10, DE CF III. Astenia, adinamia, fatiga y sensación febril con chuchos y sudoración. Cefalea crónica agudizada desde inicio del cuadro, 8/10, holocraneana pulsátil, sin irradiaciones, que cede con AINEs. Se acompaña de vómitos de todo lo ingerido, en ocasiones con bilis, no vómitos a chorro, precedido de náuseas, sin sangrados, moco ni otros elementos anormales. Diarrea acuosa, precedido de Sd ano-rectal, un episodio de moco. No hay otros focos infecciosos. TU sin particularidades. EF Lucido. Apirético. PyM: piel fina. Desnutricion proteico calorica marcada. pliegue hiperelástico lengua seca. Normocoloreado. No lesiones en la piel. BF no muguet oral. Sin exudado, no otros elem de inmunosupresión clínica PP: MAV +/+, sibilancias y crepitantes finos difusos bilaterales. Eupneico. CV: RR 70 cpm, RGB sin soplos. SNM: GCS 15, sin desviación de rasgos. No rigidez de nuca. Sg Brusinki y Kerning negativo. Moviliza 4 miembros de forma simétrica.

* *Ejemplo 5:*
SF entre 50 y 60 años AP: CARDIOPATÍA ISQUÉMICA COLOCACIÓN DE Y BYPASS, ANTIAGREGADA HTA.. MEDICACIÓN HABITUAL AAS, ATORVA, ENALAPRIL, CARVEDILOL, ALPRAZOLAM Y FAMOTIDINA NIEGA ALERGIAS EA: DOLOR EN HD POSTINGESTA DE Colecistoquinéticos, HACE 4 DIAS QUE IRRADIA A DORSO. NAUSEAS SIN VOMITOS. CHUCHOS DE FRÍO, NO CONSTATA TAX. EF LÚCIDA SIN TAQUICARDIA NO SND PIGMENTARIO NORMocoloreada BIEN HIDRATADA Y PERFUNDIDA ABD BLANDO DEPRESIBLE DOLOR A LA PALPACIÓN DE HD MURPHY + FFLL SP

* *Ejemplo 6:*
Paciente entre 40 y 50 años AP: VIH desde hace más de 15 años años, TARV con TDF 3TC DTG, al que se rotó a Biktarvy en 2024, refiere buena adherencia, CV 1000 copias, CD4 entre 200 y 250 (Diciembre 2024). NAC que requirió ingresos CTI hace prox 20 años, toxoplasmosis encefálica en 2020, hepatotoxicidad/rash cutáneo de tto antitoxoplasma, reactivación neurotoxoplasmosis y pneumocystis, y eritema multiforme moderado en 2023. - Tabaquista intenso en abstinencia hace más de 3 meses. No BC. No EPOC. - Nega alergias medicamentosas MC: Gonalgia derecha y tumefacción EA Consulta por gonalgia derecha de 2 semanas de evolución, con edema local y sg fluxivos, sin fiebre. Sin otra sintomatología a destacar. Tránsitos sp. EF: Lucido, buen estado general. Eupneico. Apirético. Pym normocoloreadas bien hidratadas y perfundidas CV RR 80 CPM RBG SL. PP eupneico mav +/+ sin estertores OA impotencia funcional de rodilla derecha, con inflamación, edema local, rubor calor y dolor.

* *Ejemplo 7:*
SM: Entre 30 y 40 años. AP: Niega patologías crónicas. Tabaquista. Consumo diario de PBC. Niega alergias. AEA: PTM diciembre de 2022: En el que se destaca Fractura de tibia y peroné izquierdo, que requirió resolución quirúrgica con osteosíntesis con placa en enero de 20224. Cursa estadía post operatoria en H.Maciel, sin complicaciones. Buen control clínico, radiológico. En diciembre de 2023 es embestido como peatón por auto presentando Fractura expuesta multifragmentaria de tibia y peroné izquierdo, con rotura de implante. Es coordinado en centro INOT, donde se realiza nueva osteosíntesis. En sala cursa estadía postoperatoria sin complicaciones, por lo que se otorga el alta correspondiente. Sin posteriores controles. EA: Consulta en diciembre de 2024 por dolor en pierna izquierda, a nivel de osteosíntesis y fiebre 4 días de evolución. Presenta herida expuesta en pierna izquierda, se visualiza placa de osteosíntesis a dicho nivel, no supuración, rubor local. Sin posterior seguimiento. En enero 2025 consulta en emergencia INOT por exposición de material de osteosíntesis de MII. Paciente refiere rubor y supuración en dicha zona de dos semanas de evolución. EF: Buen estado general. CV: RR, 86 cpm. PP: MAV ++, no estertores. MII: Exposición de material de osteosíntesis en cara anterior de pierna izquierda. Elementos fluxivos, secreción seropurulenta.

* *Ejemplo 8:*
Valoración Inicial: 70 - 80 años DM Tipo 2 HTA Hipotiroidismo Dislipemia OBESA MC: DOLOR en articulación rodilla izquierda AEA: Postoperatorio prótesis de rodilla izquierda. EA: Consulta por hematoma de rodilla , quedando con tromboprofilaxis con enoxaparina, suspendiéndose apixaban. Relata aumento de dolor y edema de rodilla derecha, agregando impotencia funcional, sin fiebre, con supuraciòn por zona de cirugía. Niega tos y expectoración, no disnea, ni dolor abdominal, no deposiciones líquidas, no disuria.

* *Ejemplo 9:*
AP: -HAF años que compromete raquis. Paraplejia secuelar, SV a permanencia. -Tabaquista -Consumidor de cocaína, -Monorreno quirúrgico pos absceso renal. IU reiteradas. -UPP trocanteriana izquierda. 07/2024 en INOT: abordaje sobre UPP, debridamiento de tejido de granulación, osteotomía del basto femoral derecho desarticulado, control estricto de hemostasia, osteotomía del fémur proximal. EA: Consulta por síndrome suboclusivo de una semana de evolución, agregando vómitos de 3 días de evolución. Acompañado con registros febriles de hasta 41°. Refiere a su vez supuración de UPP izquierda y aumento del dolor. EF: Clínica y hemodinámicamente estable. Sin dolor Sin registros febriles, sin elementos toxiinfecciosos. TU orinas hipercoloreadas, TDB moviliza normal. Paciente se retira sonda nasogástrica.

* *Ejemplo 10:*
AP: DM2 Obeso. Niega hábitos tóxicos. Niega alergias medicamentosas. AEA: Espondilodiscitis L2-L3 por SAMS hace 4 años, Artrodesis de columna complicada con ISQ precoz a A. baumannii. Buena evolución posterior con tto ATB por 4 meses y medio. MC: Dolor lumbar. EA: Consulta por dolor y supuración en cicatriz de herida quirúrgica, se acompaña de registros febriles de 39,5°C, sin SNF ni alteraciones esfinterianas. Trasladado a centro especializado Hemodinamia estable. Asintomático en lo cardiovascular y respiratorio. Niega otros focos clínicos infecciosos. TU y TD sin alteraciones. EF: Lucido, Tax 38.2. Pym: Lengua seca. Buena perfusión periférica TR menor a 2 segundos. Herida quirúrgica con signos fluxivos, supuración por la misma. Cv: rr, 90 cpm, rbg, sin soplos. No edemas. pp: eupneico, mav +/+, sin estertores.

**Generar una historia clínica para la siguiente entrada:**

Entrada:
Edad: {edad} años
Sexo: {sexo}
Motivo de consulta: {motivo}

Salida esperada:
Una historia clínica única, con lenguaje clínico rioplatense realista, sin datos identificatorios.
"""

if st.button("Generar historia clínica"):
    with st.spinner("Generando historia clínica..."):
        completion = client.chat.completions.create(
            model="google/gemini-2.0-flash-exp:free",
            messages=[
                {"role": "system", "content": "Eres un médico infectólogo uruguayo redactando historias clínicas en sintaxis rioplatense."},
                {"role": "user", "content": base_prompt},
            ],
        )
        historia = completion.choices[0].message.content.strip()
        st.text_area("Historia clínica generada", historia, height=400)


# In[5]:





# In[ ]:




