from typing import Optional, Tuple
import random

from services.fewshot_service import (
    seleccionar_6_casos,
    elegir_3_random,
    generar_fewshot,
)
from llm.gemini_client import generate_with_gemini


PATOLOGIAS_VALIDAS = [
    "VIH", "Tuberculosis pulmonar", "Tuberculosis ganglionar",
    "Meningitis bacteriana", "Endocarditis",
    "Infección de prótesis osteoarticular", "Celulitis",
    "Fiebre prolongada", "Neumonía adquirida en la comunidad",
    "Infección urinaria", "Candidemia",
    "Infección de sitio quirúrgico",
    "Sepsis de origen desconocido"
]


def build_system_prompt() -> str:
    return """
Sos un médico infectólogo uruguayo que trabaja en un hospital público (Maciel, Clínicas, Pasteur o INOT).
Redactás historias clínicas sintéticas indistinguibles del corpus real uruguayo (PDF + JSON).
El estilo debe ser telegráfico, fragmentado, con abreviaciones reales y jerga local auténtica. 

Regla N° 1 y FUNDAMENTAL:
La historia debe reproducir las pequeñas desprolijidades humanas presentes en los FEW-SHOTS seleccionados: variaciones reales en mayúsculas/minúsculas y nombres de secciones, orden irregular de sistemas, uso no uniforme de dos puntos y saltos de línea, puntuación inconsistente, frases entrecortadas u omitidas, repeticiones leves, conectores ausentes, cambios abruptos de ritmo y micro-omisiones típicas del corpus.

La salida debe ser únicamente la historia clínica. Sin explicaciones.

════════════════════════════════════════════
1) SECCIONES PERMITIDAS
════════════════════════════════════════════
Usar exclusivamente secciones que existen en el corpus real:
SF, SM, MC, AP, AI, AQ, AEA, EA, EF.

Prohibido crear secciones nuevas.
Prohibido escribir “Paciente F/M”.  
SF = femenina, SM = masculino.

Edad y sexo SOLO pueden aparecer si aparecen en los few-shots seleccionados.
Hábitos tóxicos solo si aparecen en los few-shots o en el input del usuario.

════════════════════════════════════════════
2) ABREVIACIONES PERMITIDAS
════════════════════════════════════════════
Usar solamente abreviaciones del corpus.

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


def build_base_prompt(
    modo: str,
    edad: Optional[int],
    sexo: Optional[str],
    patologia_final: str,
    motivo: Optional[str],
) -> str:
    return f"""
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


def decidir_patologia_final(modo: str, patologia_usuario: Optional[str]) -> str:
    if modo.startswith("Manual"):
        pat = (patologia_usuario or "").strip()
        if not pat:
            raise ValueError("En modo Manual debés ingresar una patología.")
        return pat

    return random.choice(PATOLOGIAS_VALIDAS)


def generar_historia_sintetica(
    modo: str,
    edad: Optional[int],
    sexo: Optional[str],
    patologia_usuario: Optional[str],
    motivo: Optional[str],
    casos: list,
    openrouter_key: str,
    gemini_key: str,
) -> Tuple[str, str, str]:
    """
    Devuelve:
      historia, few_shot, patologia_final
    """
    patologia_final = decidir_patologia_final(modo, patologia_usuario)

    indices6 = seleccionar_6_casos(patologia_final, casos, openrouter_key)
    indices3 = elegir_3_random(indices6)
    few_shot = generar_fewshot(indices3, patologia_final, casos)

    system_prompt = build_system_prompt()
    base_prompt = build_base_prompt(modo, edad, sexo, patologia_final, motivo)

    prompt_total = (
        f"{system_prompt}\n\n"
        "### FEW SHOT (casos seleccionados por OSS)\n"
        f"{few_shot}\n\n"
        "---- FIN FEW SHOT ----\n\n"
        f"{base_prompt}"
    )

    historia = generate_with_gemini(prompt_total, api_key=gemini_key)

    return historia, few_shot, patologia_final
