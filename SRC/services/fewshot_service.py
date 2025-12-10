import random
from typing import List

from llm.openrouter_client import call_model_openrouter


def seleccionar_6_casos(
    patologia: str,
    casos: list,
    openrouter_key: str,
) -> List[int]:
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

    respuesta = call_model_openrouter(prompt, api_key=openrouter_key)

    lineas = [l.strip() for l in respuesta.split("\n") if l.strip() != ""]
    ultima = lineas[-1] if lineas else ""

    if not ultima.startswith("CASOS_ELEGIDOS:"):
        import re
        nums = re.findall(r"\b([1-9]|1[0-9]|20)\b", respuesta)
        return [int(x) for x in nums[:8]]

    numeros = ultima.replace("CASOS_ELEGIDOS:", "").strip()
    indices = [int(x) for x in numeros.split(",") if x.strip().isdigit()]

    return indices


def elegir_3_random(indices_3):
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
