# services/hf_client.py
from __future__ import annotations
from typing import Optional, Dict, Any, List
from transformers import pipeline

# ------- CARGA DE MODELOS PEQUEÑOS -------

def load_flan_small() -> Any:
    """
    Carga un pipeline de text2text con FLAN-T5 small (muy liviano).
    """
    return pipeline(
        task="text2text-generation",
        model="google/flan-t5-small"
    )

def load_qwen_0_5b_instruct() -> Any:
    """
    Carga un pipeline de text-generation con Qwen 0.5B Instruct (chat).
    """
    return pipeline(
        task="text-generation",
        model="Qwen/Qwen2.5-0.5B-Instruct"
    )

# ------- EJECUCIÓN DE PROMPTS -------

def run_prompt_flan(pipe: Any, prompt: str, max_new_tokens: int = 128) -> str:
    """
    Ejecuta un prompt de tipo instrucción con FLAN (text2text).
    """
    out = pipe(prompt, max_new_tokens=max_new_tokens, do_sample=False)
    return out[0]["generated_text"]

def run_prompt_qwen(pipe: Any, prompt: str, system: Optional[str] = None,
                    max_new_tokens: int = 128, temperature: float = 0.2) -> str:
    """
    Ejecuta un prompt estilo chat con Qwen Instruct (text-generation).
    Podés pasar un 'system' breve para guiar el tono.
    """
    if system:
        full = f"<<SYS>>\n{system}\n<</SYS>>\n\n{prompt}"
    else:
        full = prompt

    out = pipe(
        full,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
        pad_token_id=pipe.tokenizer.eos_token_id,
    )
    return out[0]["generated_text"].strip()

# ------- SMOKE TESTS (opcionales) -------

def smoke_test_flan() -> str:
    pipe = load_flan_small()
    prompt = "Traduce al español: 'Data privacy matters a lot in healthcare.'"
    return run_prompt_flan(pipe, prompt)

def smoke_test_qwen() -> str:
    pipe = load_qwen_0_5b_instruct()
    prompt = "Responde 'OK' y calcula 20 + 20. Da la respuesta en una sola línea."
    return run_prompt_qwen(pipe, prompt, system="Sé conciso.")


def main():
    # 1) Cargar el pipeline de FLAN-T5 small (una sola vez)
    pipe = load_flan_small()

    # 2) Prompt simple para validar que todo anda
    prompt = "Calcula 5 + 5. Responde solo con el número."

    # 3) Inferencia (determinista)
    respuesta = run_prompt_flan(
        pipe,
        prompt=prompt,
        max_new_tokens=8  # alcanza de sobra para "10"
    )

    print("Prompt:", prompt)
    print("Respuesta del modelo:", respuesta)
    
main()