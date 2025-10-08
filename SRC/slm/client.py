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

def run_prompt_flan(
    pipe: Any,
    prompt: str,
    max_new_tokens: int = 128,
    num_beams: int = 4,
    do_sample: bool = False,
    temperature: float = 0.0,
    no_repeat_ngram_size: int = 3,
    repetition_penalty: float = 1.1,
) -> str:
    """
    Ejecuta FLAN-T5 con parámetros más robustos para evitar respuestas vacías.
    """
    out = pipe(
        prompt,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        do_sample=do_sample,
        temperature=temperature,
        no_repeat_ngram_size=no_repeat_ngram_size,
        repetition_penalty=repetition_penalty,
    )
    txt = out[0].get("generated_text", "")
    txt = "" if txt is None else str(txt)
    return txt.strip()


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

# === QWEN (Transformers, Hugging Face) ===
from typing import Any, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

def load_qwen_hf(
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    mode: str = "auto",   # "auto" | "fp16" | "bf16" | "4bit" | "cpu"
) -> Dict[str, Any]:
    """
    Carga Qwen Instruct para text-generation con su tokenizer oficial.
    - 'fp16'/'bf16' con GPU (16GB+ VRAM recomendado)
    - '4bit' usa bitsandbytes (6–8GB VRAM aprox.)
    - 'cpu' si no hay GPU (lento)
    """
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)

    if mode == "4bit":
        try:
            from transformers import BitsAndBytesConfig
            quant = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            mdl = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quant,
                device_map="auto",
                trust_remote_code=True,
            )
        except Exception as e:
            raise RuntimeError("Falta bitsandbytes o no es compatible.") from e
    elif mode in {"fp16", "bf16"}:
        dtype = torch.float16 if mode == "fp16" else torch.bfloat16
        mdl = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )
    elif mode == "cpu":
        mdl = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
    else:  # auto
        mdl = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            trust_remote_code=True,
        )

    gen = pipeline(
        "text-generation",
        model=mdl,
        tokenizer=tok,
    )
    return {"pipe": gen, "tok": tok, "name": model_name, "mode": mode}

def run_prompt_qwen_hf(
    bundle: Dict[str, Any],
    system: str,
    user: str,
    max_new_tokens: int = 256,
    temperature: float = 0.2,
    do_sample: bool = False,
) -> str:
    """
    Ejecuta Qwen Instruct usando su chat template oficial.
    Devuelve SOLO la continuación (sin eco del prompt).
    """
    tok = bundle["tok"]
    pipe = bundle["pipe"]

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})

    # Aplica el template oficial de Qwen
    prompt = tok.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    out = pipe(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.eos_token_id,
    )[0]["generated_text"]

    # Quitar el prompt si viene "eco"
    if out.startswith(prompt):
        out = out[len(prompt):]
    return out.strip()



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
    
