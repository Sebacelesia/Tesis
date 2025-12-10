import google.generativeai as genai


def generate_with_gemini(
    prompt: str,
    api_key: str,
    model_name: str = "models/gemini-2.5-flash",
    temperature: float = 0.5,
    top_p: float = 0.9,
    top_k: int = 40,
) -> str:
    genai.configure(api_key=api_key)

    model = genai.GenerativeModel(model_name)

    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        ),
    )

    return (response.text or "").strip()
