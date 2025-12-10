import requests


def call_model_openrouter(
    prompt: str,
    api_key: str,
    model: str = "openai/gpt-oss-20b:free",
    temperature: float = 0.2,
    timeout: int = 120,
) -> str:
    url = "https://openrouter.ai/api/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
    }

    resp = requests.post(url, headers=headers, json=body, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()

    return data["choices"][0]["message"]["content"].strip()
