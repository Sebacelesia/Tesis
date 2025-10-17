import requests, json

resp = requests.post(
    "http://localhost:11434/api/generate",
    json={"model": "qwen3:8b", "prompt": "Decí 'OK' y calculá 20+22 en una sola línea."},
    stream=True,
)
text = ""
for line in resp.iter_lines():
    if not line:
        continue
    chunk = json.loads(line)
    text += chunk.get("response", "")
print(text.strip())
