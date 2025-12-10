import re



MODEL_CONFIGS = {
    "Qwen 8B (Ollama)": {
        "model_name":       "qwen3:8b",
        "temperature":      0.0,
        "num_ctx":          16384,
        "num_predict":      9000,
        "top_p":            0.95,
        "top_k":            20,
        "repeat_penalty":   1.1,
        "repeat_last_n":    256,
        "presence_penalty": 0.5,
        "use_think_trunc":  True,
    },
    "Qwen 4B (Ollama)": {
        "model_name":       "qwen3:4b",
        "temperature":      0.0,
        "num_ctx":          12000,
        "num_predict":      6000,
        "top_p":            0.95,
        "top_k":            20,
        "repeat_penalty":   1.1,
        "repeat_last_n":    256,
        "presence_penalty": 0.0,
        "use_think_trunc":  True,
    },
}

OLLAMA_ENDPOINT = "http://localhost:11434"



_DEFAULT_CFG = MODEL_CONFIGS["Qwen 8B (Ollama)"]

MODEL_NAME      = _DEFAULT_CFG["model_name"]
TEMPERATURE     = _DEFAULT_CFG["temperature"]
NUM_CTX         = _DEFAULT_CFG["num_ctx"]
NUM_PREDICT     = _DEFAULT_CFG["num_predict"]

TOP_P              = _DEFAULT_CFG["top_p"]
TOP_K              = _DEFAULT_CFG["top_k"]
REPEAT_PENALTY     = _DEFAULT_CFG["repeat_penalty"]
REPEAT_LAST_N      = _DEFAULT_CFG["repeat_last_n"]
PRESENCE_PENALTY   = _DEFAULT_CFG["presence_penalty"]
USE_THINK_TRUNC    = _DEFAULT_CFG["use_think_trunc"]



USE_CHUNKING        = True
MAX_CHARS_PER_CHUNK = 15000
OVERLAP             = 0

SECTIONS_PER_BLOCK  = 1
DEBUG_PIPELINE      = True


SECTION_HEADER_REGEX = re.compile(
    r"(?m)^---\s*Secci[oó]n\s+(\d+)\s*---\s*$"
)



def set_active_model(model_label: str) -> dict:
    """
    Cambia los parámetros globales activos según MODEL_CONFIGS.
    Devuelve el cfg aplicado.
    """
    global MODEL_NAME, TEMPERATURE, NUM_CTX, NUM_PREDICT
    global TOP_P, TOP_K, REPEAT_PENALTY, REPEAT_LAST_N, PRESENCE_PENALTY, USE_THINK_TRUNC

    if model_label not in MODEL_CONFIGS:
        model_label = "Qwen 8B (Ollama)"

    cfg = MODEL_CONFIGS[model_label]

    MODEL_NAME       = cfg["model_name"]
    TEMPERATURE      = cfg["temperature"]
    NUM_CTX          = cfg["num_ctx"]
    NUM_PREDICT      = cfg["num_predict"]
    TOP_P            = cfg["top_p"]
    TOP_K            = cfg["top_k"]
    REPEAT_PENALTY   = cfg["repeat_penalty"]
    REPEAT_LAST_N    = cfg["repeat_last_n"]
    PRESENCE_PENALTY = cfg["presence_penalty"]
    USE_THINK_TRUNC  = cfg["use_think_trunc"]

    return cfg
