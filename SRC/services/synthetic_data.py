import json
from pathlib import Path


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_casos_path() -> Path:
    root = get_project_root()
    return root / "data_partesintetica" / "casos.json"


def load_casos() -> list:
    casos_path = get_casos_path()
    with casos_path.open("r", encoding="utf-8") as f:
        return json.load(f)
