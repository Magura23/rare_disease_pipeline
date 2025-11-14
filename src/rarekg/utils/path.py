from pathlib import Path
import os
import json


MARKERS = ("pyproject.toml", ".gitignore", "src", "notebooks")

def find_project_root(start: Path | None = None, markers=MARKERS) -> Path:
    cur = (start or Path.cwd()).resolve()
    for parent in [cur, *cur.parents]:
        if any((parent / m).exists() for m in markers):
            return parent
    # fallback: if nothing matched, use the starting dir
    return cur



def write_json(content_str: str, filename: str, path:str) -> str:
    os.makedirs(path, exist_ok=True)
    if not filename.lower().endswith(".json"):
        filename = filename + ".json"
    filepath = os.path.join(path, filename)

    try:
        parsed = json.loads(content_str)
    except Exception:
        parsed = content_str

    if isinstance(parsed, dict):
        payload = dict(parsed)
     
    else:
        payload = {"data": parsed}

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
