from pathlib import Path

MARKERS = ("pyproject.toml", ".gitignore", "src", "notebooks")

def find_project_root(start: Path | None = None, markers=MARKERS) -> Path:
    cur = (start or Path.cwd()).resolve()
    for parent in [cur, *cur.parents]:
        if any((parent / m).exists() for m in markers):
            return parent
    # fallback: if nothing matched, use the starting dir
    return cur
