from pathlib import Path
import os

from dotenv import load_dotenv
from src.rarekg.utils.path import find_project_root

PROJECT_ROOT = find_project_root()

DOTENV_PATH = PROJECT_ROOT / ".env"


if DOTENV_PATH.exists():
    load_dotenv(DOTENV_PATH, override=False)
else:
    print(f"Warning: .env file not found at {DOTENV_PATH}. Make sure environment variables are set elsewhere.")
    pass


PUBMED_API_KEY = os.getenv("PUBMED_API_KEY")
PUBMED_EMAIL = os.getenv("PUBMED_EMAIL")
UMLS_KEY = os.getenv("UMLS_KEY")

if PUBMED_API_KEY is None or PUBMED_EMAIL is None or UMLS_KEY is None:
    raise RuntimeError(
        "PUBMED_API_KEY and/or PUBMED_EMAIL are not set. "
        "Put them in .env at the project root or export them as env vars."
    )
