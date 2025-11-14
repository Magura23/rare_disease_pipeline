import os
import json
import tarfile
import zipfile
import io
import requests
from pathlib import Path
from src.rarekg.utils.path import find_project_root

# Create downloads directories in project root
PROJECT_ROOT = find_project_root(Path(__file__).parent)
DATA_DIR = PROJECT_ROOT / "data"
DOWNLOAD_DIR = DATA_DIR / "raw"

def ensure_download_directories():

    if not DATA_DIR.exists():
        DATA_DIR.mkdir(parents=False, exist_ok=True)
    if not DOWNLOAD_DIR.exists():
        DOWNLOAD_DIR.mkdir(parents=False, exist_ok=True)
   
   
   
def fetch_file(url, output_path, params=None, headers=None):
    """Download a file from a URL and save it to output_path."""
    print(f"Downloading {url} -> {output_path}")
    with requests.get(url, params=params, headers=headers, stream=True) as r:
        r.raise_for_status()
        with open(output_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    return output_path

def fetch_file_snomed(url, output_path, params=None, headers=None, timeout=300, trust_env=False):
   
    import shutil
    hdrs = {"Accept-Encoding": "identity", "User-Agent": "rarekg-downloader/1.0"}
    if headers:
        hdrs.update(headers)

    with requests.get(url, params=params, headers=hdrs, stream=True,
                      timeout=timeout, allow_redirects=True, trust_env=trust_env) as r:
        r.raise_for_status()
        r.raw.decode_content = True  # make sure gzip/deflate is handled if present
        with open(output_path, "wb") as f:
            shutil.copyfileobj(r.raw, f, length=1024 * 1024)  # 1MB blocks
    return output_path


def download_orphanet_alignment(language="en"):
    url = f"https://www.orphadata.com/data/json/{language}_product1.json.tar.gz"
    tgz_path = DOWNLOAD_DIR / f"{language}_product1.json.tar.gz"
    fetch_file(url, tgz_path)
    with tarfile.open(tgz_path, "r:gz") as tar:
        tar.extractall(path=DOWNLOAD_DIR)
    print("Extracted Orphanet alignment files.")

def download_orphanet_classifications(language="en"):
 
    page = requests.get("https://sciences.orphadata.com/classifications/")
    page.raise_for_status()
    import re
    files = re.findall(rf"{language}_product3_\d+\.xml", page.text)
    print(f"Found {len(files)} classification files.")
    for filename in files:
        url = f"https://www.orphadata.com/data/xml/{filename}"
        dest = DOWNLOAD_DIR / filename
        fetch_file(url, dest)
        
def download_orphanet_genes(language="en"):
    url = f"https://www.orphadata.com/data/xml/{language}_product6.xml"
    dest = DOWNLOAD_DIR / f"{language}_product6.xml"
    fetch_file(url, dest)


def download_orphanet_phenotypes(language="en"):
    url = f"https://www.orphadata.com/data/xml/{language}_product4.xml"
    dest = DOWNLOAD_DIR / f"{language}_product4.xml"
    fetch_file(url, dest)


def download_hpo(format="json"):

    base = "http://purl.obolibrary.org/obo/hp"
    url = f"{base}.{'json' if format=='json' else format}"
    dest = DOWNLOAD_DIR / f"hp.{format}"
    fetch_file(url, dest)


def download_hgnc_complete(format="tsv"):
    if format == "tsv":
        url = ("https://storage.googleapis.com/public-download-files/"
               "hgnc/tsv/tsv/hgnc_complete_set.txt")
        dest = DOWNLOAD_DIR / "hgnc_complete_set.txt"
        fetch_file(url, dest)
    else:
        url = ("https://storage.googleapis.com/public-download-files/"
               "hgnc/json/json/hgnc_complete_set.json")
        dest = DOWNLOAD_DIR / "hgnc_complete_set.json"
        fetch_file(url, dest)


def download_rxnorm_full(api_key):

    file_url = "https://download.nlm.nih.gov/umls/kss/rxnorm/RxNorm_full_current.zip"
    params = {"url": file_url, "apiKey": api_key}
    dest = DOWNLOAD_DIR / "RxNorm_full_current.zip"
    fetch_file("https://uts-ws.nlm.nih.gov/download", dest, params=params)

    with zipfile.ZipFile(dest, 'r') as zip_ref:
        zip_ref.extractall(DOWNLOAD_DIR / "rxnorm_full")





def download_snomed_ct_international(api_key: str):
    """
    Download the current SNOMED CT International Edition via UTS.
    Saves the raw .zip in DOWNLOAD_DIR and returns its path.
    """
    releases_url = "https://uts-ws.nlm.nih.gov/releases"
    params = {"releaseType": "snomed-ct-international-edition", "current": "true"}

    # discover latest file
    r = requests.get(releases_url, params=params, timeout=60)
    r.raise_for_status()
    items = r.json()
    if not isinstance(items, list) or not items or "downloadUrl" not in items[0]:
        raise RuntimeError("No SNOMED CT International downloadUrl from UTS Release API.")
    release_url = items[0]["downloadUrl"]  # timestamped zip on download.nlm.nih.gov

    # route through UTS Download API (enforces license)
    dest = DOWNLOAD_DIR / Path(release_url).name
    dl_api = "https://uts-ws.nlm.nih.gov/download"
    fetch_file(dl_api, dest, params={"url": release_url, "apiKey": api_key})
    with zipfile.ZipFile(dest, 'r') as zip_ref:
        zip_ref.extractall(DOWNLOAD_DIR / "snomed_ct")
        


def download_snomed_ct_us(api_key: str):
   
    releases_url = "https://uts-ws.nlm.nih.gov/releases"
    params = {"releaseType": "snomed-ct-us-edition", "current": "true"}

    # discover latest file
    r = requests.get(releases_url, params=params, timeout=60)
    r.raise_for_status()
    items = r.json()
    if not isinstance(items, list) or not items or "downloadUrl" not in items[0]:
        raise RuntimeError("No SNOMED CT US Edition downloadUrl from UTS Release API.")
    release_url = items[0]["downloadUrl"]  # timestamped zip on download.nlm.nih.gov

    # route through UTS Download API (enforces license)
    dest = DOWNLOAD_DIR / Path(release_url).name
    dl_api = "https://uts-ws.nlm.nih.gov/download"
    fetch_file(dl_api, dest, params={"url": release_url, "apiKey": api_key})

    # (optional) extract to a dedicated folder
    extract_dir = DOWNLOAD_DIR / "snomed_ct_us"
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(dest, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)


def download_umls_mrconso(api_key: str):
    """
    Download the *current* UMLS MRCONSO file via UTS and extract MRCONSO.RRF.
    Returns (zip_path, rrf_path).
    """
    base_dir =  DOWNLOAD_DIR
    base_dir.mkdir(parents=True, exist_ok=True)

    # 1) Discover the current MRCONSO file
    releases_url = "https://uts-ws.nlm.nih.gov/releases"
    params = {"releaseType": "umls-metathesaurus-mrconso-file", "current": "true"}
    r = requests.get(releases_url, params=params, timeout=60)
    r.raise_for_status()
    items = r.json()
    if not isinstance(items, list) or not items or "downloadUrl" not in items[0]:
        raise RuntimeError("No MRCONSO downloadUrl from UTS Release API.")

    source_url = items[0]["downloadUrl"]  # timestamped zip on download.nlm.nih.gov
    zip_path = base_dir / Path(source_url).name

    # 2) Download via the UTS Download API (routes license/auth)
    dl_api = "https://uts-ws.nlm.nih.gov/download"
    fetch_file(dl_api, zip_path, params={"url": source_url, "apiKey": api_key})

    # 3) Extract MRCONSO.RRF
    rrf_path = None
    with zipfile.ZipFile(zip_path, "r") as zf:
        # pick the member named exactly MRCONSO.RRF (usually in root of the zip)
        for name in zf.namelist():
            if name.endswith("MRCONSO.RRF"):
                rrf_path = base_dir / "MRCONSO.RRF"
                with zf.open(name) as src, open(rrf_path, "wb") as dst:
                    dst.write(src.read())
                break
    if rrf_path is None:
        raise RuntimeError("MRCONSO.RRF not found inside the downloaded zip.")

    return zip_path, rrf_path




def download_umls_mrsty(api_key: str, allow_full_fallback: bool = True):
    """
    Download the *current* UMLS MRSTY file via UTS and extract MRSTY.RRF.
    Tries the MRSTY-only artifact first; if missing and allow_full_fallback=True,
    falls back to the full Metathesaurus package and extracts MRSTY.RRF.
    Returns (zip_path, rrf_path).
    """
    base_dir = DOWNLOAD_DIR
    base_dir.mkdir(parents=True, exist_ok=True)

    releases_url = "https://uts-ws.nlm.nih.gov/releases"
    dl_api = "https://uts-ws.nlm.nih.gov/download"

    # 1) Try MRSTY-only artifact
    r = requests.get(releases_url, params={
        "releaseType": "umls-metathesaurus-mrsty-file",
        "current": "true"
    }, timeout=60)
    r.raise_for_status()
    items = r.json()
    if isinstance(items, list) and items and "downloadUrl" in items[0]:
        source_url = items[0]["downloadUrl"]
        zip_path = base_dir / Path(source_url).name
        fetch_file(dl_api, zip_path, params={"url": source_url, "apiKey": api_key})

        with zipfile.ZipFile(zip_path, "r") as zf:
            rrf_path = None
            for name in zf.namelist():
                if name.endswith("MRSTY.RRF"):
                    rrf_path = base_dir / "MRSTY.RRF"
                    with zf.open(name) as src, open(rrf_path, "wb") as dst:
                        dst.write(src.read())
                    break
        if rrf_path is None:
            raise RuntimeError("MRSTY.RRF not found inside the MRSTY zip.")
        return zip_path, rrf_path

    # 2) Optional fallback: Full Metathesaurus zip
    if not allow_full_fallback:
        raise RuntimeError("No MRSTY-only artifact available and fallback disabled.")

    r = requests.get(releases_url, timeout=60)
    r.raise_for_status()
    all_items = r.json()
    # pick the latest "full" metathesaurus package
    full = next(x for x in sorted(all_items, key=lambda d: d.get("releaseDate",""), reverse=True)
                if "metathesaurus" in x.get("releaseType","").lower()
                and "full" in x.get("releaseType","").lower())
    source_url = full["downloadUrl"]
    zip_path = base_dir / Path(source_url).name
    fetch_file(dl_api, zip_path, params={"url": source_url, "apiKey": api_key})

    with zipfile.ZipFile(zip_path, "r") as zf:
        rrf_path = None
        for name in zf.namelist():
            if name.endswith("MRSTY.RRF"):
                rrf_path = base_dir / "MRSTY.RRF"
                with zf.open(name) as src, open(rrf_path, "wb") as dst:
                    dst.write(src.read())
                break
    if rrf_path is None:
        raise RuntimeError("MRSTY.RRF not found inside the FULL UMLS zip.")
    return zip_path, rrf_path
