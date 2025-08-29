import time, requests, xml.etree.ElementTree as ET, random
from typing import List, Dict, Optional

TOOL = "andreea-rare-disease-pipeline"
EMAIL = "magureanuandreea13@gmail.com"
BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
API_KEY = "6755b8486d7b82533be134b4aad7814e2908"   # put your NCBI API key here (str) if you have one

def _txt(elem): 
    return "".join(elem.itertext()).strip() if elem is not None else ""

def parse_pubmed_xml(xml_text: str) -> List[Dict[str,str]]:
    root = ET.fromstring(xml_text)
    out = []
    for art in root.findall(".//PubmedArticle"):
        pmid = art.findtext(".//PMID") or ""
        title = _txt(art.find(".//ArticleTitle"))
        parts = []
        for at in art.findall(".//Abstract/AbstractText"):
            lab = at.get("Label"); t = _txt(at)
            parts.append(f"{lab}: {t}" if lab else t)
            
        abstract = "\n\n".join(p for p in parts if p)
        out.append({"pmid": pmid, "title": title, "abstract": abstract})
    return out

def clean_pmids(pmids: List[str]) -> List[str]:
    return [p.strip() for p in pmids if p and p.strip().isdigit()]

def epost(pmids: List[str]) -> tuple[str, str]:
    ids = ",".join(clean_pmids(pmids))
    if not ids:
        raise ValueError("No valid numeric PMIDs.")
    
    data = {"db":"pubmed", "id": ids, "tool": TOOL, "email": EMAIL}
    
    if API_KEY: data["api_key"] = API_KEY
    
    r = requests.post(f"{BASE}/epost.fcgi", data=data, timeout=60)
    r.raise_for_status()
    
    root = ET.fromstring(r.text)
    webenv    = root.findtext("WebEnv")
    query_key = root.findtext("QueryKey")
    
    if not webenv or not query_key:
        raise RuntimeError(f"EPost returned no WebEnv/QueryKey:\n{r.text[:400]}")
    return webenv, query_key

def efetch_webenv(webenv: str, query_key: str, retmax: int = 200, pause: float = 0.34) -> List[Dict[str,str]]:
    out: List[Dict[str,str]] = []
    
    retstart = 0
    s = requests.Session()
    while True:
        params = {
            "db":"pubmed", "retmode":"xml",
            "WebEnv":webenv, "query_key":query_key,
            "retstart":retstart, "retmax":retmax,
            "tool":TOOL, "email":EMAIL
        }
        if API_KEY: params["api_key"] = API_KEY

        r = s.get(f"{BASE}/efetch.fcgi", params=params, timeout=90)
        
        
        if r.status_code == 400 and retstart > 0:
            break
        r.raise_for_status()

        batch = parse_pubmed_xml(r.text)
        if not batch:
            break
        out.extend(batch)

        # last page smaller than retmax
        if len(batch) < retmax:
            break

        
        retstart += len(batch)
        time.sleep(pause)  
    return out

def fetch_titles_abstracts(pmids: List[str], page_size: int = 400) -> List[Dict[str,str]]:
    if not pmids: return []
    
    webenv, qk = epost(pmids)
    return efetch_webenv(webenv, qk, retmax=page_size)

def _esearch_all_pmids(term: str, retmax: int = 10000, pause: float = 0.34) -> List[str]:
    
    params = {"db":"pubmed","term":term,"retmode":"xml","retmax":0, "tool":TOOL, "email":EMAIL}
    
    if API_KEY: params["api_key"] = API_KEY
    
    r = requests.get(f"{BASE}/esearch.fcgi", params=params, timeout=60)
    r.raise_for_status()
    
    root = ET.fromstring(r.text)
    
    total = int(root.findtext("Count", default="0"))
    
    pmids: List[str] = []
    
    for start in range(0, total, retmax):
        params.update({"retstart": start, "retmax": retmax})
        r = requests.get(f"{BASE}/esearch.fcgi", params=params, timeout=90)
        r.raise_for_status()
        root = ET.fromstring(r.text)
        ids = [e.text for e in root.findall("./IdList/Id") if e is not None and e.text]
        pmids.extend(ids)
        time.sleep(pause)
        
    return pmids


def build_term(date_from: str, date_to: str, extra_query: str = "") -> str:
    base = f'({date_from}:{date_to}[dp] AND hasabstract[text])'
    return base if not extra_query else f'{base} AND ({extra_query})'

def has_abstract(r: Dict[str, str]) -> bool:
        return bool(r.get("abstract", "").strip())

def sample_titles_abstracts_by_period(
    date_from: str, date_to: str, n_papers: int, seed: Optional[int] = 42, extra_query: str = "", 
    *, chunk_size: int = 200, pause: float = 0.34, retries: int  = 3
) -> List[Dict[str, str]]:
    
    term = build_term(date_from, date_to, extra_query)
    
    all_pmids = _esearch_all_pmids(term)
    
    if not all_pmids:
        return []
    
    # if fewer available papers than requested, return all
    k = min(n_papers, len(all_pmids))
    
    # sample n_papers randomly
    rnd = random.Random(seed)
    sampled_pmids = rnd.sample(all_pmids, k)
    
    # fetch titles/abstracts for the random papers
    # recs = fetch_titles_abstracts(sampled_pmids)
    recs = fetch_titles_abstracts_chunked(sampled_pmids, chunk_size=chunk_size, 
                                          pause=pause, retries=retries)
    
    
    seen = set()
    out: List[Dict[str, str]] = []
    # first, take those with abstracts
    for r in recs:
        pmid = r.get("pmid", "")
        if pmid and pmid not in seen and has_abstract(r):
            seen.add(pmid)
            out.append(r)
            if len(out) == k:
                return out

    # less than k papers with abstracts; fetch more
    if len(out) < k and len(all_pmids) > k:
        remaining = [p for p in all_pmids if p not in seen and p not in sampled_pmids]
        rnd.shuffle(remaining)

        # fetch more in chunks until we fill up or don't have any other papers
        fill_needed = k - len(out)
        for i in range(0, len(remaining), chunk_size):
            if fill_needed <= 0:
                break
            batch_ids = remaining[i : i + chunk_size]
            more = fetch_titles_abstracts_chunked(
                batch_ids, chunk_size=chunk_size, pause=pause, retries=retries
            )
            for r in more:
                pmid = r.get("pmid", "")
                if pmid and pmid not in seen and has_abstract(r):
                    seen.add(pmid)
                    out.append(r)
                    fill_needed -= 1
                    if fill_needed == 0:
                        break

    # check that there is just k records
    if len(out) > k:
        out = out[:k]
    return out


def fetch_titles_abstracts_chunked(
    pmids: list[str],
    *,
    chunk_size: int = 200,
    pause: float = 0.34,
    retries: int = 3,
) -> list[dict[str, str]]:
    
    # Clean PMIDs to pure digits & dedup while preserving order
    seen = set()
    cleaned: list[str] = []
    for p in pmids:
        p = (p or "").strip()
        if p.isdigit() and p not in seen:
            seen.add(p)
            cleaned.append(p)

    if not cleaned:
        return []

    out: list[dict[str, str]] = []
    seen_pmids: set[str] = set()
    s = requests.Session()

    for i in range(0, len(cleaned), chunk_size):
        batch = cleaned[i : i + chunk_size]
        if not batch:
            continue

        data = {
            "db": "pubmed",
            "retmode": "xml",
            "id": ",".join(batch),
            "tool": TOOL,
            "email": EMAIL,
        }
        if API_KEY:
            data["api_key"] = API_KEY

        attempt = 0
        while True:
            try:
                r = s.post(f"{BASE}/efetch.fcgi", data=data, timeout=120)
                r.raise_for_status()
                xml = r.text
                recs = parse_pubmed_xml(xml)
                
                for rec in recs:
                    pmid = rec.get("pmid", "")
                    if pmid and pmid not in seen_pmids:
                        seen_pmids.add(pmid)
                        out.append(rec)
                break 
            except requests.RequestException as e:
                attempt += 1
                if attempt > retries:
                    print(f"[efetch] failed chunk {i//chunk_size+1} after {retries} retries: {e}")
                    break
                time.sleep(min(2**attempt, 10))  

        time.sleep(pause)

    return out



