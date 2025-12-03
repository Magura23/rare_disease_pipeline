import requests
from lxml import etree as LET
import xml.etree.ElementTree as ET
from time import sleep
from typing import List, Dict
from src.rarekg.utils.config_secrets import PUBMED_API_KEY, PUBMED_EMAIL
import random
import time

def chunked(iterable, size):
    """Yield successive chunks from iterable of length `size`."""
    for i in range(0, len(iterable), size):
        yield iterable[i:i + size]


EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


def chunked(iterable, size):
    """Yield successive chunks from iterable of length `size`."""
    for i in range(0, len(iterable), size):
        yield iterable[i:i + size]


def fetch_pubmed_xml_batch(pmids: List[str]) -> str:

    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
        "rettype": "abstract",
        "email": PUBMED_EMAIL,      
        "tool": "rarekg_pipeline",
        "api key": PUBMED_API_KEY
    }

    r = requests.get(EFETCH_URL, params=params)
    r.raise_for_status()
    return r.text



def parse_titles_abstracts(xml_text: str) -> Dict[str, Dict[str, str]]:
    root = ET.fromstring(xml_text)
    results: Dict[str, Dict[str, str]] = {}

    for article in root.findall(".//PubmedArticle"):
        pmid_el = article.find(".//MedlineCitation/PMID")
        if pmid_el is None:
            continue
        pmid = pmid_el.text.strip()

        art = article.find(".//Article")
        if art is None:
            continue

        title_el = art.find("ArticleTitle")
        title = title_el.text.strip() if title_el is not None and title_el.text else ""

     
        abstract_parts = []
        abstract_el = art.find("Abstract")
        if abstract_el is not None:
            for p in abstract_el.findall("AbstractText"):
                part_text = p.text or ""
                label = p.attrib.get("Label")
                if label:
                    abstract_parts.append(f"{label}: {part_text}")
                else:
                    abstract_parts.append(part_text)
        abstract = "\n".join(part.strip() for part in abstract_parts if part.strip())

        results[pmid] = {"title": title, "abstract": abstract}

    return results


def fetch_titles_abstracts_for_pmids(
    pmids: List[str],
    batch_size: int = 200,
    sleep_time: float = 0.15,
) -> Dict[str, Dict[str, str]]:
 
    all_results: Dict[str, Dict[str, str]] = {}

    for batch in chunked(pmids, batch_size):
        xml_text = fetch_pubmed_xml_batch(batch)
        batch_results = parse_titles_abstracts(xml_text)
        all_results.update(batch_results)
        sleep(sleep_time)

    return all_results


DEFAULT_DROP_HEADS = {
    "acknowledgements","acknowledgments","funding","funding information",
    "conflict of interest","competing interests","author contributions",
    "data availability","ethics","references","bibliography",
    "supplementary","supplementary material","appendix","correspondence"
}

def get_pmc_fulltext(pmcid: str,
                     min_words: int = 5,
                     drop_heads: set[str] = None,
                     timeout: int = 90):
  
    if not pmcid:
        raise ValueError("pmcid must be a non-empty string")
    pmcid_num = pmcid.upper().removeprefix("PMC")
    pmcid_full = f"PMC{pmcid_num}"

    urls = [
        f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid_full}/fullTextXML",
        f"https://www.ncbi.nlm.nih.gov/pmc/oai/oai.cgi?verb=GetRecord&identifier=oai:pubmedcentral.nih.gov:{pmcid_num}&metadataPrefix=pmc",
    ]
    drop_heads = (drop_heads or set(DEFAULT_DROP_HEADS))
    drop_heads = {h.casefold() for h in drop_heads}

    def _norm_text(node) -> str:
        return " ".join(" ".join(node.itertext()).split())

    def _fetch_xml():
        last_err = None
        headers = {"Accept": "application/xml,text/xml,*/*"}
        for url in urls:
            try:
                r = requests.get(url, headers=headers, timeout=timeout)
                r.raise_for_status()
                content = r.content
                try:
                    root = LET.fromstring(content)
                except LET.XMLSyntaxError as e:
                    last_err = e
                    continue
                if root.xpath("//*[local-name()='article']"):
                    return root
            except Exception as e:
                last_err = e
        raise RuntimeError(f"Failed to fetch XML for {pmcid_full}: {last_err}")

    def _parse(root):
        arts = root.xpath("//*[local-name()='article']")
        if not arts:
            return "", "", []
        article = arts[0]

       
        title_nodes = article.xpath(".//*[local-name()='article-title']")
        title = _norm_text(title_nodes[0]) if title_nodes else ""

      
        abs_nodes = article.xpath(".//*[local-name()='abstract']")
        abstract = _norm_text(abs_nodes[0]) if abs_nodes else ""

        
        paras = []
        seen = set() 

        
        secs = article.xpath(".//*[local-name()='body']//*[local-name()='sec']")
        def _is_bad_head(sec) -> bool:
            head = " ".join(sec.xpath(".//*[local-name()='title']/text()")).strip().casefold()
            return any(bad in head for bad in drop_heads)

        bad_anc = "[not(ancestor::*[local-name()='table-wrap' or local-name()='fig' or local-name()='caption' or local-name()='ref-list' or local-name()='back'])]"

        for sec in secs:
            if _is_bad_head(sec):
                continue
            for p in sec.xpath(f".//*[local-name()='p']{bad_anc}"):
                t = _norm_text(p)
                if len(t.split()) >= min_words and t not in seen:
                    seen.add(t)
                    paras.append(t)

        if not paras:
            for p in article.xpath(f".//*[local-name()='body']//*[local-name()='p']{bad_anc}"):
                t = _norm_text(p)
                if len(t.split()) >= min_words and t not in seen:
                    seen.add(t)
                    paras.append(t)

        return title, abstract, paras

    root = _fetch_xml()
    return _parse(root)


ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"


def get_random_pmids(
    n: int = 16000,
    year_from: int = 2010,
    year_to: int = 2024,
    chunk_size: int = 200,
) -> List[str]:
    pmids: list[str] = []

    while len(pmids) < n:
        year = random.randint(year_from, year_to)

     
        
        base_params = {
            "db": "pubmed",
            "term": f"{year}[PDAT] and hasabstract[text]",  
            "retmode": "json",
            "retmax": 0,
            "api_key": PUBMED_API_KEY,
            "email": PUBMED_EMAIL,
        }
        r = requests.get(ESEARCH_URL, params=base_params)
        r.raise_for_status()
        try:
            count = int(r.json()["esearchresult"]["count"])
        except ValueError as e:
            print(f"JSON decode error for year {year}: {e}")
            print(f"Response text: {r.text[:200]}")
            continue
        if count == 0:
            continue
        effective_count = min(count, 9999)
 
        k = min(chunk_size, n - len(pmids), effective_count)

        max_start = max(0, effective_count - k)
        start = random.randint(0, max_start)
        params = dict(base_params)
        params["retmax"] = k
        params["retstart"] = start

        r = requests.get(ESEARCH_URL, params=params)
        r.raise_for_status()
        try:
            ids = r.json()["esearchresult"]["idlist"]
        except ValueError as e:
            print(f"JSON decode error retrieving IDs: {e}")
            print(f"Response text: {r.text[:200]}")
            continue
        pmids.extend(ids)

        time.sleep(0.35)

    pmids = list(dict.fromkeys(pmids))  
    return pmids[:n]

