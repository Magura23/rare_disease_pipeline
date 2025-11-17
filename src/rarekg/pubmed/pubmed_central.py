import requests
from lxml import etree as ET

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
                    root = ET.fromstring(content)
                except ET.XMLSyntaxError as e:
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
