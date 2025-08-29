from __future__ import annotations

import os
import time
import pickle
from typing import Iterable, List, Tuple, Dict, Optional

import numpy as np
import requests
from PubMed import fetch_titles_abstracts_chunked

class PubMedRelevanceFilter:

    BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    API_KEY = "6755b8486d7b82533be134b4aad7814e2908"

    def __init__(
        self,
        *,
        model_path: str,
        threshold: float = 0.5,
        date_field: str = "PDAT",
        query_extra: str = "hasabstract[text]",
        esearch_page_size: int = 5000,
        efetch_chunk_size: int = 200,
        pause: Optional[float] = None,
    ) -> None:
        self.model = self._load_model(model_path)
        self.threshold = float(threshold)
        self.date_field = date_field
        self.query_extra = query_extra
        self.esearch_page_size = esearch_page_size
        self.efetch_chunk_size = efetch_chunk_size
        
        self.pause = 0.34
        
        self.tool = "andreea-rare-disease-pipeline"
        self.email = "magureanuandreea13@gmail.com"
        self.api_key = "6755b8486d7b82533be134b4aad7814e2908"

    def _load_model(self, model_path: str):
        with open(model_path, "rb") as f:
            return pickle.load(f)

    def _predict_scores(self, texts: List[str]) -> np.ndarray:
        model = self.model   # what you loaded with _load_model

   
        if hasattr(model, "model"):
            model = model.model

        if hasattr(model, "predict_proba"):
            return model.predict_proba(texts)[:, 1]
        return model.predict(texts).astype(float)


    def _esearch_page(self, term: str, retstart: int, retmax: int) -> Dict:
        params = {
            "db": "pubmed",
            "term": term,
            "retmode": "json",
            "retstart": retstart,
            "retmax": retmax,
            "tool": self.tool,
            "email": self.email,
        }
        if self.api_key:
            params["api_key"] = self.api_key
        r = requests.get(f"{self.BASE}/esearch.fcgi", params=params, timeout=60)
        r.raise_for_status()
        return r.json()

    def _iter_pmids(self, start_date: str, end_date: str) -> Iterable[List[str]]:
        sd = start_date.replace("-", "/")
        ed = end_date.replace("-", "/")
   
        date_term = f"{sd}:{ed}[{self.date_field}]"
        term = f"({date_term})"
        if self.query_extra:
            term = f"{term} AND {self.query_extra}"
            
        # Fetch the first page to determine the total count
        # If nothing found, return
        first = self._esearch_page(term, retstart=0, retmax=0)
        total = int(first["esearchresult"]["count"])
        if total == 0:
            return 
        
        seen: set[str] = set()
        retstart = 0
        while retstart < total:
            page = self._esearch_page(term, retstart=retstart, retmax=self.esearch_page_size)
            idlist: List[str] = page["esearchresult"].get("idlist", [])
            new_pmids = [p for p in idlist if p not in seen]
            if not new_pmids:
                break
            seen.update(new_pmids)
            yield new_pmids
            retstart += len(idlist)
            time.sleep(self.pause)

    def filter_by_period(
        self,
        start_date: str,
        end_date: str,
        *,
        return_all_scored: bool = False,
    ) -> List[Dict[str, float]]:
        relevant: List[Dict[str, float]] = []
        for pmid_page in self._iter_pmids(start_date, end_date):
            records = fetch_titles_abstracts_chunked(
                pmid_page,
                chunk_size=self.efetch_chunk_size,
                pause=self.pause,
            )
            texts: List[str] = []
            metas: List[str] = []
            for rec in records:
                t = (rec.get("title", "") + " " + rec.get("abstract", "")).strip()
                texts.append(t)
                metas.append(rec.get("pmid", ""))
            if not texts:
                continue
            scores = self._predict_scores(texts)
            for pmid, s in zip(metas, scores):
                score = float(s)
                if score >= self.threshold:
                    relevant.append({"pmid": pmid, "score": score})
                elif return_all_scored:
                    relevant.append({"pmid": pmid, "score": score})
            time.sleep(self.pause)
        return relevant


if __name__ == "__main__":
    model_path = "/home/guests/andreea_magureanu/projects/rare_disease_pipeline/paper_filter/data/rare_disease_classifier_1.pkl"


    filterer = PubMedRelevanceFilter(
        model_path=model_path,
        threshold=0.8,
        date_field="CRDT",  # CRDT (create date) 
        query_extra="hasabstract[text] AND english[lang]"
    )

    
    start, end = "2000/01/01", "2020/12/31"
    results = filterer.filter_by_period(start, end)

  
    print(f"Relevant papers between {start} and {end}: {len(results)}")
    for row in results[:20]:
        print(row)
    
    out_path = "/home/guests/andreea_magureanu/projects/rare_disease_pipeline/paper_filter/relevant_pmids_2000_2020.txt"
    with open(out_path, "w") as f:
        for row in results:
            f.write(f"{row['pmid']}\t{row['score']:.4f}\n")

    print(f"Saved {len(results)} relevant PMIDs to {out_path}")
