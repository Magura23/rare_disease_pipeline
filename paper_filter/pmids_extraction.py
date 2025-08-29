import re
import pandas as pd


def extract_hpo_pmids():
    """
    Reads the phenotype.hpoa file and returns a list of unique PMID numbers found in the reference column.
    """
    hpoa_path = "phenotype.hpoa"
    df = pd.read_csv(hpoa_path, sep="\t", comment="#", low_memory=False)


    rare_resources = ["OMIM", "ORPHA", "ORPHANET"]

    # keep just the phenotypes that are associated with OMIM/ORPHANET -> so they are just rare diseases
    df = df[df["database_id"].str.split(":").str[0].isin(rare_resources)]  


    pmid_list = (
        df['reference']
        .str.split(';')
        .explode()
        .str.strip()
        .loc[lambda s: s.str.startswith('PMID:')]
        .str.split(':', n=1).str[1]
        .drop_duplicates()
        .tolist()
    )
    return pmid_list

# ARE THE HPO NUMBERS OR STRINGS??

def extract_pmid_from_RD():
    """
    Reads the Gene-RD-provenance file and returns a list of unique PMID
    """
    rd_path = "Gene-RD-Provenance_V2.1.txt"
    
    df = pd.read_csv(rd_path, sep="\t", dtype=str).fillna("")

    # Concatenate the two columns, drop NA and duplicates, and get unique PMIDs
    pmids = pd.concat([df["PMID Gene-disease"], df["PMID Disease"]])
    pmids = pmids[pmids != ""] 
    pmids = pmids.dropna().drop_duplicates().tolist()
    
    return pmids


def positive_examples():
    """
        Combine the PMIDs from different sources
    """
    hpo_pmids = extract_hpo_pmids()
    rd_pmids = extract_pmid_from_RD()

    combined_pmids = set(hpo_pmids).union(set(rd_pmids))
    return list(combined_pmids)
