
import csv
import re
from pathlib import Path

BASE_DIR = Path(__file__).parent  # Points to the directory containing positive.py

pheno = BASE_DIR / "ids" / "phenotype.hpoa"
interact = BASE_DIR / "ids" / "pmid-Intractabl-set.txt"
orpha = BASE_DIR / "ids" / "pmid-OrphanetJR-set(1).txt"
raredis = BASE_DIR / "ids" / "pmid-RareDisjou-set.txt"
ther = BASE_DIR / "ids" / "pmid-TherAdvRar-set.txt"


def extract_pmids_from_pheno(file_path):
    pmids = set()
    with open(file_path, 'r', encoding='utf-8') as file:
        for _ in range(4):
            next(file)
            
        reader = csv.DictReader(file, delimiter='\t')
        
        if 'reference' not in reader.fieldnames:
            raise ValueError("The 'reference' column is not in the file headers.")

        for row in reader:
            ref = row['reference']
            matches = re.findall(r'PMID:(\d+)', ref)
            pmids.update(matches)

    return pmids


def read_pmids_into_set(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return {line.strip() for line in f if line.strip()}



def get_all_positive():
    ids = extract_pmids_from_pheno(pheno)
    ids |= read_pmids_into_set(interact)
    ids |= read_pmids_into_set(orpha)
    ids |= read_pmids_into_set(raredis)
    ids |= read_pmids_into_set(ther)
    return ids

