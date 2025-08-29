
import os
import re
import sqlite3
import pickle
import argparse
from typing import List, Dict, Iterable
from pmids_extraction import positive_examples
from PubMed import fetch_titles_abstracts, sample_titles_abstracts_by_period, fetch_titles_abstracts_chunked
from filter_classifier import RareDiseaseClassifier
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

PLOT_DIR = "plot"
DATA_DIR = "data"
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

def connect_db(path: str = "data/papers.db") -> sqlite3.Connection:
    """
    Open or create a SQLite database
    
    Return: a connection to the databse
    """
    con = sqlite3.connect(path)
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    return con

def ensure_schema(con: sqlite3.Connection) -> None:
    con.executescript("""
    CREATE TABLE IF NOT EXISTS positives (
      pmid TEXT PRIMARY KEY,
      title TEXT NOT NULL,
      abstract TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS negatives (
      pmid TEXT PRIMARY KEY,
      title TEXT NOT NULL,
      abstract TEXT NOT NULL
    );
    """)
    con.commit()

"""
    Insert new records into the given tables
"""
def upsert_records(con: sqlite3.Connection, table: str, records: List[Dict]) -> None:
    sql = f"""
        INSERT INTO {table} (pmid, title, abstract)
        VALUES (:pmid, :title, :abstract)
        ON CONFLICT(pmid) DO UPDATE SET
          title=excluded.title,
          abstract=excluded.abstract;
    """
    con.executemany(sql, records)
    con.commit()


def fetch_existing_pmids(con: sqlite3.Connection, table: str, pmids: List[str]) -> set:
    have = set()
    
    for i in range(0, len(pmids), 999):
        slice_ = pmids[i:i+999]
        placeholders = ",".join("?" for _ in slice_)
        cur = con.execute(f"SELECT pmid FROM {table} WHERE pmid IN ({placeholders})", slice_)
        have.update(p for (p,) in cur.fetchall())
    return have

def select_all_records(con: sqlite3.Connection, table: str) -> List[Dict]:
    """
    Return a list of dicts with keys: pmids, title and abstracts for the logistic 
    regression classifier
    """
    cur = con.execute(f"SELECT pmid, title, abstract FROM {table}")
    return [{"pmid": pmid, "title": title, "abstract": abstract} for (pmid, title, abstract) in cur.fetchall()]
    
def next_run_number(prefix="rare_disease_classifier", directory=DATA_DIR) -> int:
    files = [f for f in os.listdir(directory) if f.startswith(prefix) and f.endswith(".pkl")]
    numbers = []
    for f in files:
        m = re.search(r"_(\d+)\.pkl$", f)
        if m:
            numbers.append(int(m.group(1)))
    return (max(numbers) + 1) if numbers else 1
    
# def save_pickle_with_number(obj, prefix="rare_disease_classifier", directory=DATA_DIR):
#     # List existing .pkl files that match the prefix
#     files = [f for f in os.listdir(directory) if f.startswith(prefix) and f.endswith(".pkl")]
    
#     # Extract numbers from filenames
#     numbers = []
#     for f in files:
#         match = re.search(r"_(\d+)\.pkl$", f)
#         if match:
#             numbers.append(int(match.group(1)))
#     next_num = max(numbers) + 1 if numbers else 1
    
#     # Build new filename
#     filename = f"{prefix}_{next_num}.pkl"
#     path = os.path.join(directory, filename)
    
#     # Save the pickle
#     with open(path, "wb") as f:
#         pickle.dump(obj, f)
#     return path

def save_json(data: Dict, path: str) -> None:
    """Persist a Python dictionary as pretty‑printed JSON."""
    import json
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)


def save_pickle(obj, path: str) -> None:
    """Serialise a Python object to a pickle file."""
    with open(path, "wb") as fh:
        pickle.dump(obj, fh)

def main():
    
    run_id = next_run_number()
    
    #create database
    con = connect_db()
    ensure_schema(con)
    
    
    # 1. Get positive PMIDs
    pos_pmids = positive_examples()
    print(f"# positive PMIDs: {len(pos_pmids)}")

    #  Get positive records (titles/abstracts)
    existing_pos = fetch_existing_pmids(con, "positives", pos_pmids)
    missing_pos = [pmid for pmid in pos_pmids if pmid not in existing_pos]
    if missing_pos:
        print(f"Fetching {len(missing_pos)} missing positive records...")
        # Use chunked efetch for efficiency; fallback to fetch_titles_abstracts if needed
        
        new_pos_records = fetch_titles_abstracts_chunked(missing_pos, chunk_size=400)
        upsert_records(con, "positives", new_pos_records)
        print(f"Added {len(new_pos_records)} new positive records to the database.")
    else:
        print("No new positive records to fetch; database is up to date.")
        
    # Load all positive records for training the clasifier
    
    pos_records = select_all_records(con, "positives")
    print(f"Working with # positive records {len(pos_records)}")
    
    
    # 2. Get negative records (sampled from PubMed)
    # Always getting new negatives for training
    n_neg = len(pos_records)
    neg_records = sample_titles_abstracts_by_period(
            date_from="2000",
            date_to="2020",
            n_papers=n_neg,
            extra_query="",
        )
    upsert_records(con, "negatives", neg_records)
    print(f"Fetched and saved {len(neg_records)} negative records to database.")

    # 4. Prepare data for classifier
    clf = RareDiseaseClassifier()
    X_text, y, pmids = clf.prepare_data(pos_records, neg_records)

    # 5. Cross-validate and plot
    scores = clf.cross_validate(X_text, y)
    for metric in scores:
        plot_path = os.path.join(PLOT_DIR, f"cv_{metric}_{run_id}.png")
        
        plt.figure()
        plt.bar(range(1, len(scores[metric])+1), scores[metric])
        plt.xlabel("Fold")
        plt.ylabel(metric.capitalize())
        plt.title(f"Cross-validation {metric}(run {run_id})")
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved plot: {plot_path}")
        
    run_manifest = {
        "run_id": run_id,
        "n_pos": len(pos_records),
        "n_neg": len(neg_records),
        "scores": scores,
        "params": {
            "negative_sample_date_from": "2000",
            "negative_sample_date_to": "2020",
        },
    }
    save_json(run_manifest, os.path.join(DATA_DIR, f"run_{run_id}.json"))
    print(f"Saved run manifest: {os.path.join(DATA_DIR, f'run_{run_id}.json')}")
    
    # 6. Train final model and save
    clf.train(X_text, y)
    model_path = os.path.join(DATA_DIR, f"rare_disease_classifier_{run_id}.pkl")
    save_pickle(clf, model_path)
    print(f"Saved trained model to {model_path}")
    
    
if __name__ == "__main__":
   main()
