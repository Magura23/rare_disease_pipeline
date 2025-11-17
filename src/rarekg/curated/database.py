import sqlite3
from typing import Literal
from rarekg.utils.string_utils import normalize_name

EntityType = Literal[
    "rare_disease",
    "gene",
    "genotype",
    "phenotype",
    "treatment",
    "drug",
]


def get_connection(path: str) -> sqlite3.Connection:
    """
    Open a SQLite connection with foreign keys enabled.
    """
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn



###########
#   Tables creation
###########
def init_schema(conn: sqlite3.Connection) -> None:
    """
    Create all tables needed for the entity database.

    Tables (exactly as requested):
      - entity(id, type)
      - rare_disease_name(normalized_name, entity_id)
      - phenotype_name(normalized_name, entity_id)
      - treatment_name(normalized_name, entity_id)
      - drug_name(normalized_name, entity_id)
      - gene_symbol(symbol, entity_id)
      - gene_name(normalized_name, entity_id)
    """

    cur = conn.cursor()


    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS entity (
            id   TEXT PRIMARY KEY,
            type TEXT NOT NULL CHECK (
                type IN (
                    'rare_disease',
                    'gene',
                    'genotype',
                    'phenotype',
                    'treatment',
                    'drug'
                )
            )
        );
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS rare_disease_name (
            normalized_name TEXT PRIMARY KEY,
            entity_id       TEXT NOT NULL,
            FOREIGN KEY (entity_id) REFERENCES entity(id) ON DELETE CASCADE
        );
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS phenotype_name (
            normalized_name TEXT PRIMARY KEY,
            entity_id       TEXT NOT NULL,
            FOREIGN KEY (entity_id) REFERENCES entity(id) ON DELETE CASCADE
        );
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS treatment_name (
            normalized_name TEXT PRIMARY KEY,
            entity_id       TEXT NOT NULL,
            FOREIGN KEY (entity_id) REFERENCES entity(id) ON DELETE CASCADE
        );
        """
    )


    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS drug_name (
            normalized_name TEXT PRIMARY KEY,
            entity_id       TEXT NOT NULL,
            FOREIGN KEY (entity_id) REFERENCES entity(id) ON DELETE CASCADE
        );
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS gene_symbol (
            symbol    TEXT PRIMARY KEY,
            entity_id TEXT NOT NULL,
            FOREIGN KEY (entity_id) REFERENCES entity(id) ON DELETE CASCADE
        );
        """
    )


    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS gene_name (
            normalized_name TEXT PRIMARY KEY,
            entity_id       TEXT NOT NULL,
            FOREIGN KEY (entity_id) REFERENCES entity(id) ON DELETE CASCADE
        );
        """
    )
    
  ######################
    #   Fast look-up
###################

    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_rare_disease_name_entity_id
        ON rare_disease_name(entity_id);
        """
    )
    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_phenotype_name_entity_id
        ON phenotype_name(entity_id);
        """
    )
    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_treatment_name_entity_id
        ON treatment_name(entity_id);
        """
    )
    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_drug_name_entity_id
        ON drug_name(entity_id);
        """
    )
    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_gene_symbol_entity_id
        ON gene_symbol(entity_id);
        """
    )
    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_gene_name_entity_id
        ON gene_name(entity_id);
        """
    )

    conn.commit()


# -------------------------
# Insertions
# -------------------------

def insert_entity(conn: sqlite3.Connection, entity_id: str, entity_type: EntityType) -> None:
    """
    Insert a row into entity(id, type).
    """
    conn.execute(
        "INSERT OR IGNORE INTO entity(id, type) VALUES (?, ?);",
        (entity_id, entity_type),
    )
    conn.commit()


def insert_rare_disease_name(conn: sqlite3.Connection, entity_id: str, raw_name: str) -> str:
    """
    Normalize a rare disease name and insert into rare_disease_name.
    Returns the normalized name used as PK.
    """
    norm = normalize_name(raw_name)
    conn.execute(
        "INSERT OR IGNORE INTO rare_disease_name(normalized_name, entity_id) VALUES (?, ?);",
        (norm, entity_id),
    )
    conn.commit()
    return norm


def insert_phenotype_name(conn: sqlite3.Connection, entity_id: str, raw_name: str) -> str:
    norm = normalize_name(raw_name)
    conn.execute(
        "INSERT OR IGNORE INTO phenotype_name(normalized_name, entity_id) VALUES (?, ?);",
        (norm, entity_id),
    )
    conn.commit()
    return norm


def insert_treatment_name(conn: sqlite3.Connection, entity_id: str, raw_name: str) -> str:
    norm = normalize_name(raw_name)
    conn.execute(
        "INSERT OR IGNORE INTO treatment_name(normalized_name, entity_id) VALUES (?, ?);",
        (norm, entity_id),
    )
    conn.commit()
    return norm


def insert_drug_name(conn: sqlite3.Connection, entity_id: str, raw_name: str) -> str:
    norm = normalize_name(raw_name)
    conn.execute(
        "INSERT OR IGNORE INTO drug_name(normalized_name, entity_id) VALUES (?, ?);",
        (norm, entity_id),
    )
    conn.commit()
    return norm


def insert_gene_symbol(conn: sqlite3.Connection, entity_id: str, symbol: str) -> None:
    """
    Insert the gene symbol exactly as it appears in the text.
    """
    conn.execute(
        "INSERT OR IGNORE INTO gene_symbol(symbol, entity_id) VALUES (?, ?);",
        (symbol, entity_id),
    )
    conn.commit()


def insert_gene_name(conn: sqlite3.Connection, entity_id: str, raw_name: str) -> str:
    """
    Normalize a gene full name and insert into gene_name.
    """
    norm = normalize_name(raw_name)
    conn.execute(
        "INSERT OR IGNORE INTO gene_name(normalized_name, entity_id) VALUES (?, ?);",
        (norm, entity_id),
    )
    conn.commit()
    return norm