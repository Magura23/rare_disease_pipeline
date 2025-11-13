from pydantic import BaseModel, Field
from enum import Enum
from typing import List, Literal, Optional, Tuple

class EntType(str, Enum):
    disease = "disease"
    gene = "gene"
    phenotype = "phenotype"
    genotype = "genotype"   # variant
    treatment = "treatment"

class EntityMention(BaseModel):
    id: str                  # ORPHA:xxxx, HGNC:xxxx, HP:xxxx, VCV/rsID/HGVS, RxNorm/CHEBI/DrugBank
    type: EntType
    name: str                # surface form
    norm_name: Optional[str] = None  # your normalized surface (lowercase, canon rules)
    acronym: Optional[str] = None    # filled in step 2 if found
    doc_id: str

class RelationType(str, Enum):
    disease_gene = "disease_gene"
    disease_phenotype = "disease_phenotype"
    disease_genotype = "disease_genotype"
    disease_treatment = "disease_treatment"
    
    gene_phenotype = "gene_phenotype"
    gene_genotype = "gene_genotype"
    gene_treatment = "gene_treatment"
    
    treatment_genotype = "treatment_genotype"
    treatment_phenotype = "treatment_phenotype"
    
    phenotype_genotype = "phenotype_genotype"
    
    co_occurs_with = "co_occurs_with"

class Relation(BaseModel):
    type: RelationType
    head_id: str
    tail_id: str
    evidence_span: str
    evidence_offsets: Tuple[int, int]
    doc_id: str
    negated: bool = False
    hedged: bool = False
    confidence: float = 0.0
    source_model: Literal["manual","llama8b","medgemma4b","heuristic"]

class ExtractionDoc(BaseModel):
    doc_id: str
    entities: List[EntityMention]
    relations: List[Relation]
