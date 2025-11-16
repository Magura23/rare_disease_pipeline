from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple
import json

from pydantic import BaseModel, Field, ValidationError
from openai import OpenAI



class EntityType(str, Enum):
    rare_disease = "rare_disease"
    gene = "gene"
    genotype = "genotype"
    phenotype = "phenotype"
    treatment = "treatment"
    drug  = "drug"
    
    def __str__(self):
        return self.value  


class Entity(BaseModel):
    name: str = Field(..., min_length=1)
    type: EntityType


class ExtractionResult(BaseModel):
   entities: List[Entity] = Field(default_factory=list)
   
   

JSON_SCHEM = {
  "type": "object",
  "required": ["entities"],
  "properties": {
    "entities": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["name","type"],
        "properties": {
          "name": { "type": "string" },
          "type": { "type": "string", "enum": ["rare_disease","gene","genotype","phenotype","treatment", "drug"] }
        },
        "additionalProperties": False
      }
    }
  },
  "additionalProperties": False
}

SGR_ALL_PROMPT = """
    You are a precise biomedical entity tagger.
    Follow the next rules silently, do NOT output any reason.

    INTERNAL SGR LOOP (keep private; do not output):
    
    1. IDENTIFY candidates by type:
        - rare_disease: named rare disorders (Orphanet-style labels; prevalence < 1/2000 or clearly used as a specific rare condition).
        - gene: HGNC-like symbols in ALL-CAPS (optionally with digits/hyphen) or explicit gene phrases (e.g. " breast cancer 1 gene").
        - genotype: HGVS-style sequence variants only (e.g. "c.123A>G", "NM_000123.3:c.123A>G").
        - phenotype: specific phenotypic abnormalities; without modifiers, inheritance, frequency.
        - treatment: therapy/management strategies or procedures (e.g. "enzyme replacement therapy", "bone marrow transplantation").
        - drug: concrete medicinal substances or products (e.g. "alglucosidase alfa", "miglustat").
                
    2. VALIDATE(soft checks; silently discard failures)
        - rare_disease: 
            If a phrase clearly functions as the disease name even if it contains phenotype words, extract it: e.g. "microcephaly with pontine and cerebellar hypoplasia"
            Do NOT extract broad categories: e.g. "lysosomal storage disease"
            
        - gene resembles HGNC symbol or clear gene phrase
        
        
        - genotype matches: valid if matches formats: c.* or NM_*:c.* 
        
        - phenotype is a phenotypic abnormality; without clinical modifiers/course, inheritence, frequency; 
            in case of shared-head coordinated phenotype mentions, split into separate phenotypes
        - treatment is therapy strategy/management
        - drug is concrete medicinal product/substance
        
        
    3. EMIT
         Output the validated entities. If nothing valid, output exactly: {"entities":[]}
"""


FEW_SHOT_ALL: Sequence[Dict[str, str]] = [
    {
        "role": "user",
        "content": "Loss-of-function (LOF) mutations in CASK cause severe developmental phenotypes, including microcephaly with pontine and cerebellar hypoplasia, X-linked intellectual disability, and autism. We detected NM_004006.2:c.4375C>T in CASK."
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "entities": [
                {"name": "microcephaly with pontine and cerebellar hypoplasia", "type": "rare_disease"},
                {"name": "CASK", "type": "gene"},
                {"name": "NM_004006.2:c.4375C>T", "type": "genotype"},
                {"name": "intellectual disability", "type": "phenotype"},
                {"name": "autism", "type": "phenotype"},
            ]
        })
    },
    {
        "role": "user",
        "content": "Alpha-mannosidosis is an autosomal recessive lysosomal storage disease. Patients often receive enzyme replacement therapy."
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "entities": [
                {"name": "Alpha-mannosidosis", "type": "rare_disease"},
                {"name": "enzyme replacement therapy", "type": "treatment"}
            ]
        })
    },
    {
        "role": "user",
        "content": "We observed severe muscular hypotonia and ptosis. Treatment was initiated with alglucosidase alfa and miglustat."
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "entities": [
                {"name": "hypotonia", "type": "phenotype"},
                {"name": "ptosis", "type": "phenotype"},
                {"name": "alglucosidase alfa", "type": "drug"},
                {"name": "miglustat", "type": "drug"}
            ]
        })
    },
    {
        "role": "user",
        "content": "Patients with Joubert syndrome with renal involvement show cerebellar and brainstem malformations."
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "entities": [
                {"name": "Joubert syndrome with renal involvement", "type": "rare_disease"},
                {"name": "cerebellar malformation", "type": "phenotype"},
                {"name": "brainstem malformation", "type": "phenotype"}
            ]
        })
    },
    {
        "role": "user",
        "content": "Here, we investigate cell-autonomous functions of CASK in cortical excitatory induced neurons (iNs) generated from CASK knockout(KO) isogenic human embryonic stem cells (hESCs) using gene expression, morphometrics, and electrophysiology."
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "entities": [
                {"name": "CASK", "type": "gene"},
            ]
        })
    },
    {
        "role": "user",
        "content": "The RNA exosome is an evolutionarily conserved, multi-subunits ribonuclease complex that is critical for processing and/or degrading a variety of RNA molecules."
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "entities": []
        })
    },
    {
        "role": "user",
        "content": "The mutation c.617G>A, p.R258S (HGVS: NC_000002.11:g.158626896 C>G) in activin A receptor type I  can present different clinical features."
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "entities": [
                {"name": "c.617G>A", "type": "genotype"},
                {"name": "activin A receptor type I", "type": "gene"},
            ]
        })
    }
    {
        "role": "user",
        "content": "Fibrodysplasia ossificans progressiva, which mostly displayed the mutation c.617G>A, p.R206H, is characterized by congenital malformation of the great toe and progressive extra-skeletal ossification of ligaments, tendons and muscles."
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "entities": [
                {"name": "fibrodysplasia ossificans progressiva", "type": "rare_disease"},
                {"name": "c.617G>A", "type": "genetype"},
                {"name": "malformation of the great toe", "type": "phenotype"},
                {"name": "extra-skeletal ossification", "type": "phenotype"},   
            ]
        })
    }
]