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
   
class Alias(BaseModel):

    id: int
    alias: str = Field(..., min_length=1)


class AliasResult(BaseModel):

    aliases: List[Alias] = Field(default_factory=list)

class RelationType(str, Enum):
  
    disease_gene = "disease_gene"
    disease_genotype = "disease_genotype"
    disease_phenotype = "disease_phenotype"
    disease_drug = "disease_drug"
    disease_treatment = "disease_treatment"
    
    genotype_gene = "genotype_gene"

    def __str__(self) -> str:
        return self.value
    
class Relation(BaseModel):
    subject_id: int = Field(..., ge=1)
    object_id: int = Field(..., ge=1)
    type: RelationType

class RelationResult(BaseModel):
    relations: List[Relation] = Field(default_factory=list)
    
JSON_RELATION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["relations"],
    "properties": {
        "relations": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["subject_id", "object_id", "type"],
                "properties": {
                    "subject_id": {"type": "integer"},
                    "object_id": {"type": "integer"},
                    "type": {
                        "type": "string",
                        "enum": [
                                c
                        ],
                    },
                },
                "additionalProperties": False,
            },
        }
    },
    "additionalProperties": False,
}

JSON_ALIAS_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["aliases"],
    "properties": {
        "aliases": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "alias"],
                "properties": {
                    "id": {"type": "integer"},
                    "alias": {"type": "string"},
                },
                "additionalProperties": False,
            },
        }
    },
    "additionalProperties": False,
}

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

SGR_RD_PROMPT = """
    You are a precise biomedical entity tagger.
    Follow the next rules silently, do NOT output any reason.

    INTERNAL SGR LOOP (keep private; do not output):
    
    1. IDENTIFY candidates by type:
        - rare_disease: named rare disorders (Orphanet-style labels; prevalence < 1/2000 or clearly used as a specific rare condition).
        - gene: HGNC-like symbols in ALL-CAPS (optionally with digits/hyphen) or explicit gene phrases (e.g. " breast cancer 1 gene").
        - genotype: HGVS-style sequence variants only (e.g. "c.123A>G", "NM_000123.3:c.123A>G").
        
                
    2. VALIDATE(soft checks; silently discard failures)
        - rare_disease: 
            If a phrase clearly functions as the disease name even if it contains phenotype words, extract it: e.g. "microcephaly with pontine and cerebellar hypoplasia"
            Do NOT extract broad categories: e.g. "lysosomal storage disease"
            
        - gene resembles HGNC symbol or clear gene phrase
        
        
        - genotype matches: valid if matches formats: c.* or NM_*:c.* 
        
        
    3. EMIT
         Output the validated entities. If nothing valid, output exactly: {"entities":[]}
"""
SGR_Pheno_PROMPT = """
    You are a precise biomedical entity tagger.
    Follow the next rules silently, do NOT output any reason.

    INTERNAL SGR LOOP (keep private; do not output):
    
    1. IDENTIFY candidates by type:
        - phenotype: specific phenotypic abnormalities; without modifiers, inheritance, frequency.
                
    2. VALIDATE(soft checks; silently discard failures)
        - phenotype is a phenotypic abnormality; without clinical modifiers/course, inheritence, frequency; 
            in case of shared-head coordinated phenotype mentions, split into separate phenotypes

    3. EMIT
         Output the validated entities. If nothing valid, output exactly: {"entities":[]}
"""

SGR_treatment_PROMPT = """
    You are a precise biomedical entity tagger.
    Follow the next rules silently, do NOT output any reason.

    INTERNAL SGR LOOP (keep private; do not output):
    
    1. IDENTIFY candidates by type:
        - treatment: therapy/management strategies or procedures (e.g. "enzyme replacement therapy", "bone marrow transplantation").
        - drug: concrete medicinal substances or products (e.g. "alglucosidase alfa", "miglustat").
                
    2. VALIDATE(soft checks; silently discard failures)
        - treatment is therapy strategy/management
        - drug is concrete medicinal product/substance
        
    3. EMIT
         Output the validated entities. If nothing valid, output exactly: {"entities":[]}
"""

FEW_SHOT_ALL: Sequence[Dict[str, str]] = [
    {
        "role": "user",
        "content": (
            "Loss-of-function (LOF) mutations in CASK cause severe developmental "
            "phenotypes, including microcephaly with pontine and cerebellar hypoplasia, "
            "X-linked intellectual disability, and autism. We detected NM_004006.2:c.4375C>T in CASK."
        ),
    },
    {
        "role": "assistant",
        "content": json.dumps(
            {
                "entities": [
                    {
                        "name": "microcephaly with pontine and cerebellar hypoplasia",
                        "type": "rare_disease",
                    },
                    {"name": "CASK", "type": "gene"},
                    {"name": "NM_004006.2:c.4375C>T", "type": "genotype"},
                    {"name": "intellectual disability", "type": "phenotype"},
                    {"name": "autism", "type": "phenotype"},
                ]
            }
        ),
    },
    {
        "role": "user",
        "content": (
            "Alpha-mannosidosis is an autosomal recessive lysosomal storage disease. "
            "Patients often receive enzyme replacement therapy."
        ),
    },
    {
        "role": "assistant",
        "content": json.dumps(
            {
                "entities": [
                    {"name": "Alpha-mannosidosis", "type": "rare_disease"},
                    {"name": "enzyme replacement therapy", "type": "treatment"},
                ]
            }
        ),
    },
    {
        "role": "user",
        "content": (
            "We observed severe muscular hypotonia and ptosis. Treatment was initiated "
            "with alglucosidase alfa and miglustat."
        ),
    },
    {
        "role": "assistant",
        "content": json.dumps(
            {
                "entities": [
                    {"name": "hypotonia", "type": "phenotype"},
                    {"name": "ptosis", "type": "phenotype"},
                    {"name": "alglucosidase alfa", "type": "drug"},
                    {"name": "miglustat", "type": "drug"},
                ]
            }
        ),
    },
    {
        "role": "user",
        "content": (
            "Patients with Joubert syndrome with renal involvement show cerebellar "
            "and brainstem malformations."
        ),
    },
    {
        "role": "assistant",
        "content": json.dumps(
            {
                "entities": [
                    {
                        "name": "Joubert syndrome with renal involvement",
                        "type": "rare_disease",
                    },
                    {"name": "cerebellar malformation", "type": "phenotype"},
                    {"name": "brainstem malformation", "type": "phenotype"},
                ]
            }
        ),
    },
    {
        "role": "user",
        "content": (
            "Here, we investigate cell-autonomous functions of CASK in cortical "
            "excitatory induced neurons (iNs) generated from CASK knockout(KO) "
            "isogenic human embryonic stem cells (hESCs) using gene expression, "
            "morphometrics, and electrophysiology."
        ),
    },
    {
        "role": "assistant",
        "content": json.dumps({"entities": [{"name": "CASK", "type": "gene"}]}),
    },
    {
        "role": "user",
        "content": (
            "The RNA exosome is an evolutionarily conserved, multi-subunits "
            "ribonuclease complex that is critical for processing and/or degrading "
            "a variety of RNA molecules."
        ),
    },
    {
        "role": "assistant",
        "content": json.dumps({"entities": []}),
    },
    {
        "role": "user",
        "content": (
            "The mutation c.617G>A, p.R258S (HGVS: NC_000002.11:g.158626896 C>G) "
            "in activin A receptor type I can present different clinical features."
        ),
    },
    {
        "role": "assistant",
        "content": json.dumps(
            {
                "entities": [
                    {"name": "c.617G>A", "type": "genotype"},
                    {"name": "activin A receptor type I", "type": "gene"},
                ]
            }
        ),
    },
    {
        "role": "user",
        "content": (
            "Fibrodysplasia ossificans progressiva, which mostly displayed the "
            "mutation c.617G>A, p.R206H, is characterized by congenital malformation "
            "of the great toe and progressive extra-skeletal ossification of "
            "ligaments, tendons and muscles."
        ),
    },
    {
        "role": "assistant",
        "content": json.dumps(
            {
                "entities": [
                    {
                        "name": "fibrodysplasia ossificans progressiva",
                        "type": "rare_disease",
                    },
                    {"name": "c.617G>A", "type": "genotype"},
                    {"name": "malformation of the great toe", "type": "phenotype"},
                    {"name": "extra-skeletal ossification", "type": "phenotype"},
                ]
            }
        ),
    },
    {
        "role": "user",
        "content": (
            "Cystic fibrosis is caused by biallelic variants in CFTR. "
            "The common F508del mutation leads to progressive bronchiectasis, "
            "and some patients respond to ivacaftor."
        ),
    },
    {
        "role": "assistant",
        "content": json.dumps(
            {
                "entities": [
                    {"name": "Cystic fibrosis", "type": "rare_disease"},
                    {"name": "CFTR", "type": "gene"},
                    {"name": "bronchiectasis", "type": "phenotype"},
                    {"name": "ivacaftor", "type": "drug"},
                ]
            }
        ),
    },
]

FEW_SHOT_RD: Sequence[Dict[str, str]] = [
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
            ]
        }),
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
                ]
        }),
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
        }),
    },
    {
        "role": "user",
        "content": "The RNA exosome is an evolutionarily conserved, multi-subunits ribonuclease complex that is critical for processing and/or degrading a variety of RNA molecules."
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "entities": []
        }),
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
        }),
    },
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
            ]
        }),
    },
]
FEW_SHOT_pheno: Sequence[Dict[str, str]] = [
    {
        "role": "user",
        "content": "Loss-of-function (LOF) mutations in CASK cause severe developmental phenotypes, including microcephaly with pontine and cerebellar hypoplasia, X-linked intellectual disability, and autism. We detected NM_004006.2:c.4375C>T in CASK."
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "entities": [
                {"name": "intellectual disability", "type": "phenotype"},
                {"name": "autism", "type": "phenotype"},
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
                {"name": "cerebellar malformation", "type": "phenotype"},
                {"name": "brainstem malformation", "type": "phenotype"}
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
        "content": "Fibrodysplasia ossificans progressiva, which mostly displayed the mutation c.617G>A, p.R206H, is characterized by congenital malformation of the great toe and progressive extra-skeletal ossification of ligaments, tendons and muscles."
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "entities": [
                {"name": "malformation of the great toe", "type": "phenotype"},
                {"name": "extra-skeletal ossification", "type": "phenotype"},   
            ]
        })
    }
]
FEW_SHOT_treatment: Sequence[Dict[str, str]] = [
    {
        "role": "user",
        "content": "Alpha-mannosidosis is an autosomal recessive lysosomal storage disease. Patients often receive enzyme replacement therapy."
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "entities": [
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
                {"name": "alglucosidase alfa", "type": "drug"},
                {"name": "miglustat", "type": "drug"}
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
]

SGR_ACRONYM_PROMPT = """
You are a precise biomedical acronym extractor.
You will be given a list of candidate entities identified by local numbers and a
piece of scientific text.  Your task is to find any acronyms or initialisms
defined in the text that refer to those candidate entities.  Acronyms are
short, usually uppercase letter sequences (optionally containing digits or
hyphens) that serve as abbreviations for longer terms.  Use the
following rules silently and do not output any reasoning:

1. INPUT:  The user will provide two sections:
    • A candidate list in the form of numbered lines (e.g. "[1] fibrodysplasia ossificans progressiva").
      The numbers correspond to local identifiers for each entity.
    • A TEXT section containing one or more sentences.

2. IDENTIFY ACRONYMS:
    • Look for uppercase words or capitalized mixtures of letters/digits that
      follow or precede the full name of a candidate entity, typically enclosed
      in parentheses or introduced by phrases such as "also called", "also known as",
      "aka", or "formerly".
    • The acronym must clearly refer to one of the provided candidates.  If
      multiple candidates appear in a sentence, match acronyms to the correct
      candidate based on proximity.  If unsure, skip the acronym.
    • If more than one acronym is defined for a candidate, output each one
      separately.

3. OUTPUT:
    Your output must be compatible with the provided schema.
"""
FEW_SHOT_ACRONYM: Sequence[Dict[str, str]] = [
    {
        "role": "user",
        "content": (
            "CANDIDATES:\n"
            "[1] fibrodysplasia ossificans progressiva\n"
            "[2] Alpha-mannosidosis\n\n"
            "TEXT:\n"
            "Fibrodysplasia ossificans progressiva (FOP) and alpha-mannosidosis (AM) are rare diseases."
        ),
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "aliases": [
                {"id": 1, "alias": "FOP"},
                {"id": 2, "alias": "AM"},
            ]
        }),
    },
    {
        "role": "user",
        "content": (
            "CANDIDATES:\n"
            "[1] Joubert syndrome with renal involvement\n\n"
            "TEXT:\n"
            "Patients with Joubert syndrome with renal involvement (JS-RI) have cerebellar malformations."
        ),
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "aliases": [
                {"id": 1, "alias": "JS-RI"},
            ]
        }),
    },  
]
def enumerate_entities(entities: List[Entity]) -> List[Dict[str, Any]]:
    return [
        {"id": idx + 1, "name": ent.name, "type": str(ent.type)}
        for idx, ent in enumerate(entities)
    ]

def build_acronym_messages(
    text: str,
    candidates: List[Dict[str, Any]],
    *,
    system: str,
    examples: Optional[Sequence[Dict[str, str]]] = None,
    use_few_shot: bool = True,
) -> List[Dict[str, str]]:
    lines = [f"[{c['id']}] {c['name']}" for c in candidates]
    candidate_block = "\n".join(lines)
    user_content = f"CANDIDATES:\n{candidate_block}\n\nTEXT:\n{text}"
    msgs: List[Dict[str, str]] = [{"role": "system", "content": system}]
    if use_few_shot and examples:
        msgs.extend(examples)
    msgs.append({"role": "user", "content": user_content})
    return msgs


def extract_acronyms(
    text: str,
    candidates: List[Dict[str, Any]],
    client: OpenAI,
    *,
    model: str = "medgemma",
    use_few_shot: bool = True,
    use_response_format: bool = False,
) -> AliasResult:
    messages = build_acronym_messages(
        text,
        candidates,
        system=SGR_ACRONYM_PROMPT,
        examples=FEW_SHOT_ACRONYM,
        use_few_shot=use_few_shot,
    )
    raw = _call_with_schema(
        client=client,
        model=model,
        messages=messages,
        schema=JSON_ALIAS_SCHEMA,
        use_response_format=use_response_format,
    )
    try:
        return AliasResult.model_validate(json.loads(raw))
    except (json.JSONDecodeError, ValidationError):
        return AliasResult(aliases=[])

def build_prod_messages(text: str, system: str, examples: Optional[Sequence[Dict[str, str]]] = None, use_few_shot: bool = True,) -> List[Dict[str, str]]:
    msgs: List[Dict[str, str]] = [{"role": "system", "content": system}]

    if use_few_shot and examples:
        msgs.extend(examples)

    msgs.append(
        {
            "role": "user",
            "content": f"Extract all relevant biomedical entities from the following text:\n\n{text}",
        }
    )
    return msgs

def _call_with_schema(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, str]],
    schema: Dict[str, Any],
    use_response_format: bool = True,   # set False to use guided_json
    max_tokens: int = 600
) -> str:
    if use_response_format:
        comp = client.chat.completions.create(
            model=model,
            messages=messages,
            response_format={"type": "json_schema", "json_schema": {"name": "schema", "schema": schema}},
            temperature=0.0,
            max_tokens=max_tokens,
        )
    else:
        comp = client.chat.completions.create(
            model=model,
            messages=messages,
            extra_body={"guided_json": schema},
            temperature=0.0,
            max_tokens=max_tokens,
        )
    return comp.choices[0].message.content or ""


def extract_entities(
    text: str,
    client: OpenAI,
    model: str = "medgemma",
    entity_group: str = "all",  # "all", "rare_disease", "phenotype", "treatment"
    use_few_shot: bool = True,
    use_response_format: bool = False,  # vLLM you usually want False
) -> ExtractionResult:

    if entity_group == "all":
        system = SGR_ALL_PROMPT
        examples = FEW_SHOT_ALL
    elif entity_group == "rare_disease":
        system = SGR_RD_PROMPT
        examples = FEW_SHOT_RD
    elif entity_group == "phenotype":
        system = SGR_Pheno_PROMPT
        examples = FEW_SHOT_pheno
    elif entity_group == "treatment":
        system = SGR_treatment_PROMPT
        examples = FEW_SHOT_treatment
    else:
        raise ValueError(f"Unknown entity_group: {entity_group}")

    messages = build_prod_messages(
        text=text,
        system=system,
        examples=examples,
        use_few_shot=use_few_shot,
    )

    raw = _call_with_schema(
        client,
        model=model,
        messages=messages,
        schema=JSON_SCHEM,
        use_response_format=use_response_format,
    )

    try:
        data = json.loads(raw)
        return ExtractionResult.model_validate(data)
    except (json.JSONDecodeError, ValidationError):
        return ExtractionResult(entities=[])