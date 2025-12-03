from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import json

from pydantic import BaseModel, Field, ValidationError
from openai import OpenAI


#################################################
#
# Entity extraction with all entity types
#
#################################################
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
    
    1. IDENTIFY candidates by type with its canonical name:
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
          In case of shared-head coordinated phenotype mentions, split into separate phenotypes
          
            
        - treatment is therapy strategy/management
        
        - drug is concrete medicinal product/substance
        
        
    3. EMIT
         Output the validated entities. If nothing valid, output exactly: {"entities":[]}
"""
FEW_SHOT_ALL: Sequence[Dict[str, str]] = [
    {
        "role": "user",
        "content": (
            "Affected patients typically show hearing impairment, progressive neuropsychological decline,"
            "ataxia, dysarthria, dyspraxia, coarse facial features and skeletal abnormalities such as"
            "spine and vertebral deformities, scoliosis, pes planus, hip dysplasia and knock knees (genu valgus)"
            "which often lead to arthropathy, splenomegaly, frequent infections and psychiatric symptoms. "
            "(Malm et al., 2000, 2005; Malm and Nilssen, 2008)."
        ),
    },
    {
        "role": "assistant",
        "content": json.dumps(
            {
                "entities": [
                    {"name": "hearing impairment", "type": "phenotype"},
                    {"name": "neuropsychological decline", "type": "phenotype"},
                    {"name": "ataxia", "type": "phenotype"},
                    {"name": "dysarthria", "type": "phenotype"},
                    {"name": "dyspraxia", "type": "phenotype"},
                    {"name": "coarse facial feature", "type": "phenotype"},
                    {"name": "spine deformities", "type": "phenotype"},
                    {"name": "vertebral deformities", "type": "phenotype"},
                    {"name": "scoliosis", "type": "phenotype"},
                    {"name": "pes planus", "type": "phenotype"},
                    {"name": "hip dysplasia", "type": "phenotype"},
                    {"name": " knock knees (genu valgus)", "type": "phenotype"},
                    {"name": "arthropathy", "type": "phenotype"},
                    {"name": "splenomegaly", "type": "phenotype"},
                    {"name": "frequent infections", "type": "phenotype"},
                ]
            }
            
        ),
    },
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
                    {"name": "extra-skeletal ossification of ligaments", "type": "phenotype"},
                    {"name": "extra-skeletal ossification of tendons", "type": "phenotype"},
                    {"name": "extra-skeletal ossification of muscles", "type": "phenotype"},
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

###########################################################
#
# Entity extraction just for rare diseases and causal genes
#
###########################################################

class RGEntityType(str, Enum):
    """Entity types for rare disease and gene extraction."""

    rare_disease = "rare_disease"
    gene = "gene"

    def __str__(self) -> str:
        return self.value


class RGEntity(BaseModel):
    """An entity with a name and type for the rare disease/gene task."""

    name: str = Field(..., min_length=1)
    type: RGEntityType


class RGExtractionResult(BaseModel):
    entities: List[RGEntity] = Field(default_factory=list)


JSON_SCHEMA_RARE_GENE: Dict[str, Any] = {
    "type": "object",
    "required": ["entities"],
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["name", "type"],
                "properties": {
                    "name": {"type": "string"},
                    "type": {
                        "type": "string",
                        "enum": ["rare_disease", "gene"],
                    },
                },
                "additionalProperties": False,
            },
        }
    },
    "additionalProperties": False,
}


SGR_RARE_GENE_PROMPT = """
You are a precise biomedical entity tagger.
Follow the next rules silently, do NOT output any reason.

INTERNAL SGR LOOP (keep private; do not output):

1. IDENTIFY candidates by type:
    - rare_disease: specific named rare disorders or syndromes.  Phrases that contain an acronym along
      with descriptive words (e.g. "JS with renal involvement") count as
      disease names.  Stand-alone acronyms without descriptive context must be ignored.
    - gene: HGNC-like symbols in ALL CAPS (with optional digits or
      hyphens) such as "TMEM237", or explicit gene phrases such as
      "breast cancer 1 gene", "tumor protein p53 gene" or "adenomatous
      polyposis coli gene".

2. VALIDATE (soft checks; silently discard failures):
    - For rare_disease, ensure the phrase clearly functions as a
      disease name.  Do not emit general categories like "lysosomal
      storage disease".  Ignore isolated acronyms such as "CMT1A" or
      "TSC" if no descriptive words accompany them.
    - For gene, ensure the candidate is a relevant gene symbol or
      a gene phrase that explicitly contains the word "gene".

3. EMIT:
     - If nothing valid is found, outputexactly: {"entities": []}
     
"""


FEW_SHOT_RARE_GENE: Sequence[Dict[str, str]] = [
    {
        "role": "user",
        "content": (
            "Charcot-Marie-Tooth disease type 1A (CMT1A) is caused by duplication "
            "of the PMP22 gene."
        ),
    },
    {
        "role": "assistant",
        "content": json.dumps(
            {
                "entities": [
                    {"name": "Charcot-Marie-Tooth disease type 1A", "type": "rare_disease"},
                    {"name": "PMP22", "type": "gene"},
                ]
            }
        ),
    },
    {
        "role": "user",
        "content": (
            "Mutations in the adenomatous polyposis coli gene (APC) cause familial "
            "adenomatous polyposis, while variants in the tumor protein p53 gene "
            "(TP53) are associated with Li-Fraumeni syndrome."
        ),
    },
    {
        "role": "assistant",
        "content": json.dumps(
            {
                "entities": [
                    {"name": "adenomatous polyposis coli gene", "type": "gene"},
                    {"name": "APC", "type": "gene"},
                    {"name": "familial adenomatous polyposis", "type": "rare_disease"},
                    {"name": "tumor protein p53 gene", "type": "gene"},
                    {"name": "TP53", "type": "gene"},
                    {"name": "Li-Fraumeni syndrome", "type": "rare_disease"},
                ]
            }
        ),
    },
    {
        "role": "user",
        "content": (
            "Patients with fragile X syndrome (FXS) carry expansions in the FMR1 gene, "
            "whereas those with tuberous sclerosis complex (TSC) may have mutations in the TSC1 or TSC2 genes."
        ),
    },
    {
        "role": "assistant",
        "content": json.dumps(
            {
                "entities": [
                    {"name": "fragile X syndrome", "type": "rare_disease"},
                    {"name": "FMR1", "type": "gene"},
                    {"name": "tuberous sclerosis complex", "type": "rare_disease"},
                    {"name": "TSC1", "type": "gene"},
                    {"name": "TSC2", "type": "gene"},
                ]
            }
        ),
    },
    {
        "role": "user",
        "content": (
            "This review summarises research on the DNA repair gene BRCA1 and its role "
            "in hereditary breast and ovarian cancer."
        ),
    },
    {
        "role": "assistant",
        "content": json.dumps(
            {
                "entities": [
                    {"name": "BRCA1", "type": "gene"},
                ]
            }
        ),
    },
]


##############################################
#
# Extraction of phenotypes
#
##############################################

class PhenotypeEntity(BaseModel):
    """Entity for phenotype extraction."""

    name: str = Field(..., min_length=1)
    type: str = Field("phenotype", const=True)


class PhenotypeExtractionResult(BaseModel):
    entities: List[PhenotypeEntity] = Field(default_factory=list)


JSON_SCHEMA_PHENOTYPE: Dict[str, Any] = {
    "type": "object",
    "required": ["entities"],
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["name", "type"],
                "properties": {
                    "name": {"type": "string"},
                    "type": {"type": "string", "enum": ["phenotype"]},
                },
                "additionalProperties": False,
            },
        }
    },
    "additionalProperties": False,
}
SGR_PHENOTYPE_PROMPT = """
You are a precise biomedical phenotype extractor.
Follow the next rules silently, do NOT output any reason.

INTERNAL SGR LOOP (keep private; do not output):

1. IDENTIFY phenotypes:
    – Specific phenotypic abnormalities, either as standalone terms
      (e.g. "ataxia", "hypotonia") or as compound phrases combining
      modifiers, anatomical sites and pathological issues (e.g.
      "cerebellar hypoplasia", "flexion contractures of fingers II–IV").

2. VALIDATE (soft checks; silently discard failures):
    – Exclude clinical course, inheritance, frequency and unrelated
      physiological processes.
    – When a conjunction shares a common head (e.g. "cerebellar and
      pontine hypoplasia"), split into separate phenotypes by repeating
      the head with each modifier ("cerebellar hypoplasia" and
      "pontine hypoplasia").  Similarly, separate coordinated lists
      such as "difficulty swallowing and speaking clearly" into
      "difficulty swallowing" and "difficulty speaking clearly".
    – Do not truncate long descriptive phrases; retain full
      expressions such as "malformation of the midbrain–hindbrain
      junction" or "camptodactyly of digits III".

3. EMIT: 
    If nothing valid is found, output exactly: {"entities": []}
"""


# Few‑shot examples for the phenotype task
FEW_SHOT_PHENOTYPE: Sequence[Dict[str, str]] = [
    {
        "role": "user",
        "content": (
            "The patient exhibited mild cerebellar and pontine hypoplasia, as well as "
            "difficulty swallowing and speaking clearly."
        ),
    },
    {
        "role": "assistant",
        "content": json.dumps(
            {
                "entities": [
                    {"name": "cerebellar hypoplasia", "type": "phenotype"},
                    {"name": "pontine hypoplasia", "type": "phenotype"},
                    {"name": "difficulty swallowing", "type": "phenotype"},
                    {"name": "difficulty speaking clearly", "type": "phenotype"},
                ]
            }
        ),
    },
    {
        "role": "user",
        "content": (
            "Imaging revealed malformation of the hindbrain–midbrain junction and abnormal "
            "vermian development along with flexion contractures of fingers II–IV."
        ),
    },
    {
        "role": "assistant",
        "content": json.dumps(
            {
                "entities": [
                    {"name": "malformation of the hindbrain–midbrain junction", "type": "phenotype"},
                    {"name": "abnormal vermian development", "type": "phenotype"},
                    {"name": "flexion contractures of fingers II–IV", "type": "phenotype"},
                ]
            }
        ),
    },
    {
        "role": "user",
        "content": (
            "Affected patients typically show hearing impairment, progressive neuropsychological decline,"
            "ataxia, dysarthria, dyspraxia, coarse facial features and skeletal abnormalities such as"
            "spine and vertebral deformities, scoliosis, pes planus, hip dysplasia and knock knees (genu valgus)"
            "which often lead to arthropathy, splenomegaly, frequent infections and psychiatric symptoms. "
            "(Malm et al., 2000, 2005; Malm and Nilssen, 2008)."
        ),
    },
    {
        "role": "assistant",
        "content": json.dumps(
            {
                "entities": [
                    {"name": "hearing impairment", "type": "phenotype"},
                    {"name": "neuropsychological decline", "type": "phenotype"},
                    {"name": "ataxia", "type": "phenotype"},
                    {"name": "dysarthria", "type": "phenotype"},
                    {"name": "dyspraxia", "type": "phenotype"},
                    {"name": "coarse facial feature", "type": "phenotype"},
                    {"name": "spine deformities", "type": "phenotype"},
                    {"name": "vertebral deformities", "type": "phenotype"},
                    {"name": "scoliosis", "type": "phenotype"},
                    {"name": "pes planus", "type": "phenotype"},
                    {"name": "hip dysplasia", "type": "phenotype"},
                    {"name": " knock knees (genu valgus)", "type": "phenotype"},
                    {"name": "arthropathy", "type": "phenotype"},
                    {"name": "splenomegaly", "type": "phenotype"},
                    {"name": "frequent infections", "type": "phenotype"},
                ]
            }
            
        ),
    },
    {
        "role": "user",
        "content": (
            "He presented with retinal pigmentary abnormalities, congenital muscular dystrophy "
            "and progressive spastic paraparesis."
        ),
    },
    {
        "role": "assistant",
        "content": json.dumps(
            {
                "entities": [
                    {"name": "retinal pigmentary abnormalities", "type": "phenotype"},
                    {"name": "congenital muscular dystrophy", "type": "phenotype"},
                    {"name": "spastic paraparesis", "type": "phenotype"},
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
                    {"name": "malformation of the great toe", "type": "phenotype"},
                    {"name": "extra-skeletal ossification of ligaments", "type": "phenotype"},
                    {"name": "extra-skeletal ossification of tendons", "type": "phenotype"},
                    {"name": "extra-skeletal ossification of muscles", "type": "phenotype"},
                ]
            }
        ),
    },
]




###############################################
#
#
###############################################

def build_prod_messages(text: str, system: str, examples: Optional[Sequence[Dict[str, str]]] = None) -> List[Dict[str, str]]:
    msgs: List[Dict[str, str]] = [{"role": "system", "content": system}]


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
    use_response_format: bool = False,   # set False to use guided_json
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

AnyExtractionResult = Union[ExtractionResult, RGExtractionResult, PhenotypeExtractionResult]
def extract_entities(
    text: str,
    client: OpenAI,
    model: str = "medgemma",
    extraction: str = "all",
    use_response_format: bool = False,  # vLLM you usually want False
) -> AnyExtractionResult:
    
    if extraction == "all":
        schema_json = JSON_SCHEM
        system_prompt = SGR_ALL_PROMPT
        few_shot = FEW_SHOT_ALL
        result_model = ExtractionResult
    elif extraction == "rare_gene":
        schema_json = JSON_SCHEMA_RARE_GENE
        system_prompt = SGR_RARE_GENE_PROMPT
        few_shot = FEW_SHOT_RARE_GENE
        result_model = RGExtractionResult
    elif extraction == "phenotype":
        schema_json = JSON_SCHEMA_PHENOTYPE
        system_prompt = SGR_PHENOTYPE_PROMPT
        few_shot = FEW_SHOT_PHENOTYPE
        result_model = PhenotypeExtractionResult
    else:
        raise ValueError(f"Unknown extraction type: {extraction}")
    messages = build_prod_messages(
        text=text,
        system=system_prompt,
        examples=few_shot,
    )

    raw = _call_with_schema(
        client,
        model=model,
        messages=messages,
        schema=schema_json,
        use_response_format=use_response_format,
    )

    try:
        data = json.loads(raw)
        return result_model.model_validate(data)
    except (json.JSONDecodeError, ValidationError):
        return result_model(entities=[])
