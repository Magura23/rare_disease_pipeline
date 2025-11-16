from typing import List, Dict
"""
    PROMPT 1
    
    SIMPLE version of the prompt for entity name extraction 
    NO RULES about the entity extraction, just the output format
    
"""
def get_prompt1(title, abstract):
    system = (
    "You are a precise medical information extraction agent. Your task is to identify and normalize medical entities from the provided text. "
    "Output MUST be a SINGLE JSON object with ONLY an 'entities' key, whose value is a list of objects. "
    "Each entity object must have: "
    "'name' (string, the primary/canonical name as used in the text), "
    "'type' (string, must be one of: 'rare_disease', 'phenotype', 'gene', 'genotype', 'treatment', 'drugs'), "
    "If an entity is mentioned multiple times, keep ONLY the first occurrence.  "
    "DO NOT REPEAT IDENTICAL ENTITIES"
    "Do not output any relations. Do not output any other keys or explanatory text."
    )
    user = (
        f"Title: {title}\n"
        f"Abstract: {abstract}\n\n"
        "Task: Extract entities of the specified types and return ONLY the JSON object."
    )

    return [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]

"""
    PROMPT 2
"""
def get_prompt2(title, abstract):
    system = (
    "You are a precise medical information extraction agent.\n\n"
    "Return a SINGLE JSON object with ONLY the key `entities`.\n\n"
    "Rules:\n"
    "- Use exact surface text (no paraphrase).\n"
    "- One mention per distinct surface string (deduplicate exact repeats).\n"
    "- Split coordinated phenotypes.\n"
    "- Variants: accept c./ and KEEP EACH FORM SEPARATELY even if they refer to the same variant.\n"
    "- Ignore proteins, protein complexes, databases/resources, methods, pathways.\n"
    "- If nothing is found, return {\"entities\":[]}.\n\n"
    "Schema for each entity:\n"
    "- 'name' (string, the primary/canonical name as used in the text)\n"
    "- 'type' (string, must be one of: 'rare_disease', 'phenotype', 'gene', 'genotype', 'treatment')\n\n"
    "Do not output any relations. Do not output any other keys or explanatory text."
    )

    user = (
        f"Title: {title}\n"
        f"Abstract: {abstract}\n\n"
        "Task: Extract entities of the specified types and return ONLY the JSON object."
    )


    return [
    {"role": "system", "content": system},
    {"role": "user",   "content": user},
    ]
    
"""
    PROMPT 3
"""

def get_prompt3(title, abstract):
    examples = (
    "Text: Gene and protein structures of DSPP and DMP-1 were compared.\n"
    "Output:\n"
    "{\n"
    "  \"entities\": [\n"
    "    {\"name\": \"DSPP\", \"type\": \"gene\"}\n"
    "  ]\n"
    "}\n\n"
    "Text: The patient shows pathognomonic cerebellar and brainstem malformation.\n"
    "Output:\n"
    "{\n"
    "  \"entities\": [\n"
    "    {\"name\": \"cerebellar malformation\", \"type\": \"phenotype\"},\n"
    "    {\"name\": \"brainstem malformation\", \"type\": \"phenotype\"}\n"
    "  ]\n"
    "}\n\n"
    "Text: A c.1375G>A mutation in DSPP and a p.Ala459Thr annotation were reported.\n"
    "Output:\n"
    "{\n"
    "  \"entities\": [\n"
    "    {\"name\": \"c.1375G>A\", \"type\": \"genotype\"},\n"
    "    {\"name\": \"DSPP\", \"type\": \"gene\"},\n"
    "    {\"name\": \"p.Ala459Thr\", \"type\": \"genotype\"}\n"
    "  ]\n"
    "}"
    )
    system = (
    "You are a precise medical information extraction agent.\n\n"
    "Return a SINGLE JSON object with ONLY the key `entities`.\n\n"
    "Rules:\n"
    "- Use exact surface text (no paraphrase).\n"
    "- One mention per distinct surface string (deduplicate exact repeats).\n"
    "- Split coordinated phenotypes.\n"
    "- Variants: accept c./ and KEEP EACH FORM SEPARATELY even if they refer to the same variant.\n"
    "- Ignore proteins, protein complexes, databases/resources, methods, pathways.\n"
    "- If nothing is found, return {\"entities\":[]}.\n\n"
    "Schema for each entity:\n"
    "- 'name' (string, the primary/canonical name as used in the text)\n"
    "- 'type' (string, must be one of: 'rare_disease', 'phenotype', 'gene', 'genotype', 'treatment')\n\n"
    "Do not output any relations. Do not output any other keys or explanatory text."
    )
   

    user = (
        examples + "\n\n"
        f"Title: {title}\n"
        f"Abstract: {abstract}\n\n"
        "Task: Extract entities of the specified types and return ONLY the JSON object."
    )


    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]




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


class RuleCheck(BaseModel):
    rule_id: str
    passed: bool
    note: Optional[str] = None


class EntityAudit(BaseModel):
    candidate_text: str
    decided_type: Optional[EntityType] = None
    kept: bool
    reason: Optional[str] = None
    normalization_before: Optional[str] = None
    normalization_after: Optional[str] = None
    dedupe_key: Optional[str] = None
    rule_checks: List[RuleCheck] = Field(default_factory=list)


class AuditReport(BaseModel):
    audit: List[EntityAudit] = Field(default_factory=list)



# PROD_JSON_SCHEMA: Dict[str, Any] = ExtractionResult.model_json_schema()
# AUDIT_JSON_SCHEMA: Dict[str, Any] = AuditReport.model_json_schema()

# ENTITY_JSON_SCHEMA: Dict[str, Any] = Entity.model_json_schema()
# ENTITIES_ARRAY_SCHEMA: Dict[str, Any] = {"type": "array", "items": ENTITY_JSON_SCHEMA}



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
AUDIT_SCHEMA = {
  "type": "object",
  "required": ["audit"],
  "properties": {
    "audit": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["candidate_text", "kept", "rule_checks"],
        "properties": {
          "candidate_text": {"type": "string"},
          "decided_type": {
            "type": ["string", "null"]  
          },
          "kept": {"type": "boolean"},
          "reason": {"type": "string"},
          "normalization_before": {"type": "string"},
          "normalization_after": {"type": "string"},
          "dedupe_key": {"type": "string"},
          "rule_checks": {
            "type": "array",
            "items": {
              "type": "object",
              "required": ["rule_id", "passed"],
              "properties": {
                "rule_id": {"type": "string"},
                "passed": {"type": "boolean"},
                "note": {"type": "string"},
              },
              "additionalProperties": False
            }
          }
        },
        "additionalProperties": False
      }
    }
  },
  "additionalProperties": False
}



ENTITIES_ARRAY_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "required": ["name", "type"],
        "properties": {
            "name": { "type": "string" },
            "type": {
                "type": "string",
                "enum": ["rare_disease","gene","genotype","phenotype","treatment", "drug"]
            },
        },
        "additionalProperties": False,
    },
}




SGR_ENTITY_RULES = """
1) IDENTIFY candidates by type
  - rare_disease: concrete rare disorders (EU: <1/2000) ... (etc)
  - gene: HGNC-style symbol ... 
  - genotype: HGVS only ...
  - phenotype: HPO-like; EXCLUDE inheritance/onset/modifier/frequency ...
  - treatment: therapy classes/strategies only (ERT, HSCT, gene therapy, SRT)
  - drug: concrete drug products/substances (e.g., alglucosidase alfa, miglustat)

2) NORMALIZE & DEDUPE(silent)
  - rare_disease/phenotype/treatment/drug: trim/collapse; lowercase-dedupe
  - gene: case-insensitive compare; keep ALL-CAPS if valid
  - genotype: exact string (post whitespace); exact-dedupe

3) VALIDATE(soft checks; silently discard failures)
  - gene resembles HGNC symbol or clear gene phrase
  - genotype matches HGVS shape (c./g./p. ± NM_/NC_/gene prefix)
  - phenotype is specific; not inheritance/onset/modifier/frequency
  - rare_disease not a broad class
  - treatment is therapy strategy/class
  - drug is concrete medicinal product/substance

4) EMIT
  - Production: Output ONLY the final JSON. If nothing valid, output exactly: {"entities": []}
  - Audit: If there are no candidate mentions, output exactly: {"audit": []}
"""



SYSTEM_SGR_PROD = f"""You are a precise biomedical entity tagger. Work silently with schema-guided reasoning and output ONE JSON object
    that EXACTLY matches the schema provided out-of-band (response_format/guided_json). Do NOT reveal reasoning.

    INTERNAL SGR LOOP (keep private; do not output):
    {SGR_ENTITY_RULES}
"""

SYSTEM_SGR_AUDIT = f"""You are auditing a biomedical entity extraction using schema-guided reasoning. Produce a structured AUDIT ONLY (no entities array). Use the EXACT SAME rules as the extractor.

For EACH candidate mention (kept or discarded), output:
- candidate_text
- decided_type (or null if discarded before typing)
- kept (true/false)
- reason (short)
- normalization_before
- normalization_after
- dedupe_key
- rule_checks: array of {{rule_id, passed, note}}

Rule ID namespaces:
- IDENT.[type].N  (identification checks)
- NORM.[type].N   (normalization/deduping logic)
- VALID.[type].N  (validity checks)
- EMIT.[type].N   (emission decision)

Apply these rules:
{SGR_ENTITY_RULES}
"""




# Few‑shot examples for both production and audit modes.  These examples help
# steer smaller language models toward the correct output shapes.  They are
# optional and controlled via the ``use_few_shot`` parameter.
FEW_SHOT_PROD: Sequence[Dict[str, str]] = [
    {
        "role": "user",
        "content": "CASK-related intellectual disability includes microcephaly with pontine and cerebellar hypoplasia (MICPCH). We detected NM_004006.2:c.4375C>T in CASK."
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "entities": [
                {"name": "microcephaly with pontine and cerebellar hypoplasia", "type": "rare_disease"},
                {"name": "CASK", "type": "gene"},
                {"name": "NM_004006.2:c.4375C>T", "type": "genotype"}
            ]
        })
    },
]




def build_prod_messages(text: str, use_few_shot: bool = True) -> List[Dict[str, str]]:
    msgs: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_SGR_PROD}]
    if use_few_shot:
        msgs.extend(FEW_SHOT_PROD)
    msgs.append({
        "role": "user",
        "content": f"Extract all relevant biomedical entities from the following text:\n\n{text}"
    })
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
    use_few_shot: bool = True,
    use_response_format: bool = False
) -> ExtractionResult:
    messages = build_prod_messages(text, use_few_shot=use_few_shot)
    raw = _call_with_schema(client, model, messages, JSON_SCHEM, use_response_format=use_response_format)
    try:
        return ExtractionResult.model_validate(json.loads(raw))
    except (json.JSONDecodeError, ValidationError):
        return ExtractionResult(entities=[])


