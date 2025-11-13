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
    "You are a precise medical information extraction agent. Your task is to identify and normalize medical entities from the provided text. "
    "Output MUST be a SINGLE JSON object with ONLY an 'entities' key, whose value is a list of objects. "
    "Each entity object must have: "
    "'name' (string, the primary/canonical name as used in the text), "
    "'type' (string, must be one of: 'rare_disease', 'phenotype', 'gene', 'genotype', 'treatment'), "
    "If an entity is mentioned multiple times, keep ONLY the first occurrence.  "
    "DO NOT REPEAT IDENTICAL ENTITIES"
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


"""
Utility helpers for running MedGemma-3.1-4B with schema‑guided reasoning and
producing both a list of extracted entities and an optional per‑entity audit.

This module exposes two primary functions:

1. ``extract_entities`` – given an input passage, return a list of entity
   dictionaries containing just the ``name`` and ``type`` fields.  This is
   intended for production use where only the final extraction is needed.

2. ``extract_entities_with_audit`` – behaves like ``extract_entities`` but
   additionally returns a structured audit report describing how each
   candidate mention was handled by the system.  Each audit entry includes
   details such as the candidate text, the decided type (if any), whether
   the candidate was kept or discarded, normalization keys, dedupe keys and
   the outcome of each rule check.

In both cases, the OpenAI‑compatible client for vLLM is used under the
hood.  The guided JSON schemas are derived from the ``pydantic`` models to
ensure that responses conform exactly to the expected shape.  The module
also exposes the schemas themselves via ``PROD_JSON_SCHEMA`` and
``AUDIT_JSON_SCHEMA`` for developers who wish to embed them directly in a
prompt or inspect them for debugging.

Example usage:

>>> from medgemma_audit import extract_entities, extract_entities_with_audit
>>> passage = "We report a patient with alpha‑mannosidosis. Brain MRI showed pontine hypoplasia."
>>> entities = extract_entities(passage)
>>> entities
{'entities': [{'name': 'alpha‑mannosidosis', 'type': 'rare_disease'},
              {'name': 'pontine hypoplasia', 'type': 'phenotype'}]}

>>> entities, audit = extract_entities_with_audit(passage)
>>> audit.audit[0]
EntityAudit(candidate_text='alpha‑mannosidosis', decided_type=<EntityType.rare_disease: 'rare_disease'>, kept=True, ...)

"""

from __future__ import annotations

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
    """Aggregates audit information for all candidate mentions in a passage."""

    audit: List[EntityAudit] = Field(default_factory=list)


# Generate JSON schemas from the Pydantic models.  These are exported so
# callers can embed them directly in prompts when using guided JSON to
# constrain the model's output shape.
PROD_JSON_SCHEMA: Dict[str, Any] = ExtractionResult.model_json_schema()
AUDIT_JSON_SCHEMA: Dict[str, Any] = AuditReport.model_json_schema()


def _call_guided_json(messages: List[Dict[str, str]], schema: Dict[str, Any]) -> str:
    """
    Helper to call the vLLM service with guided JSON.  This method wraps
    ``client.chat.completions.create`` and handles the specification of
    ``temperature`` and ``max_tokens``.  It returns the raw string produced
    by the model.
    """
    completion = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        extra_body={"guided_json": schema},
        temperature=0.0,
        max_tokens=1200,
    )
    return completion.choices[0].message.content or ""


# System prompts (Schema‑Guided Reasoning) for production and audit modes.
# These are essentially the same as in the original example but are
# encapsulated here as module constants so they can be reused by helper
# functions.
SYSTEM_SGR_PROD = """You are a precise biomedical entity tagger. Work silently with schema-guided reasoning and output ONE JSON object that EXACTLY matches the schema at the end. Do NOT reveal reasoning.

INTERNAL SGR LOOP (keep private; do not output):
1) IDENTIFY candidates by type
  - rare_disease: concrete rare disorders (EU: <1/2000). If a phrase is a known rare disorder even if it includes phenotype words (e.g., “microcephaly with pontine and cerebellar hypoplasia”), treat it as ONE rare_disease (do not split).
  - gene: HGNC-style symbol (ALL-CAPS, 2–10 chars, digits/hyphen allowed) or, if absent, a clearly gene-like phrase (e.g., “CASK gene”).
  - genotype: HGVS forms only: NM_*:c.*, NC_*:g.*, gene:c.*, or bare c.*.
  - phenotype: specific phenotypic abnormalities (HPO-like). EXCLUDE Mode of Inheritance, Onset/Clinical course, Modifiers, and Frequency buckets. Avoid very general terms.
    * Coordinations: split into specific items (“pontine and cerebellar hypoplasia” → “pontine hypoplasia”, “cerebellar hypoplasia”) UNLESS the full phrase is a recognized rare disease (then extract as rare_disease only).
  - treatment: therapeutic strategies/classes only (e.g., enzyme replacement therapy, hematopoietic stem-cell transplantation, gene therapy, substrate reduction therapy). Do NOT output specific drug products.

2) NORMALIZE & DEDUPE (silent)
  - rare_disease / phenotype / treatment: trim, collapse spaces; compare lowercase; emit once.
  - gene: compare case-insensitively; KEEP ALL-CAPS symbol in output if valid-looking.
  - genotype: keep exact string (post whitespace collapse); dedupe EXACT matches only.

3) VALIDATE (soft checks; silently discard failures)
  - gene: looks like HGNC symbol or clear gene phrase.
  - genotype: looks like HGVS (c./g./p. with optional NM_/NC_/gene prefix).
  - phenotype: not inheritance/onset/modifier/frequency; sufficiently specific.
  - rare_disease: concrete disorder, not a broad class.
  - treatment: strategy/class only (ERT, HSCT, SRT, gene therapy, etc.).

4) EMIT
  - Output ONLY the final JSON. If nothing valid, output: {"entities": []}
  - No comments, no extra keys.

JSON SCHEMA (final output must match exactly):
{
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
          "type": { "type": "string", "enum": ["rare_disease","gene","genotype","phenotype","treatment"] }
        },
        "additionalProperties": false
      }
    }
  },
  "additionalProperties": false
}
"""

SYSTEM_SGR_AUDIT = """You are auditing a biomedical entity extraction using schema-guided reasoning. Produce a structured AUDIT ONLY (no entities array). For EACH candidate mention (kept or discarded), output a record with:
    - candidate_text
    - decided_type (or null if discarded before typing)
    - kept (true/false)
    - reason (short)
    - normalization_before
    - normalization_after
    - dedupe_key
    - rule_checks: array of {rule_id, passed, note}

Use the following rule_id namespaces:
    - IDENT.[type].N  (identification checks)
    - NORM.[type].N   (normalization/deduping logic)
    - VALID.[type].N  (soft validation checks)
    - EMIT.[type].N   (emission constraints)

ENTITY-TYPE RULES (same as extractor):
    1) IDENT
      - rare_disease: concrete rare disorder (<1/2000 EU). If phrase is a named disorder even if it contains phenotype words, keep as ONE rare_disease.
      - gene: HGNC-style symbol (ALL-CAPS, 2–10 chars, digits/hyphen allowed) or clear gene-like phrase.
      - genotype: HGVS only: NM_*:c.*, NC_*:g.*, gene:c.*, or bare c.* / g.* / p.*.
      - phenotype: specific abnormalities (HPO-like). EXCLUDE inheritance, onset/clinical course, modifiers, frequency buckets; avoid very general categories.
        * Coordinations split into discrete abnormalities unless the full phrase is a named disorder (then that belongs under rare_disease).
      - treatment: therapy strategies/classes only (ERT, HSCT, gene therapy, SRT). No specific drug products.

    2) NORM & DEDUPE
      - rare_disease / phenotype / treatment: trim/collapse spaces → lowercase; dedupe by lowercase string.
      - gene: compare case-insensitively; keep ALL-CAPS symbol if valid.
      - genotype: keep exact post-whitespace string; dedupe exact.

    3) VALID
      - gene: looks like HGNC symbol or clear gene phrase.
      - genotype: matches HGVS shape (c./g./p. ± NM_/NC_/gene prefix).
      - phenotype: not inheritance/onset/modifier/frequency; sufficiently specific.
      - rare_disease: not just a broad class.
      - treatment: is a therapy class/strategy.

    4) EMIT
      - Candidates failing VALID are discarded. Otherwise, kept.

OUTPUT ONLY THE AUDIT JSON (no commentary).

AUDIT JSON SCHEMA:
{
  "type": "object",
  "required": ["audit"],
  "properties": {
    "audit": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["candidate_text","kept","rule_checks"],
        "properties": {
          "candidate_text": {"type":"string"},
          "decided_type": {"type":"string", "enum": ["rare_disease","gene","genotype","phenotype","treatment", null]},
          "kept": {"type":"boolean"},
          "reason": {"type":"string"},
          "normalization_before": {"type":"string"},
          "normalization_after": {"type":"string"},
          "dedupe_key": {"type":"string"},
          "rule_checks": {
            "type":"array",
            "items": {
              "type":"object",
              "required":["rule_id","passed"],
              "properties":{
                "rule_id":{"type":"string"},
                "passed":{"type":"boolean"},
                "note":{"type":"string"}
              },
              "additionalProperties": false
            }
          }
        },
        "additionalProperties": false
      }
    }
  },
  "additionalProperties": false
}
"""


def build_prod_messages(text: str, use_few_shot: bool = True) -> List[Dict[str, str]]:
    """
    Construct the message list for the production extraction call.  The
    ``SYSTEM_SGR_PROD`` prompt is prepended, optionally followed by few‑shot
    examples, and the user query with the instruction to extract entities.
    """
    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_SGR_PROD}]
    if use_few_shot:
        messages.extend(FEW_SHOT_PROD)
    messages.append({
        "role": "user",
        "content": f"Extract entities from the passage below and return exactly one JSON object:\n\n{text}"
    })
    return messages


def build_audit_messages(text: str, use_few_shot: bool = True) -> List[Dict[str, str]]:
    """
    Construct the message list for the audit call.  This prepends the
    ``SYSTEM_SGR_AUDIT`` prompt and few‑shot audit examples, then asks the
    model to produce only the audit JSON for the given passage.
    """
    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_SGR_AUDIT}]
    if use_few_shot:
        messages.extend(FEW_SHOT_AUDIT)
    messages.append({
        "role": "user",
        "content": f"Audit the entity extraction for the passage below. Output ONLY the audit JSON:\n\n{text}"
    })
    return messages


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

FEW_SHOT_AUDIT: Sequence[Dict[str, str]] = [
    {
        "role": "user",
        "content": "Alpha-mannosidosis is an autosomal recessive lysosomal storage disease. Patients often receive enzyme replacement therapy."
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "audit": [
                {
                    "candidate_text": "Alpha-mannosidosis",
                    "decided_type": "rare_disease",
                    "kept": True,
                    "reason": "accepted",
                    "normalization_before": "Alpha-mannosidosis",
                    "normalization_after": "alpha-mannosidosis",
                    "dedupe_key": "rare_disease|alpha-mannosidosis",
                    "rule_checks": [
                        {"rule_id":"IDENT.rare_disease.1","passed":True,"note":"concrete rare disorder"},
                        {"rule_id":"NORM.rare_disease.1","passed":True,"note":"lowercase + collapse spaces"},
                        {"rule_id":"VALID.rare_disease.1","passed":True,"note":"not a broad class"},
                        {"rule_id":"EMIT.rare_disease.1","passed":True,"note":"emitted"}
                    ]
                },
                {
                    "candidate_text": "enzyme replacement therapy",
                    "decided_type": "treatment",
                    "kept": True,
                    "reason": "accepted",
                    "normalization_before": "enzyme replacement therapy",
                    "normalization_after": "enzyme replacement therapy",
                    "dedupe_key": "treatment|enzyme replacement therapy",
                    "rule_checks": [
                        {"rule_id":"IDENT.treatment.1","passed":True,"note":"therapy strategy/class"},
                        {"rule_id":"NORM.treatment.1","passed":True,"note":"lowercase + collapse spaces"},
                        {"rule_id":"VALID.treatment.1","passed":True,"note":"ERT allowed"},
                        {"rule_id":"EMIT.treatment.1","passed":True,"note":"emitted"}
                    ]
                },
                {
                    "candidate_text": "autosomal recessive",
                    "decided_type": None,
                    "kept": False,
                    "reason": "discarded: inheritance bucket, not phenotype",
                    "normalization_before": "autosomal recessive",
                    "normalization_after": "autosomal recessive",
                    "dedupe_key": "skip",
                    "rule_checks": [
                        {"rule_id":"IDENT.phenotype.1","passed":False,"note":"this is inheritance, not a phenotypic abnormality"},
                        {"rule_id":"EMIT.phenotype.1","passed":False,"note":"not emitted"}
                    ]
                }
            ]
        })
    }
]


def extract_entities(text: str, use_few_shot: bool = True) -> ExtractionResult:
    """
    Extract biomedical entities from the provided text.  This function
    returns an ``ExtractionResult`` containing a list of entities with
    just ``name`` and ``type`` fields.

    :param text: The input passage from which to extract entities.
    :param use_few_shot: If ``True``, include a couple of few‑shot examples
        to guide the model toward the desired output format.  If ``False``,
        no few‑shot examples are included.
    :returns: An ``ExtractionResult`` object.  If no entities are
        extracted, the ``entities`` list will be empty.
    """
    # Construct the messages and call the model
    prod_messages = build_prod_messages(text, use_few_shot=use_few_shot)
    raw_response = _call_guided_json(prod_messages, PROD_JSON_SCHEMA)
    try:
        result = ExtractionResult.model_validate(json.loads(raw_response))
    except (json.JSONDecodeError, ValidationError):
        # If the response cannot be parsed, fall back to an empty result
        result = ExtractionResult(entities=[])
    return result


def extract_entities_with_audit(text: str, use_few_shot: bool = True) -> Tuple[ExtractionResult, AuditReport]:
    """
    Extract biomedical entities and return a detailed audit for each
    candidate mention.  The audit provides insight into why certain
    candidates were kept or discarded.  This can be useful for debugging
    extraction pipelines or for research purposes where traceability is
    required.

    :param text: The input passage from which to extract entities.
    :param use_few_shot: If ``True``, include few‑shot examples in both
        production and audit calls.  If ``False``, no few‑shot examples are
        used.
    :returns: A tuple consisting of the ``ExtractionResult`` and an
        ``AuditReport``.  If the audit cannot be parsed, its ``audit`` list
        will be empty.
    """
    # First call: production entities
    prod_messages = build_prod_messages(text, use_few_shot=use_few_shot)
    raw_prod = _call_guided_json(prod_messages, PROD_JSON_SCHEMA)
    try:
        prod = ExtractionResult.model_validate(json.loads(raw_prod))
    except (json.JSONDecodeError, ValidationError):
        prod = ExtractionResult(entities=[])

    # Second call: audit details
    audit_messages = build_audit_messages(text, use_few_shot=use_few_shot)
    raw_audit = _call_guided_json(audit_messages, AUDIT_JSON_SCHEMA)
    try:
        audit = AuditReport.model_validate(json.loads(raw_audit))
    except (json.JSONDecodeError, ValidationError):
        audit = AuditReport(audit=[])
    return prod, audit


