
import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Iterable
import os
import torch 

import networkx as nx

import pdfplumber  # type: ignore


from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline  

@dataclass
class Entity:
    name: str
    typ: str
    score: float = 1.0


@dataclass
class Relation:
    subject: str
    predicate: str
    obj: str
    score: float = 1.0


class KnowledgeGraph:
    def __init__(self) -> None:
        self.graph = nx.MultiDiGraph()

    def add_entity(self, entity: Entity) -> None:
        key = entity.name.lower().strip()
        if key in self.graph.nodes:
            self.graph.nodes[key]["score"] += 1.0
        else:
            self.graph.add_node(key, type=entity.typ, score=entity.score)

    def add_relation(self, relation: Relation) -> None:
        subj = relation.subject.lower().strip()
        obj = relation.obj.lower().strip()
        if subj not in self.graph.nodes:
            self.graph.add_node(subj, type="entity", score=1.0)
        if obj not in self.graph.nodes:
            self.graph.add_node(obj, type="entity", score=1.0)
        found = False
        for _, _, data in self.graph.out_edges(subj, data=True):
            if data.get("predicate") == relation.predicate and data.get("obj") == obj:
                data["score"] += 1.0
                found = True
                break
        if not found:
            self.graph.add_edge(
                subj,
                obj,
                predicate=relation.predicate,
                obj=obj,
                score=relation.score,
            )

    def report(self) -> None:
        print("Entities:")
        for node, data in self.graph.nodes(data=True):
            print(f"  {node} (type={data['type']}, score={data['score']})")
        print("Relations:")
        for u, v, data in self.graph.edges(data=True):
            print(
                f"  {u} --{data['predicate']}--> {v} (score={data['score']})"
            )


def parse_pdf_to_sections(pdf_path: str) -> List[str]:
    if pdfplumber is None:
        raise ImportError(
            "pdfplumber is not installed; please install it with pip install pdfplumber"
        )
    sections: List[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            # Replace multiple blank lines with a single delimiter
            cleaned = re.sub(r"\n{2,}", "\n\n", text)
            parts = [p.strip() for p in cleaned.split("\n\n") if p.strip()]
            sections.extend(parts)
    return sections


class MedGemmaExtractor:
    
    def __init__(self,
                 model_dir: str = "~/projects/rare_disease/models/medgemma-4b-it",
                 trust_remote_code: bool = False) -> None:

        path = os.path.expanduser(model_dir)
        if not os.path.isdir(path):
            raise FileNotFoundError(f"Model directory not found: {path}")

        print(f"Loading MedGemma from: {path}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            path, local_files_only=True, use_fast=True, trust_remote_code=trust_remote_code
        )

        has_cuda = torch.cuda.is_available()
        dtype = torch.float16 if has_cuda else torch.float32

        if has_cuda:
            # GPU path
            self.model = AutoModelForCausalLM.from_pretrained(
                path,
                local_files_only=True,
                device_map="auto",
                torch_dtype=dtype,
                trust_remote_code=trust_remote_code,
            )
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=256,   
            )
        else:
            # CPU path
            self.model = AutoModelForCausalLM.from_pretrained(
                path,
                local_files_only=True,
                device_map={"": "cpu"},
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                trust_remote_code=trust_remote_code,
            )
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=-1,            # CPU
                max_new_tokens=256,   
            )

        print(f"CUDA available? {has_cuda} | dtype={dtype}")

    def extract(self, text: str) -> Tuple[List[Entity], List[Relation]]:
        prompt = (
            "You are a medical information extraction agent. "
            "Given the following text, identify all diseases, causative genes, "
            "phenotypes, genotypes and treatments mentioned. "
            "Return your answer as a JSON object with two keys: 'entities' and 'relations'. "
            "The 'entities' value should be a list of objects with 'name' and 'type' keys, "
            "where type is one of 'disease', 'gene', 'phenotype', 'genotype', 'treatment'. "
            "The 'relations' value should be a list of objects with 'subject', 'predicate' and 'object' keys, "
            "describing relations between the entities (e.g. 'causes', 'associated_with', etc.). "
            "If no relations are present, return an empty list for 'relations'. "
            "Provide only the JSON object as output without explanation.\n\n"
            f"Text: {text}\n"
        )
        
        outputs = self.generator(prompt)
        raw_output = outputs[0]["generated_text"]
        start_idx = raw_output.find("{")
        end_idx = raw_output.rfind("}")
        entities: List[Entity] = []
        relations: List[Relation] = []
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            json_str = raw_output[start_idx : end_idx + 1]
            try:
                data = json.loads(json_str)
                for item in data.get("entities", []):
                    name = str(item.get("name", "")).strip().lower()
                    typ = str(item.get("type", "entity")).strip().lower()
                    entities.append(Entity(name=name, typ=typ))
                for rel in data.get("relations", []):
                    subj = str(rel.get("subject", "")).strip().lower()
                    pred = str(rel.get("predicate", "associated_with")).strip().lower()
                    obj = str(rel.get("object", "")).strip().lower()
                    relations.append(
                        Relation(subject=subj, predicate=pred, obj=obj)
                    )
            except json.JSONDecodeError:
                pass
        return entities, relations


def build_knowledge_graph_from_pdf(
    pdf_path: str,
    model_dir: str = "~/projects/rare_disease/models/medgemma-4b-it",
    trust_remote_code: bool = False,
) -> KnowledgeGraph:
    sections = parse_pdf_to_sections(pdf_path)
    kg = KnowledgeGraph()

    extractor = MedGemmaExtractor(model_dir=model_dir, trust_remote_code=trust_remote_code)

    for idx, section in enumerate(sections, 1):
        if not section.strip():
            continue
        print(f"Processing section {idx}/{len(sections)}...")
        # (optional) truncate very long chunks so CPU doesn’t crawl:
        text = section[:2500]
        try:
            entities, relations = extractor.extract(text)
        except Exception as e:
            print(f"Section {idx} failed: {e}")
            continue

        for ent in entities:
            kg.add_entity(ent)
        for rel in relations:
            kg.add_relation(rel)

    return kg



if __name__ == "__main__":
    PDF_PATH = "JS_paper_new_model.pdf"
    try:
        graph = build_knowledge_graph_from_pdf(PDF_PATH)
        print("\nFinished extraction. Knowledge graph summary:\n")
        graph.report()
    except ImportError as e:
        print(f"Missing dependency: {e}")
    except Exception as exc:
        print(f"Error during processing: {exc}")