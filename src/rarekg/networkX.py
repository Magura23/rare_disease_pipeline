from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set, Tuple
import networkx as nx


@dataclass
class KnowledgeGraph:
   
    graph: nx.Graph = field(default_factory=nx.Graph)

    def add_entity(self, node_id: str, entity_type: str, pubmed_papers: Iterable[str]) -> None:
       
       
        new_citations: Set[str] = {str(p) for p in pubmed_papers} if pubmed_papers else set()

        if node_id not in self.graph:
            
            self.graph.add_node(node_id, entity_type=entity_type or "Unknown", pubmed_papers=new_citations)
        else:
          
            node_data = self.graph.nodes[node_id]
            existing_citations: Set[str] = node_data.get("pubmed_papers", set())
            if not isinstance(existing_citations, set):
                existing_citations = set(existing_citations)
            existing_citations.update(new_citations)
            node_data["pubmed_papers"] = existing_citations

    def _initialize_edge(self, node1_id: str, node2_id: str) -> None:
        if not self.graph.has_edge(node1_id, node2_id):
        
            self.graph.add_edge(node1_id, node2_id, relations={})
        else:
           
            edge_data = self.graph.edges[node1_id, node2_id]
            if "relations" not in edge_data or not isinstance(edge_data["relations"], dict):
                edge_data["relations"] = {}

    def add_relation(
        self,
        node1_id: str,
        node2_id: str,
        relation_type: str,
        pubmed_papers: Iterable[str],
        node1_type: Optional[str] = None,
        node2_type: Optional[str] = None,
    ) -> None:
    
        if node1_id not in self.graph:
            self.add_entity(node1_id, node1_type or "Unknown", [])
        if node2_id not in self.graph:
            self.add_entity(node2_id, node2_type or "Unknown", [])

        self._initialize_edge(node1_id, node2_id)

     
        new_citations: Set[str] = {str(p) for p in pubmed_papers} if pubmed_papers else set()

        edge_data = self.graph.edges[node1_id, node2_id]
        relations: Dict[str, Set[str]] = edge_data.get("relations", {})

       
        if relation_type in relations:
            existing_citations = relations[relation_type]
            if not isinstance(existing_citations, set):
             
                existing_citations = set(existing_citations)
            existing_citations.update(new_citations)
            relations[relation_type] = existing_citations
        else:
      
            relations[relation_type] = new_citations

        edge_data["relations"] = relations

    def get_entity(self, node_id: str) -> Optional[Dict[str, List[str]]]:
        
        if node_id not in self.graph:
            return None
        node_data = self.graph.nodes[node_id]
        citations: Set[str] = node_data.get("pubmed_papers", set())
        if not isinstance(citations, set):
            citations = set(citations)
        return {
            "entity_type": node_data.get("entity_type", "Unknown"),
            "pubmed_papers": sorted(citations),
        }

    def get_relation(self, node1_id: str, node2_id: str) -> Optional[Dict[str, List[str]]]:
        
        if not self.graph.has_edge(node1_id, node2_id):
            return None
        edge_data = self.graph.edges[node1_id, node2_id]
        relations: Dict[str, Set[str]] = edge_data.get("relations", {})
       
        return {rel: sorted(cits if isinstance(cits, set) else set(cits)) for rel, cits in relations.items()}

    def __contains__(self, node_id: str) -> bool:
   
        return node_id in self.graph

    def __len__(self) -> int:
      
        return self.graph.number_of_nodes()