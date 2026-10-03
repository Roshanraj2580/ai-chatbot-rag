"""
Knowledge Graph Engine for Graph RAG (Entity-Relationship Graph Extraction).
Constructs node-and-edge network graphs from ingested document chunks,
enabling multi-hop relational retrieval and visual graph exploration.
"""

import re
import logging
from typing import Dict, List, Any, Set, Tuple
from collections import Counter, defaultdict

logger = logging.getLogger(__name__)

# Common stopwords to exclude from entity extraction
STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "with",
    "by", "about", "against", "between", "into", "through", "during", "before",
    "after", "above", "below", "from", "up", "down", "in", "out", "over", "under",
    "again", "further", "then", "once", "here", "there", "when", "where", "why",
    "how", "all", "any", "both", "each", "few", "more", "most", "other", "some",
    "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very",
    "can", "will", "just", "don", "should", "now", "this", "that", "these", "those",
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having",
    "do", "does", "did", "doing", "would", "could", "must", "shall", "page", "document"
}


def extract_entities_from_text(text: str, max_entities: int = 15) -> List[str]:
    """
    Extract significant entities and key domain concepts from text chunks.
    Uses pattern recognition for capitalized noun phrases, technical acronyms, and domain keywords.
    """
    if not text:
        return []

    # 1. Match acronyms and capitalized multi-word phrases (e.g. "ChromaDB", "FastAPI", "Machine Learning")
    capitalized_phrase_pattern = r'\b[A-Z][a-zA-Z0-9_\-\.]{2,}(?:\s+[A-Z][a-zA-Z0-9_\-\.]{2,})*\b'
    candidates = re.findall(capitalized_phrase_pattern, text)

    # 2. Extract technical and domain keywords
    tech_pattern = r'\b(?:Python|Flask|FastAPI|Redis|SQLite|ChromaDB|Gemini|Docker|Hadoop|Spark|PySpark|PyTorch|TensorFlow|AWS|S3|RAG|LLM|API|Embeddings|Vector|Pipeline|Database|MongoDB|JanusGraph)\b'
    tech_matches = re.findall(tech_pattern, text, re.IGNORECASE)

    raw_entities = candidates + tech_matches

    # Clean and filter
    filtered = []
    for entity in raw_entities:
        clean = entity.strip()
        if len(clean) < 3 or clean.lower() in STOPWORDS:
            continue
        filtered.append(clean)

    # Return most frequent
    counts = Counter(filtered)
    return [item[0] for item in counts.most_common(max_entities)]


def build_knowledge_graph(chroma_client) -> Dict[str, Any]:
    """
    Build a complete Node-and-Edge Knowledge Graph from vector store document chunks.
    
    Returns:
        Dict with nodes, edges, and summary graph statistics.
    """
    if not chroma_client:
        return {"nodes": [], "edges": [], "stats": {"total_nodes": 0, "total_edges": 0, "documents": 0, "entities": 0}}

    try:
        data = chroma_client.get_all_documents()
    except Exception as e:
        logger.error(f"Failed to fetch documents for Knowledge Graph: {e}")
        return {"nodes": [], "edges": [], "stats": {"total_nodes": 0, "total_edges": 0, "documents": 0, "entities": 0}}

    metadatas = data.get("metadatas", [])
    documents_text = data.get("documents", [])

    if not metadatas:
        return {
            "nodes": [
                {"id": "demo_rag", "label": "RAG System", "group": "system", "shape": "box", "color": "#0d6efd", "size": 30},
                {"id": "demo_redis", "label": "Redis Cache", "group": "cache", "shape": "dot", "color": "#ffc107", "size": 20},
                {"id": "demo_chroma", "label": "ChromaDB Vectors", "group": "storage", "shape": "dot", "color": "#198754", "size": 20},
                {"id": "demo_gemini", "label": "Gemini 3.8 Flash", "group": "model", "shape": "star", "color": "#0dcaf0", "size": 25}
            ],
            "edges": [
                {"from": "demo_rag", "to": "demo_redis", "label": "sub-5ms cache"},
                {"from": "demo_rag", "to": "demo_chroma", "label": "dense retrieval"},
                {"from": "demo_rag", "to": "demo_gemini", "label": "reasoning"}
            ],
            "stats": {"total_nodes": 4, "total_edges": 3, "documents": 0, "entities": 4}
        }

    # Document aggregation
    docs_map = {}
    doc_entities = defaultdict(Counter)

    for i, meta in enumerate(metadatas):
        doc_id = meta.get("doc_id", "default_doc")
        source = meta.get("source_filename", "Document")
        
        if doc_id not in docs_map:
            docs_map[doc_id] = {
                "id": f"doc_{doc_id}",
                "label": source,
                "title": f"Document: {source}",
                "group": "document",
                "shape": "box",
                "color": "#0d6efd",
                "font": {"color": "#ffffff", "face": "system-ui"},
                "size": 25
            }

        text = documents_text[i] if i < len(documents_text) else ""
        if text:
            chunk_entities = extract_entities_from_text(text, max_entities=8)
            for ent in chunk_entities:
                doc_entities[doc_id][ent] += 1

    nodes = list(docs_map.values())
    edges = []
    entity_nodes = {}
    co_occurrence = defaultdict(int)

    # Build entity nodes and document-to-entity edges
    for doc_id, entities in doc_entities.items():
        doc_node_id = f"doc_{doc_id}"
        top_entities = [item[0] for item in entities.most_common(12)]

        for ent in top_entities:
            ent_id = f"ent_{re.sub(r'[^a-zA-Z0-9_]', '_', ent.lower())}"
            if ent_id not in entity_nodes:
                entity_nodes[ent_id] = {
                    "id": ent_id,
                    "label": ent,
                    "title": f"Entity: {ent}",
                    "group": "entity",
                    "shape": "dot",
                    "color": "#198754",
                    "size": 14
                }
            
            # Document -> Entity edge
            edges.append({
                "from": doc_node_id,
                "to": ent_id,
                "label": "mentions",
                "color": {"color": "#94a3b8", "opacity": 0.6},
                "arrows": "to"
            })

        # Track entity co-occurrences within document
        for i in range(len(top_entities)):
            for j in range(i + 1, min(i + 3, len(top_entities))):
                pair = tuple(sorted([top_entities[i], top_entities[j]]))
                co_occurrence[pair] += 1

    # Add entity nodes
    nodes.extend(entity_nodes.values())

    # Build entity-to-entity co-occurrence edges (Graph relations)
    for (ent_a, ent_b), weight in co_occurrence.items():
        id_a = f"ent_{re.sub(r'[^a-zA-Z0-9_]', '_', ent_a.lower())}"
        id_b = f"ent_{re.sub(r'[^a-zA-Z0-9_]', '_', ent_b.lower())}"
        if id_a in entity_nodes and id_b in entity_nodes:
            edges.append({
                "from": id_a,
                "to": id_b,
                "label": f"relates ({weight})",
                "color": {"color": "#6c757d", "opacity": 0.4},
                "dashes": True
            })

    stats = {
        "total_nodes": len(nodes),
        "total_edges": len(edges),
        "documents": len(docs_map),
        "entities": len(entity_nodes)
    }

    return {
        "nodes": nodes,
        "edges": edges,
        "stats": stats
    }
