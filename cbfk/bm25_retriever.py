# cbfk/bm25_retriever.py

import pickle
from pathlib import Path
from typing import Any, Mapping, Sequence

from llama_index.core.schema import Node, NodeWithScore, TextNode
from rank_bm25 import BM25Okapi


class BM25Retriever:
    """
    A very thin BM25 wrapper.

    * `self.nodes` keeps the original `Node` objects, so their `.metadata`
      (file name, chapter, etc.) is still available when we surface a result.
    * `self.tokenized_corpus` is built once at construction time and cached.
    """

    def __init__(self,
                 nodes: Sequence[Node],
                 configstr: str,
                 from_cache: bool = False):
        self.nodes: list[Node] = list(nodes)
        self.configstr = configstr
        self.from_cache = from_cache
        # simple whitespace tokenizer over the *text* of each node
        self.tokenized_corpus: list[list[str]] = [
            n.get_content().lower().split() for n in self.nodes
        ]
        if not self.tokenized_corpus or all(len(toks) == 0 for toks in self.tokenized_corpus):
            raise ValueError("BM25Retriever: tokenized_corpus is empty. Ensure all nodes have non-empty text.")
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def retrieve(self, query: str, top_k: int = 5) -> list[NodeWithScore]:
        """Retrieve top-k documents for a given query, returning NodeWithScore objects."""
        tokenized_query = query.lower().split()
        doc_scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:top_k]
        # Normalize scores to [0, 1] (1.0 = highest score for this query)
        max_score = max([doc_scores[i] for i in top_indices], default=0.0)
        results: list[NodeWithScore] = []
        for i in top_indices:
            raw_node = self.nodes[i]                    # ← keeps original metadata
            raw_score = float(doc_scores[i])
            norm_score = raw_score / max_score if max_score > 0 else 0.0
            results.append(NodeWithScore(node=raw_node, score=norm_score))
        return results

    def save(self, persist_path: Path):
        """Persist the bare minimum: (content, metadata) pairs are pickle-safe."""
        if not self.from_cache:
            persist_path.parent.mkdir(parents=True, exist_ok=True)
            compact = [
                {"text": n.get_content(), "metadata": dict(n.metadata)}
                for n in self.nodes
            ]
            with open(persist_path, "wb") as f:
                pickle.dump(compact, f)

    # get_persist_path method removed - path construction is now the caller's responsibility

    @classmethod
    def make_config_str(cls, splitter_type: Any, chunk_size: int, chunk_overlap: int, ) -> str:
        return f"bm25_{splitter_type.name}_{chunk_size}-{chunk_overlap}"

    @classmethod
    def load(cls, persist_path: Path) -> "BM25Retriever":
        """Reverse of `save` — re-inflate `Node`s with metadata."""
        with open(persist_path, "rb") as f:
            compact: list[Mapping[str, Any]] = pickle.load(f)

        nodes: list[Node] = []
        for doc in compact:
            n = TextNode(text=doc["text"])
            n.metadata.update(doc.get("metadata", {}))
            nodes.append(n)

        return cls(nodes, persist_path.stem, from_cache=True)

    @classmethod
    def get_bm25_retriever(cls, index_nodes: list[Node], configstr: str, persist_path: Path | None = None) -> "BM25Retriever":
        """Get a BM25Retriever instance, either from cache or by creating a new one.
        
        Args:
            index_nodes: List of nodes to create the retriever from
            configstr: Configuration string for the BM25 retriever
            persist_path: Optional path to load from or save to. If None, no persistence is used.
            
        Returns:
            BM25Retriever instance
        """
        if persist_path is not None and persist_path.exists():
            return cls.load(persist_path)

        return cls(index_nodes, configstr)
