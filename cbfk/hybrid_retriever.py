# cbfk/hybrid_retriever.py

import logging

from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore
from sentence_transformers import CrossEncoder

from cbfk.experiment_config import ExperimentConfig

# Configure logging using LogConfig
from cbfk.log.log_config import LogConfig as LC
from cbfk.query_rewriter import QueryRewriter

LC.setup_logging(log_level=logging.INFO)
logger = logging.getLogger(__name__)



class HybridRetriever(BaseRetriever):
    """
    Hybrid Retriever: Actually a super hybrid retriever. 
    - Combines vector and BM25 retrievers.
    - Deduplicates based on text content.
    - Reranks
    Config parameters:
        similarity_top_k: Number of nodes to return in the final result.
        vector_top_k: Number of nodes to retrieve from the vector retriever.
        bm25_top_k: Number of nodes to retrieve from the BM25 retriever.
        deduplicate: Whether to deduplicate nodes based on text content.
        rerank: Whether to rerank nodes based on the cross-encoder.
        crossencoder_model: The cross-encoder model to use for reranking.
    When deduplicate or rerank are enabled, vector_top_k nodes are retrieved from vector store, then merged 
    with bm25_top_k nodes from BM25 retriever, then deduplicated and reranked. Then the top similarity_top_k nodes are returned.
    """
    def __init__(
        self,
        vector_retriever: object,
        bm25_retriever: object,
        config: ExperimentConfig,    ):
        """       
        Args:
            vector_retriever: Must have a .retrieve(query_text) -> List[NodeWithScore] method.
            bm25_retriever: Must have a .retrieve(query_text, top_k) -> List[NodeWithScore] method.
            config: Experiment configuration
        """
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.config = config
        if self.config.rerank:
            self.crossencoder = CrossEncoder(config.crossencoder_model)
        self.last_query_rewrite = None
        self.retrieved_nodes: dict[str, list[NodeWithScore]] = {}
        self.store_retrieved_nodes = False


    def get_retrieved_nodes(self) -> dict[str, list[NodeWithScore]]:
        """Return the retrieved nodes for each retriever and query text.
        Returns:
            dict[str, list[NodeWithScore]]: 
                A dictionary where the keys are the retriever names and the values are lists of retrieved nodes.
        """
        ret = self.retrieved_nodes
        self.retrieved_nodes = {}
        return ret


    def _add_retrieved_nodes(self, retriever: str, nodes: list[NodeWithScore]):
        if not self.store_retrieved_nodes:
            return
        if retriever not in self.retrieved_nodes:
            self.retrieved_nodes[retriever] = []
        self.retrieved_nodes[retriever].extend(nodes)


    def _rerank(self, query_bundle, nodes: list[NodeWithScore]) -> list[NodeWithScore]:
        # Rerank nodes based on cross-encoder scores
        input_pairs = [[query_bundle, doc.node.get_content()] for doc in nodes]
        scores = self.crossencoder.predict(input_pairs, show_progress_bar=False)

        # Combine documents with their scores and sort descending
        results = list(zip(nodes, scores, strict=False))
        results = sorted(results, key=lambda x: x[1], reverse=True)
        
        # Add reranked score to metadata and extract nodes
        reranked_nodes = []
        for node, score in results:
            if node.metadata is None:
                node.metadata = {}
            node.metadata["retriever_score"] = node.score # Save the original score
            node.score = float(score)  
            reranked_nodes.append(node)
        return reranked_nodes


    def _combinded_retrieve(self, query_text: str, query_origin: str) -> list[NodeWithScore]:
        # Retrieve nodes from vector and BM25 retrievers and return combined reusult

        # retrieve nodes from vector retriever
        vector_nodes = self.vector_retriever.retrieve(query_text)
        for node in vector_nodes:
            node.node.metadata["retriever_source"] = "vector"
            node.node.metadata["query_origin"] = query_origin
        self._add_retrieved_nodes("vector", vector_nodes)

        # retrieve nodes from BM25 retriever
        bm25_nodes = self.bm25_retriever.retrieve(query_text, top_k=self.config.bm25_top_k)
        for node in bm25_nodes:
            node.node.metadata["retriever_source"] = "bm25"
            node.node.metadata["query_origin"] = query_origin
        self._add_retrieved_nodes("bm25", bm25_nodes)

        # combine nodes from vector and BM25 retrievers in a single list, 
        combined = vector_nodes + bm25_nodes
        return combined


    def _retrieve(self, query_bundle, **kwargs) -> list[NodeWithScore]:
        """
        Implementation of the abstract method required by BaseRetriever.
        Combines vector and BM25 retrieval results.
        Deduplicates, reranks and truncates according to self.config.
        """
        query_text = query_bundle #.query_str
        result = self._combinded_retrieve(query_text, **kwargs)
        if self.config.deduplicate:
            result = self._deduplicate(result)
        if self.config.rerank:
            result = self._rerank(query_text, result)
        result = result[:self.config.similarity_top_k]        
        self._add_retrieved_nodes("combined", result)
        return result


    def _deduplicate(self, combined: list[NodeWithScore]) -> list[NodeWithScore]:
        seen_texts = set()
        unique_combined = []
        for item in combined:
            text = item.node.get_content()
            if text not in seen_texts:
                unique_combined.append(item)
                seen_texts.add(text)
        return unique_combined


    def retrieve(self, user_query: str) -> list[NodeWithScore]:
        
        user_combined = self._retrieve(user_query, query_origin="user")
        
        if self.config.query_rewrite:
            self.last_query_rewrite = QueryRewriter.rewrite(user_query, self.config)
            rewritten_combined = self._retrieve(self.last_query_rewrite, query_origin="rewrite")
            combined = user_combined + rewritten_combined
        else:
            combined = user_combined

        combined.sort(key=lambda x: x.score, reverse=True)
        if self.config.deduplicate or self.config.rerank or self.config.query_rewrite:
            result = combined[:self.config.similarity_top_k]
        else:
            result = combined
        return result   