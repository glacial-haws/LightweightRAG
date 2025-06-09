

from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from typing import Any, Optional

from llama_index.core.node_parser import SemanticSplitterNodeParser, SentenceSplitter, TokenTextSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


class SplitterType(Enum):
    """Enum for different types of text splitters."""
    TOKEN = auto()        # TokenTextSplitter
    SENTENCE = auto()     # SentenceSplitter
    SEMANTIC = auto()     # SemanticSplitterNodeParser

    def get_text_splitter(self, chunk_size: int, chunk_overlap: int, embed_model: HuggingFaceEmbedding):
        # Initialize the appropriate text splitter based on the splitter_type
        if self == SplitterType.TOKEN:
            text_splitter = TokenTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
        elif self == SplitterType.SENTENCE:
            text_splitter = SentenceSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
        elif self == SplitterType.SEMANTIC:
            # SemanticSplitterNodeParser requires an embedding model
            text_splitter = SemanticSplitterNodeParser(
                chunk_size=int(chunk_size*.5),      # it uses this as a guide line, not a rule
                buffer_size=int(chunk_overlap*.5),  # ... so we add some head room
                similarity_threshold=0.4,    # determines how quicky a new chunk is made, lower is smaller chunks
                embed_model=embed_model
            )
        else:
            raise ValueError(f"Unsupported splitter type: {self}")    
        return text_splitter

    @classmethod
    @classmethod
    def from_str(cls, splitter_type: str):
        return cls[splitter_type.upper()]



@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run."""
   
    # Chatbot LLM parameters
    llm_model: str
    llm_temperature: float
    llm_max_tokens: int

    # Retrieval parameters
    similarity_top_k: int
    bm25_top_k: int
    vector_top_k: int
    rag_prompt: str
    
    # Ingestion parameters
    embedding_model: str
    splitter_type: SplitterType
    chunk_size: int
    chunk_overlap_pct: float 
    chunk_overlap: int = 20
    use_cached_index: bool = True
    
    # Augmenting parameters
    augment_chunks: bool = False
    augmenting_model: str = "gemma3:1b"
    augmenting_temperature: float = 0.7
    augmenting_max_tokens: int = 4096

    # Hybrid retriever parameters  
    deduplicate: bool = True
    rerank: bool = True
    crossencoder_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    query_rewrite: bool = True  
    query_rewrite_model: str = "mistral:7b"
    query_rewrite_temperature: float = 0.7
    query_rewrite_max_tokens: int = 4096

    # Evaluation parameters
    evaluate_graded_accuracy: bool = False
    evaluating_model: str = "gemma3:1b"
    evaluating_temperature: float = 0.7
    evaluating_max_tokens: int = 4096

    # Extras
    experiment_id: str = ""
    skip_llm: bool = False
    corpus_path: str = ""

    # Optional query instructions for embedding model
    query_instruction: Optional[str] = None
    text_instruction: Optional[str] = None


    def __post_init__(self):
        """Init and generate a unique experiment ID if not provided."""
        # Set query and text instructions for Nomic embedding models
        if 'nomic' in self.embedding_model:
            self.query_instruction = "search_query: "
            self.text_instruction = "search_document: "
        
        # Calculate chunk overlap from percentage
        if self.chunk_overlap_pct > 0:
            self.chunk_overlap = self.calc_chunk_overlap_from_pct(self.chunk_size, self.chunk_overlap_pct)
        
        # Generate experiment ID if not provided
        if not self.experiment_id:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Get splitter type name
            splitter_name = self.splitter_type.name.lower()
            short_embedding_model_name = self.embedding_model.split('/')[-1]
            short_embedding_model_name = short_embedding_model_name.split('@')[0]
            self.short_embedding_model_name = short_embedding_model_name            
            self.experiment_id = (
                f"exp_{timestamp}_"
                f"{short_embedding_model_name}_"
                f"{splitter_name}_"
                f"{self.chunk_size}-{self.chunk_overlap}_"
                f"bm{self.bm25_top_k}_"
                f"v{self.vector_top_k}_"
                f"k{self.similarity_top_k}_"
                f"{self.llm_model}_"
                f"t{self.llm_temperature}_"
                f"mxt{self.llm_max_tokens}_"
            )
            if self.deduplicate: self.experiment_id += "Ddup"
            if self.rerank: self.experiment_id += "Rrnk"
            if self.query_rewrite: self.experiment_id += "Qrew"
            if self.augment_chunks: self.experiment_id += "Augm"

    def calc_chunk_overlap_from_pct(self, chunk_size: int, chunk_overlap_pct: float) -> int:
        # Calculate chunk overlap based on chunk size and overlap percentage
        # Limit minimum chunk_overlap to 20 tokens
        # Limit maximum chunk_overlap to 40% of chunk size
        chunk_overlap = max(int(chunk_size * chunk_overlap_pct), min(20, int(chunk_size * 0.4)))
        return chunk_overlap

    def to_dict(self) -> dict[str, Any]:
        """Convert the configuration to a dictionary."""
        return {
            "embedding_model": self.embedding_model,
            "chunk_size": self.chunk_size,
            "chunk_overlap_pct": self.chunk_overlap_pct,
            "chunk_overlap": self.chunk_overlap,
            "splitter_type": self.splitter_type.name,
            "similarity_top_k": self.similarity_top_k,
            "vector_top_k": self.vector_top_k,
            "llm_model": self.llm_model,
            "llm_temperature": self.llm_temperature,
            "llm_max_tokens": self.llm_max_tokens,
            "rag_prompt": self.rag_prompt,
            "bm25_top_k": self.bm25_top_k,
            "deduplicate": self.deduplicate,
            "rerank": self.rerank,
            "crossencoder_model": self.crossencoder_model,
            "query_rewrite": self.query_rewrite,
            "query_rewrite_model": self.query_rewrite_model,
            "query_rewrite_temperature": self.query_rewrite_temperature,
            "query_rewrite_max_tokens": self.query_rewrite_max_tokens,
            "augment_chunks": self.augment_chunks,
            "augmenting_model": self.augmenting_model,
            "augmenting_temperature": self.augmenting_temperature,
            "augmenting_max_tokens": self.augmenting_max_tokens,
            "evaluate_graded_accuracy": self.evaluate_graded_accuracy,
            "evaluating_model": self.evaluating_model,
            "evaluating_temperature": self.evaluating_temperature,
            "evaluating_max_tokens": self.evaluating_max_tokens,
        }

