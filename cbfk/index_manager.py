# Set TOKENIZERS_PARALLELISM environment variable to prevent warnings
import json
import logging
import os
import re
import statistics
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Optional, Tuple

from llama_index.core import Settings, SimpleDirectoryReader, StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.base.response.schema import Response
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import Document, NodeWithScore, TextNode
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from tqdm import tqdm
from transformers import AutoTokenizer

import cbfk.experiment_config
import cbfk.llm_prompter
import cbfk.models
from cbfk.bm25_retriever import BM25Retriever
from cbfk.hybrid_retriever import HybridRetriever
from cbfk.log.log_config import LogConfig as LC

# Configure logging using LogConfig
LC.setup_logging(log_level=logging.INFO)
logger = logging.getLogger(__name__)
# set logging level for SentenceTransformer to ERROR
logging.getLogger("sentence_transformers.SentenceTransformer").setLevel(logging.ERROR)
# set logging level for modeling_hf_nomi to ERROR
logging.getLogger("modeling_hf_nomi").setLevel(logging.ERROR)


EMBED_BATCH_SIZE = 32



class InstructionEmbedding(HuggingFaceEmbedding):
    """
    Extension of HuggingFaceEmbedding that supports instruction-based embeddings.
    Prepends instructions to queries and texts before embedding.
    """
    
    def __init__(
        self,
        model_name: str,
        query_instruction: Optional[str] = None,
        text_instruction: Optional[str] = None,
        **kwargs
    ):
        super().__init__(model_name=model_name, **kwargs)
        self.query_instruction = query_instruction
        self.text_instruction = text_instruction
    
    def _get_query_embedding(self, query: str) -> list[float]:
        """Apply query instruction and get embedding."""
        if self.query_instruction:
            query = f"{self.query_instruction}{query}"
        return super()._get_query_embedding(query)
    
    def _get_text_embedding(self, text: str) -> list[float]:
        """Apply text instruction and get embedding."""
        if self.text_instruction:
            text = f"{self.text_instruction}{text}"
        return super()._get_text_embedding(text)
    
    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Apply text instruction to a batch of texts and get embeddings."""
        if self.text_instruction:
            texts = [f"{self.text_instruction}{text}" for text in texts]
        return super()._get_text_embeddings(texts)


def _get_embedding_model_cache_folder() -> Path:
    return Path('index') / "embedding_model_cache"


def initialize_embedding_model(
    short_name: str,
    query_instruction: Optional[str] = None,
    text_instruction: Optional[str] = None, ) -> HuggingFaceEmbedding:
    """
    Initialize an embedding model with the specified short name.
    For instruction-tuned models, instructions are applied when embedding text.
    
    Args:
        short_name: Short name of the HuggingFace embedding model to use
        query_instruction: Optional instruction for encoding queries
        text_instruction: Optional instruction for encoding text
    
    Returns:
        Initialized embedding model with instruction handling
    """
    # Use the same cache folder for all embedding models to prevent repeated downloads
    cache_folder = _get_embedding_model_cache_folder()
    
    # Get model info from registry
    model_info = cbfk.models.get_model_registry().get_model_info(short_name)
    query_instruction = model_info.query_instruction
    text_instruction = model_info.text_instruction
        
    # Parameters for embedding
    params = {
        "model_name": model_info.repo_id,
        "embed_batch_size": EMBED_BATCH_SIZE,
        "trust_remote_code": True,
        "cache_folder": cache_folder
    }
    if model_info.revision:  params["revision"] = model_info.revision
    
    # Create the embedding model with or without instructions
    if query_instruction or text_instruction:
        embed_model = InstructionEmbedding(
            query_instruction=query_instruction,
            text_instruction=text_instruction,
            **params
        )
    else:
        embed_model = HuggingFaceEmbedding(**params)
    
    return embed_model


def adjust_chunk_size(chunk_size: int, chunk_overlap: int, short_name: str) -> tuple[int, int]:
    """
    Adjust chunk size and overlap based on model capacity.
    
    Args:
        chunk_size: Original chunk size
        chunk_overlap: Original chunk overlap
        short_name: Short name of the embedding model
    Returns:
        tuple[int, int]: Adjusted chunk size and overlap
    """
    model_max_length = cbfk.models.get_model_registry().get_max_sequence_length(short_name)
    
    # Adjust chunk size if it exceeds model's capacity
    # Use 90% of max length to leave some room for special tokens
    max_recommended_chunk_size = int(model_max_length * 0.9)
    
    if chunk_size > max_recommended_chunk_size:
        original_chunk_size = chunk_size
        chunk_size = max_recommended_chunk_size
        logger.warning(f"{LC.AMBER}Chunk size {original_chunk_size} exceeds model max length {model_max_length}{LC.RESET}. "
                        f"Adjusted to {chunk_size} tokens (90% of model's {model_max_length} limit)")
    else:
        logger.info(f"Chunk size {chunk_size} is within {LC.HI}model's capacity {model_max_length}{LC.RESET}")
    
    # Also adjust chunk overlap if needed
    if chunk_overlap > chunk_size * 0.5:
        chunk_overlap = int(chunk_size * 0.3)  # Set to 30% of chunk size
        logger.warning(f"Chunk overlap too large for new chunk size. Adjusted to {chunk_overlap}")
    return chunk_size, chunk_overlap


def get_index_params(persist_dir: Path) -> dict[str, Any]:
    """
    Get the parameters used to create the index.
    
    Args:
        persist_dir: Directory where the index is stored
        
    Returns:
        Dictionary containing the index parameters, or empty dict if not found
    """
    params_file = Path(persist_dir) / "index_stats.json"
    if params_file.exists():
        try:
            with open(params_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load index parameters: {e!s}")
    return {}


def build_augmented_chunk(chunk_text: str, questions: list[str]) -> str:
    if not questions:
        return chunk_text
    questions_text = "\n".join(f"Q: {q}" for q in questions)
    return f"{questions_text}\n\n{chunk_text}"


def _get_all_nodes_from_docstore(index: VectorStoreIndex) -> list[NodeWithScore]:
    all_nodes = []
    try:
        # Get all document IDs
        docstore = index.storage_context.docstore
        all_doc_ids = list(docstore.docs.keys())
        
        # Get all nodes
        for doc_id in all_doc_ids:
            try:
                node = docstore.get_document(doc_id)
                if hasattr(node, 'get_content'):
                    all_nodes.append(node)
            except Exception as e:
                logger.warning(f"Could not retrieve document {doc_id}: {e}")
    except Exception as e:
        logger.error(f"Error accessing docstore: {e}")

    return all_nodes


def _make_hybrid_retriever(
    vector_retriever: BaseRetriever,
    bm25_retriever: BaseRetriever | None,
    config: cbfk.experiment_config.ExperimentConfig) -> BaseRetriever:
    
    logger.info(f"Retriever config: vector top_k={config.similarity_top_k}, bm25 top_k={config.bm25_top_k}")

    # Create the final retriever
    if bm25_retriever is not None:
        retriever = HybridRetriever(vector_retriever, bm25_retriever, config)
    else:
        retriever = vector_retriever
        logger.info("Fallback: using only vector retriever")
    return retriever


def load_index(
    persist_dir: Path,
    config: cbfk.experiment_config.ExperimentConfig,
    use_stored_params: bool = True,
    ) -> Tuple[Any, dict[str, Any], Any]:

    try:
        logger.info(f"Loading index from {LC.HI}{persist_dir}{LC.RESET}")
        
        stored_params = get_index_params(persist_dir) if use_stored_params else {}
        
        # Use config values if provided, otherwise fall back to stored params
        actual_model_name = (config.embedding_model if config else None) or stored_params.get("model_name", "BAAI/bge-small-en-v1.5")
        actual_query_instruction = (config.query_instruction if config else None) or stored_params.get("query_instruction")
        actual_text_instruction = (config.text_instruction if config else None) or stored_params.get("text_instruction")
        stats = {
            "model_name": actual_model_name,
            "query_instruction": actual_query_instruction,
            "text_instruction": actual_text_instruction,
            "stored_params": stored_params
        } 

        # Initialize the embedding model
        embed_model = initialize_embedding_model(actual_model_name, actual_query_instruction, actual_text_instruction) 
        
        # Load the VectorStoreIndex
        index_load_start = time.time()
        storage_context = StorageContext.from_defaults(persist_dir=str(persist_dir))
        index = load_index_from_storage(storage_context, embed_model=embed_model)
        stats["index_load_time"] = time.time() - index_load_start
        vector_retriever = index.as_retriever(similarity_top_k=config.vector_top_k)

        # Try to load BM25 retriever
        # Use config values if provided, otherwise use stored params
        splitter_type_val = config.splitter_type if config else None or stored_params.get("splitter_type")
        if isinstance(splitter_type_val, str):
            splitter_type_val = cbfk.experiment_config.SplitterType[splitter_type_val]
            
        chunk_size_val = config.chunk_size if config else None or stored_params.get("chunk_size")
        chunk_overlap_val = config.chunk_overlap if config else None or stored_params.get("chunk_overlap")
        
        if config.bm25_top_k > 0:
            bm25_configstr = BM25Retriever.make_config_str(splitter_type_val, chunk_size_val, chunk_overlap_val)
            bm25_retriever_path = Path(persist_dir) / f"{bm25_configstr}.pkl"
            if bm25_retriever_path.exists():
                bm25_load_start = time.time()
                bm25_retriever = BM25Retriever.load(bm25_retriever_path)
                stats["bm25_load_time"] = time.time() - bm25_load_start
                logger.info(f"Loaded BM25 retriever from {LC.HI}{bm25_retriever_path}{LC.RESET}")
            else:
                bm25_retriever = None
                logger.error(f"No BM25 retriever found at {LC.AMBER}{bm25_retriever_path}{LC.RESET}")
                raise FileNotFoundError(f"No BM25 retriever found at {bm25_retriever_path}")
        
            retriever = _make_hybrid_retriever(vector_retriever, bm25_retriever, config)
        else:
            retriever = vector_retriever

        return index, stats, retriever

    except Exception as e:
        logger.error(f"Error loading index: {e!s}")
        raise

def _split_into_parts(text: str, file_name: str, max_file_size: float) -> list[Document]:

    def _split_into_paragraphs(text: str, file_name: str) -> list[Document]:
        paragraphs = re.split(r'(?m)^#{1,3} ', text)
        headers = re.findall(r'(?m)^#{1,3} (.*)', text)      
        docs = []
        for i, chunk in enumerate(paragraphs[1:], start=0):  # skip text before first header
            paragraph_title = headers[i].strip() if i < len(headers) else f"{i+1}"
            docs.append(Document(text=chunk.strip(), metadata={
                "file_name": file_name,
                "paragraph": paragraph_title
            }))
        return docs 

    total = len(text)
    num_parts = (total + max_file_size - 1) // max_file_size
    max_part_len = -(-total // num_parts)    

    docs = _split_into_paragraphs(text, file_name)
    parts = []
    current_text = ""
    for _, doc in enumerate(docs):
        current_text = current_text + '\n\n' + doc.text
        if len(current_text) > max_part_len:
            parts.append(Document(text=current_text.strip(), metadata={
                "file_name": file_name,
                "chapter": f"Part {len(parts)+1} of {num_parts}"
            }))
            current_text = ""
    if current_text:
        parts.append(Document(text=current_text.strip(), metadata={
            "file_name": file_name,
            "chapter": f"Part {len(parts)+1} of {num_parts}"
        }))
    return parts


def _load_files_from_dir(corpus_dir: Path, persist_dir: Path, stats: dict[str, Any]) -> list[Document]:
    """
    Load files from directory and split into chapters.
    
    Args:
        corpus_dir: Directory containing documents to ingest
        persist_dir: Directory to store the index and embeddings
    
    Returns:
        tuple[list[Document], dict[str, Any]]: The documents and statistics about the index
    """
    # Create persist directory if it doesn't exist
    persist_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Loading documents from {LC.HI}{corpus_dir}{LC.RESET}...")
    
    # Load files from directory
    load_start = time.time()
    file_docs = SimpleDirectoryReader(input_dir=str(corpus_dir), recursive=True).load_data()
    
    # Split large files into sections 
    max_file_size = 128000
    documents = []
    for file_doc in file_docs:
        file_name = file_doc.metadata.get("file_name", "UnknownFile")
        num_parts = 1
        if len(file_doc.text) > max_file_size:
            parts = _split_into_parts(file_doc.text, file_name, max_file_size)
            num_parts = len(parts)
            documents.extend(parts)
        else:
            documents.append(file_doc)
        logger.info(f"Loaded {LC.HI}{file_name}{LC.RESET}, {LC.HI}{len(file_doc.text):,}{LC.RESET} characters, {LC.HI}{num_parts}{LC.RESET} parts")

    stats["files_total"] = len(file_docs)
    stats["files_parts_total"] = len(documents)
    stats["file_load_time"] = time.time() - load_start

    logger.info(f"Loaded {LC.HI}{len(file_docs)}{LC.RESET} files, {LC.HI}{len(documents)}{LC.RESET} parts")
    return documents


def _augment_chunks(chunks: list[Document], config: cbfk.experiment_config.ExperimentConfig, stats: dict[str, Any]) -> list[Document]:
    # Augment chunks with leading questions about the content
    start_augmenting = time.time()
    
    # generate questions
    question_responses = [cbfk.llm_prompter.prompt_augmenting_question(
                                doc.text, 
                                model=config.augmenting_model, 
                                temperature=config.augmenting_temperature,
                                max_tokens=config.augmenting_max_tokens) 
                            for doc in tqdm(chunks, desc=f"Augmenting questions with {config.augmenting_model}")]

    # merge questions with chunk text
    for doc, response in zip(chunks, question_responses, strict=False):
        original_text = doc.text
        questions = [line.strip()[2:].strip() for line in response.split("\n") if line.strip()[:2] in {"1.", "2.", "3.", "4.", "5."}]
        doc.text = build_augmented_chunk(original_text, questions)
        doc.metadata["original_text"] = original_text
        doc.metadata["questions"] = questions       
    logger.info(f"Augmented {LC.HI}{len(chunks)}{LC.RESET} chunks in {LC.HI}{time.time() - start_augmenting:.2f} seconds{LC.RESET}")
    text_lengths = [len(doc.text) for doc in chunks]
    stats["augmented_chunk_char_avg"] = sum(text_lengths) / len(text_lengths)
    stats["augmented_chunk_char_min"] = min(text_lengths)
    stats["augmented_chunk_char_max"] = max(text_lengths)
    stats["augmented_chunk_char_std_dev"] = statistics.stdev(text_lengths) if text_lengths else 0
    stats["augment_time_sec"] = time.time() - start_augmenting
    logger.info(f"Augmented {LC.HI}{len(chunks)}{LC.RESET} chunks in {LC.HI}{time.time() - start_augmenting:.2f} seconds{LC.RESET}")
    return chunks    


def _chunk_documents(
    documents: list[Document],
    embed_model: HuggingFaceEmbedding,
    chunk_size: int,
    chunk_overlap: int,
    config: cbfk.experiment_config.ExperimentConfig, 
    stats: dict[str, Any]                           ) -> list[Document]:
    """
    Chunk documents into smaller chunks using the specified splitter type and chunk size.
    
    Args:
        documents: List of documents to chunk
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks
        embed_model: Embedding model to use for chunking
        splitter_type: Splitter type to use for chunking
    
    Returns:
        List of chunked documents
    """
    start_time = time.time()
    # Configure text splitter for chunking with customizable parameters
    logger.info(f"Using {LC.HI}{config.splitter_type.name}{LC.RESET} splitter with chunk size: {LC.HI}{chunk_size}{LC.RESET}, chunk overlap: {LC.HI}{chunk_overlap}{LC.RESET}")
    text_splitter = config.splitter_type.get_text_splitter(chunk_size, chunk_overlap, embed_model)
    
    # Set TOKENIZERS_PARALLELISM environment variable to false to prevent warnings from huggingface/tokenizers
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    # Chunk the documents first (needed regardless of augmentation)
    from llama_index.core.node_parser import NodeParser, SimpleNodeParser
    if isinstance(text_splitter, NodeParser):
        # SentenceSplitter or SemanticSplitterNodeParser
        parser = text_splitter
    else:
        # TokenTextSplitter or similar
        parser = SimpleNodeParser(text_splitter=text_splitter)

    #chunks = parser.get_nodes_from_documents(documents)
    from tqdm import tqdm
    chunks = []
    for doc in tqdm(documents, desc=f"Chunking with {config.splitter_type.name}"):
        chunks.extend(parser.get_nodes_from_documents([doc]))    

    # add some stats
    stats["chunk_count"] = len(chunks)
    stats["chunk_length_char_avg"] = sum([len(chunk.text) for chunk in chunks]) / len(chunks)
    stats["chunk_length_char_min"] = min([len(chunk.text) for chunk in chunks])
    stats["chunk_length_char_max"] = max([len(chunk.text) for chunk in chunks])
    stats["chunk_length_char_std_dev"] = statistics.stdev([len(chunk.text) for chunk in chunks])
    stats["chunking_time_sec"] = time.time() - start_time    
    return chunks    


def _create_vector_store(chunks, embed_model, config: cbfk.experiment_config.ExperimentConfig, persist_dir: Path, stats: dict[str, Any]) -> VectorStoreIndex:
    start_indexing = time.time()
    # Initialize storage context with a vector store
    storage_context = StorageContext.from_defaults(
        vector_store=SimpleVectorStore()
    )
    # Embed the documents using the local model
    start_indexing = time.time()        
    index = VectorStoreIndex(
        nodes=chunks,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True,
    )
    logger.info(f"Indexing completed in {LC.HI}{time.time() - start_indexing:.2f} seconds{LC.RESET}")
    stats["indexing_time_sec"] = time.time() - start_indexing
    vector_retriever = index.as_retriever(similarity_top_k=config.vector_top_k)
    
    # Persist the index to disk 
    start_persist = time.time()
    index.storage_context.persist(persist_dir=str(persist_dir))
    logger.info(f"Persisted index to disk in {LC.HI}{time.time() - start_persist:.2f} seconds{LC.RESET}")
    stats["index_persist_time_sec"] = time.time() - start_persist
    return index, vector_retriever


def _create_bm25_index(index: VectorStoreIndex, config: cbfk.experiment_config.ExperimentConfig, persist_dir: Path, chunk_size: int, chunk_overlap: int, stats: dict[str, Any]) -> BM25Retriever:
    # Create bm25 index
    start_bm25 = time.time()            
    if config.augment_chunks:
        # Rebuild BM25-safe nodes with original text
        all_nodes_aug = _get_all_nodes_from_docstore(index)
        all_nodes = []
        for node in all_nodes_aug:
            original_text = node.metadata.get("original_text", node.get_content())
            clean_node = TextNode(
            text=original_text,
            metadata=node.metadata,
            id_=node.node_id,
        )
        all_nodes.append(clean_node)
    else:
        all_nodes = _get_all_nodes_from_docstore(index)
    
    bm25_configstr = BM25Retriever.make_config_str(config.splitter_type, chunk_size, chunk_overlap)
    bm25_save_path = Path(persist_dir) / f"{bm25_configstr}.pkl"
    bm25_save_path.parent.mkdir(parents=True, exist_ok=True)
    bm25_retriever = BM25Retriever.get_bm25_retriever(all_nodes, bm25_configstr, persist_path=bm25_save_path)
    bm25_retriever.save(bm25_save_path)
    logger.info(f"Created and saved BM25 to {LC.HI}{bm25_save_path}{LC.RESET} in {LC.HI}{time.time() - start_bm25:.2f} seconds{LC.RESET}")

    # Update index creation parameters
    stats["bm25_configstr"] = bm25_configstr
    stats["bm25_save_path"] = str(bm25_save_path)
    stats["bm25_num_docs"] = len(bm25_retriever.nodes)
    stats["bm25_num_tokens"] = sum(len(node.get_content().split()) for node in bm25_retriever.nodes)
    stats["bm25_time_sec"] = time.time() - start_bm25
    return bm25_retriever


def ingest_documents(
    corpus_dir: Path,
    persist_dir: Path,
    config: cbfk.experiment_config.ExperimentConfig,
    auto_adjust_chunk_size: bool = True,   ) -> tuple[VectorStoreIndex, dict[str, Any], Any]:

    """
    Ingest documents from specified directory, chunk them, embed them using a local embedding model, and store them locally.

    Args:
        corpus_dir (Path): Directory containing documents to ingest
        persist_dir (Path): Directory to store the index and embeddings
        config (ExperimentConfig): Configuration object containing all experiment parameters
        auto_adjust_chunk_size (bool): Whether to adjust chunk size based on model capacity

    Returns:
        tuple[VectorStoreIndex, dict[str, Any], Any]: The index, statistics about the index, and the hybrid retriever
    """
    
    # Get model max length if auto-adjustment is enabled
    chunk_size = config.chunk_size
    chunk_overlap = config.chunk_overlap
    if auto_adjust_chunk_size:
        chunk_size, chunk_overlap = adjust_chunk_size(chunk_size, chunk_overlap, config.embedding_model)

    # Store index creation parameters
    stats = {
        "embedding_model": config.embedding_model,
        "query_instruction": config.query_instruction,
        "text_instruction": config.text_instruction,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "splitter_type": config.splitter_type.name,
        "creation_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "augment_chunks": config.augment_chunks,
    }
    if config.augment_chunks:
        stats["augmenting_model"] = config.augmenting_model
        stats["augmenting_temperature"] = config.augmenting_temperature
        stats["augmenting_max_tokens"] = config.augmenting_max_tokens
    # Load files from directory
    documents = _load_files_from_dir(corpus_dir, persist_dir, stats)
    
    # Initialize the embedding model using the shared function
    embed_model = initialize_embedding_model(config.embedding_model, config.query_instruction, config.text_instruction)
    # Set the embedding model in the global settings
    Settings.embed_model = embed_model
    
    # Chunk the documents
    chunks = _chunk_documents(documents, embed_model, chunk_size, chunk_overlap, config, stats)
    
    # Chunk augmenting
    if config.augment_chunks:
        chunks = _augment_chunks(chunks, config, stats)
    
    # Create vector store
    index, vector_retriever = _create_vector_store(chunks, embed_model, config, persist_dir, stats)

    # Create bm25 index
    bm25_retriever = _create_bm25_index(index, config, persist_dir, chunk_size, chunk_overlap, stats)

    # Generate statistics about the index and add them to stats
    stats = stats | get_index_content_stats(config, index, bm25_retriever)
    
    # Save parameters to a JSON file in the persist directory
    params_file = Path(persist_dir) / "index_stats.json"
    with open(params_file, 'w') as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Saved index stats to {LC.HI}{params_file}{LC.RESET}")

    # Create hybrid retriever from the vector retriever and bm25 retriever
    retriever = _make_hybrid_retriever(vector_retriever, bm25_retriever, config)

    return index, stats, retriever


def query_index(retriever: cbfk.hybrid_retriever.HybridRetriever, query_text: str) -> Response:
    """Query the index and get a response (supports HybridRetriever)."""
    try:
        retriever.store_retrieved_nodes = True
        nodes_or_chunks = retriever.retrieve(query_text)
        retrieved_nodes = retriever.get_retrieved_nodes()
        retriever.store_retrieved_nodes = False
        
        # Prepare response text
        node_texts = []
        for item in nodes_or_chunks:
            if hasattr(item, "node"):
                node_texts.append(item.node.get_content())  # Vector node
            else:
                node_texts.append(item)  # Raw BM25 chunk string

        combined_text = "\n\n".join(node_texts)
        
        from llama_index.core.base.response.schema import Response
        response = Response(response=combined_text, source_nodes=nodes_or_chunks)
        return response, retrieved_nodes

    except Exception as e:
        logger.error(f"Error querying index: {e!s}")
        raise


def batch_query_index(retriever, queries: list[str], config: cbfk.experiment_config.ExperimentConfig) -> list[Response]:
    """Query the index with multiple queries at once and get responses without using an LLM.
    
    Args:
        retriever: The retriever to use for retrieving nodes
        queries: List of query strings to process
        similarity_top_k: Number of similar nodes to retrieve for each query
        
    Returns:
        List of Response objects corresponding to each query
    """
    try:
        logger.info(f"Batch querying index with {len(queries)} queries")
        
        # Process each query and collect responses
        responses: list[Response] = []
        retrieved_node_dicts: list[dict[str, list[Any]]] = []
        
        display_text = f"with {config.crossencoder_model}" if config.rerank else "without reranking"
        for query_text in tqdm(queries, desc=f"Batch querying index {display_text}", unit="query"):
            try:
                # Retrieve the nodes for this query
                response, retrieved_nodes = query_index(retriever, query_text)
                if isinstance(retriever, HybridRetriever):
                    response.query_rewrite = retriever.last_query_rewrite
                responses.append(response)
                retrieved_node_dicts.append(retrieved_nodes)
                
            except Exception as e:
                logger.error(f"Error processing query '{query_text}': {e!s}\n{traceback.format_exc()}")
                raise
        
        logger.info(f"Batch query completed successfully for {len(responses)} queries")
        return responses, retrieved_node_dicts
        
    except Exception as e:
        logger.error(f"Error in batch query processing: {e!s}")
        # Return a list of error responses matching the length of the input queries
        return [Response(response=f"Error: {e!s}") for _ in queries]


def get_index_content_stats(config: cbfk.experiment_config.ExperimentConfig, index: VectorStoreIndex, bm25_retriever: BM25Retriever = None) -> dict[str, Any]:
    """
    Generate statistics about the index content, including number of chunks, character/word/token counts,
    and other relevant metrics. If a BM25Retriever is provided, also includes BM25-specific statistics.
    TODO: add chunk length statistics
    Args:
        config: The ExperimentConfig to use for getting the embedding model
        index: The VectorStoreIndex to analyze
        bm25_retriever: Optional BM25Retriever to analyze
        
    Returns:
        dict containing various statistics about the index
    """
    # Get all nodes from the docstore
    all_nodes = _get_all_nodes_from_docstore(index)
    
    # Extract content from all nodes
    contents = [node.get_content() for node in all_nodes]
    
    # Calculate basic statistics
    char_counts = [len(content) for content in contents]
    word_counts = [len(content.split()) for content in contents]
    
    # Estimate token counts (rough approximation - 4 chars per token on average)
    #token_counts = [len(content) // 4 for content in contents]
    
    model_info = cbfk.models.get_model_registry().get_model_info(config.embedding_model)
    tokenizer = AutoTokenizer.from_pretrained(model_info.repo_id, cache_folder=_get_embedding_model_cache_folder())
    tokenizer.model_max_length = cbfk.models.get_model_registry().get_max_sequence_length(config.embedding_model)
    token_counts = [len(tokenizer.encode(content, add_special_tokens=False)) for content in contents]    
    
    # Calculate document sources
    doc_sources = [node.metadata.get('file_name', 'Unknown') for node in all_nodes]
    source_counter = Counter(doc_sources)
    
    # Word frequency analysis (top 50 words, excluding common stop words)
    stop_words = {'the', 'and', 'a', 'to', 'of', 'in', 'is', 'that', 'it', 'for', 'with', 'as', 'be', 'on', 'not', 'this'}
    all_words = []
    for content in contents:
        # Convert to lowercase and remove punctuation
        clean_text = re.sub(r'[^\w\s]', '', content.lower())
        words = clean_text.split()
        all_words.extend([w for w in words if w not in stop_words and len(w) > 2])
    
    word_freq = Counter(all_words).most_common(50)
    
    # Compile statistics
    # Prepare sample chunks data with their stats
    sample_chunks = []
    for i in range(min(12, len(contents))):
        sample_chunks.append({
            'content': contents[i],
            'char_count': char_counts[i],
            'word_count': word_counts[i],
            'token_count': token_counts[i],
            'source': doc_sources[i] if i < len(doc_sources) else 'Unknown'
        })
    
    stats = {
        'total_chunks': len(contents),
        'total_chars': sum(char_counts),
        'total_words': sum(word_counts),
        'total_tokens_est': sum(token_counts),
        'avg_chars_per_chunk': statistics.mean(char_counts) if char_counts else 0,
        'avg_words_per_chunk': statistics.mean(word_counts) if word_counts else 0,
        'avg_tokens_per_chunk': statistics.mean(token_counts) if token_counts else 0,
        'min_chars': min(char_counts) if char_counts else 0,
        'max_chars': max(char_counts) if char_counts else 0,
        'min_words': min(word_counts) if word_counts else 0,
        'max_words': max(word_counts) if word_counts else 0,
        'min_tokens': min(token_counts) if token_counts else 0,
        'max_tokens': max(token_counts) if token_counts else 0,
        'std_dev_chars': statistics.stdev(char_counts) if len(char_counts) > 1 else 0,
        'std_dev_words': statistics.stdev(word_counts) if len(word_counts) > 1 else 0,
        'std_dev_tokens': statistics.stdev(token_counts) if len(token_counts) > 1 else 0,
        'char_count_distribution': char_counts,
        'word_count_distribution': word_counts,
        'token_count_distribution': token_counts,
        'source_distribution': dict(source_counter),
        'top_words': dict(word_freq),
        'sample_chunks': sample_chunks
    }
    
    # Add BM25 statistics if a retriever is provided
    if bm25_retriever is not None:
        # Extract BM25 configuration
        bm25_config = {
            'configstr': bm25_retriever.configstr,
            'from_cache': bm25_retriever.from_cache,
            'num_documents': len(bm25_retriever.nodes),
        }
        
        # Analyze the tokenized corpus
        token_lengths = [len(doc) for doc in bm25_retriever.tokenized_corpus]
        unique_token_counts = [len(set(doc)) for doc in bm25_retriever.tokenized_corpus]
        
        # Calculate vocabulary statistics
        all_tokens = [token for doc in bm25_retriever.tokenized_corpus for token in doc]
        unique_tokens = set(all_tokens)
        token_frequency = Counter(all_tokens)
        
        # Extract IDF values for common tokens (top 20)
        common_tokens = [token for token, _ in token_frequency.most_common(20)]
        idf_values = {token: bm25_retriever.bm25.idf.get(token, 0) for token in common_tokens}
        
        # Add BM25 statistics to the overall stats
        stats['bm25'] = {
            'config': bm25_config,
            'corpus_stats': {
                'total_tokens': len(all_tokens),
                'unique_tokens': len(unique_tokens),
                'avg_tokens_per_doc': statistics.mean(token_lengths) if token_lengths else 0,
                'min_tokens': min(token_lengths) if token_lengths else 0,
                'max_tokens': max(token_lengths) if token_lengths else 0,
                'std_dev_tokens': statistics.stdev(token_lengths) if len(token_lengths) > 1 else 0,
                'avg_unique_tokens': statistics.mean(unique_token_counts) if unique_token_counts else 0,
                'token_length_distribution': token_lengths,
            },
            'vocabulary': {
                'top_tokens': dict(token_frequency.most_common(50)),
                'idf_sample': idf_values,
                'avg_idf': statistics.mean(list(bm25_retriever.bm25.idf.values())) if bm25_retriever.bm25.idf else 0,
            },
            'parameters': {
                'k1': bm25_retriever.bm25.k1,
                'b': bm25_retriever.bm25.b,
                'epsilon': bm25_retriever.bm25.epsilon,
            }
        }
    
    return stats
