#!/usr/bin/env python3
import logging
import sys
from pathlib import Path

import tiktoken

# Import project modules
import cbfk.experiment_config
import cbfk.index_manager
from cbfk.log.log_config import LogConfig as LC
from insights.plot_utils import PlotUtils as PU

# Set up logging
LC.setup_logging(log_level=logging.INFO)
logger = logging.getLogger(__name__)

def load_index_from_dir(persist_dir: Path, embedding_model: str):
    """Load index from directory"""
    experiment = cbfk.experiment_config.ExperimentConfig(
                                llm_model = "no llm", 
                                llm_temperature = 0.0,
                                llm_max_tokens = 2000,
                                similarity_top_k = 10,
                                bm25_top_k = 0,
                                vector_top_k = 10,
                                rag_prompt = "no prompt",
                                embedding_model = embedding_model,
                                splitter_type = cbfk.experiment_config.SplitterType.SENTENCE,
                                chunk_size = 512,
                                chunk_overlap_pct = 0.1,   
                                )
    index, _, _ = cbfk.index_manager.load_index(persist_dir, experiment, True)
    logger.info(f"Loaded index from {persist_dir}")
    return index

def count_tokens(text: str) -> int:
    """Count tokens in text using tiktoken"""
    try:
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        return len(encoding.encode(text))
    except Exception as e:
        logger.warning(f"Error counting tokens: {e}")
        # Rough approximation if tiktoken fails
        return len(text.split()) * 1.3

def extract_chunks_from_index(index) -> list[dict]:
    """Extract all chunks from the index with their metadata"""
    docstore = index.storage_context.docstore.docs
    
    chunks = []
    for node_id, node in docstore.items():
        text = node.text
        metadata = node.metadata
        origin = metadata.get("file_name", "Unknown")
        token_count = count_tokens(text)
        chunks.append({
            "id": node_id,
            "text": text,
            "tokens": token_count,
            "origin": origin
        })
    
    return chunks

def print_largest_chunks(folders: list[Path]):
    """Print the 3 largest chunks from all indexes in the provided folders"""
    all_chunks = []
    
    for folder in folders:
        # Parse folder name to get model name
        folder_parts = folder.name.split('_')
        if len(folder_parts) >= 2:
            model = folder_parts[1]
        else:
            model = "unknown"
        
        try:
            index = load_index_from_dir(folder, model)
            chunks = extract_chunks_from_index(index)
            all_chunks.extend(chunks)
            logger.info(f"Extracted {len(chunks)} chunks from {folder}")
        except Exception as e:
            logger.error(f"Error processing folder {folder}: {e}")
    
    # Sort by token count in descending order
    all_chunks.sort(key=lambda x: x["tokens"], reverse=True)
    
    # Print the 3 largest chunks
    print(f"\n{'='*80}\nTHE 3 LARGEST CHUNKS\n{'='*80}")
    for i, chunk in enumerate(all_chunks[:3], 1):
        print(f"\nCHUNK #{i}")
        print(f"Origin: {chunk['origin']}")
        print(f"Token count: {chunk['tokens']}")
        print(f"Content:\n{'-'*40}\n{chunk['text']}\n{'-'*40}")

def main():
    parser = PU._get_parser()
    args = parser.parse_args()
    
    if not args.folders or len(args.folders) == 0:
        print("Error: No folders specified. Use --folders to specify index directories.")
        sys.exit(1)
    
    print_largest_chunks(args.folders)

if __name__ == "__main__":
    main()
