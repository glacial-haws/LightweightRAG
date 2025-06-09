#!/usr/bin/env python3
"""
Script to systematically explore the parameter space of the document indexing and retrieval system.
"""
import json
import logging
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Tuple

import portalocker

import cbfk.experiment_config
import cbfk.log.log_config
import cbfk.log.wandb_logger
from cbfk.log.log_config import LogConfig as LC

# Configure logging
LC.setup_logging(log_level=logging.INFO)
logger = logging.getLogger(__name__)



def _make_json_serializable(obj: object) -> object:
    """Recursively convert objects to something JSON serializable."""
    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_json_serializable(v) for v in obj]
    # Try common serialization methods
    if hasattr(obj, 'to_dict') and callable(obj.to_dict):
        return _make_json_serializable(obj.to_dict())
    if hasattr(obj, 'dict') and callable(obj.dict):
        return _make_json_serializable(obj.dict())
    if hasattr(obj, '__dict__'):
        return _make_json_serializable(obj.__dict__)
    # Fallback to string
    try:
        json.dumps(obj)
        return obj
    except Exception:
        return str(obj)


class ChunkSizeTooLargeError(Exception):
    """Exception raised when chunk size is larger than model sequence length."""
    pass


class ParameterExplorer:
    """Class to handle parameter space exploration for the document system."""
    
    def __init__(
        self, 
        corpus_dir: Path,
        results_dir: Path,
        base_persist_dir: Path,
        wandb_logger: cbfk.log.wandb_logger.WandbLogger):
        """
        Initialize the parameter explorer.
        
        Args:
            corpus_dir: Directory containing documents to ingest
            results_dir: Directory to store experiment results
            base_persist_dir: Base directory to store indices
            wandb_logger: wandb logger instance
        """
        self.corpus_dir = corpus_dir
        self.results_dir = results_dir
        self.base_persist_dir = base_persist_dir
        self.wandb_logger = wandb_logger
        
        # Create directories if they don't exist
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.base_persist_dir.mkdir(parents=True, exist_ok=True)
        
        # Results storage
        self.results = []


    def get_persist_dir(self, config: cbfk.experiment_config.ExperimentConfig) -> Path:
        """
        Get the persistence directory for a specific experiment configuration.
        
        Args:
            config: Experiment configuration
            
        Returns:
            Path to the persistence directory
        """
        # Create a directory name based on relevant parameters
        splitter_name = config.splitter_type.name.lower()
        dir_name = (
            f"{self.corpus_dir.name}_"
            f"{config.embedding_model.split('/')[-1]}_"
            f"{splitter_name}_"
            f"{config.chunk_size}-{config.chunk_overlap}"
        )
        if config.augment_chunks:
            dir_name += f"_Aug-{config.augmenting_model}-{config.augmenting_temperature}-{config.augmenting_max_tokens}" 
        return self.base_persist_dir / dir_name
        

    def load_ground_truth(self) -> tuple[list[dict], list[str]]:
        # Load ground truth from Excel file next to corpus dir
        import sys
        excel_path = self.corpus_dir.parent / f"{self.corpus_dir.name} Ground Truth.xlsx"
        try:
            ground_truth = cbfk.ground_truth.GroundTruth(excel_path).get_all()
            logger.info(f"Loaded ground truth from {excel_path}")
        except Exception as e:
            logger.error(f"Failed to load ground truth from {excel_path}: {e}")
            sys.exit(1)
        queries = [truth["query"] for truth in ground_truth]    
        return ground_truth, queries
   

    def ingest_and_index(self, config: cbfk.experiment_config.ExperimentConfig) -> Tuple[Any, dict[str, Any], Any]:
        """
        Ingest documents and create an index with the given configuration.
        
        Args:
            config: Experiment configuration
            
        Returns:
            Tuple of (index, stats)
        """


        persist_dir = self.get_persist_dir(config)
        lockfile = persist_dir / ".persist.lock"

        # Ensure the persist directory exists before trying to lock it
        persist_dir.mkdir(parents=True, exist_ok=True)

        logging.info(f"Aquiring lock on {persist_dir} ...")
        with portalocker.Lock(lockfile, timeout=24*60*60):  # waits up to 24h if needed

            # Check if this index already exists
            if config.use_cached_index and (persist_dir / "docstore.json").exists():
                logger.info(f"Index already exists at {persist_dir}, skipping ingestion")
                index, _, retriever = cbfk.index_manager.load_index(
                    persist_dir=persist_dir,
                    config=config
                )
                return index, {'using_cached_index': True}, retriever

            model_max_sequence_length = cbfk.models.get_model_registry().get_max_sequence_length(config.embedding_model)
            if config.chunk_size > model_max_sequence_length:
                raise ChunkSizeTooLargeError(
                    f"Chunk size {config.chunk_size} is larger than {config.embedding_model} sequence length {model_max_sequence_length}"
                )

            logger.info(f"Ingesting documents with config: {config.experiment_id}")
            try:
                start_time = time.time()
                index, stats, retriever = cbfk.index_manager.ingest_documents(
                    corpus_dir=self.corpus_dir,
                    persist_dir=persist_dir,
                    config=config,
                    auto_adjust_chunk_size=False,
                )
                ingestion_time = time.time() - start_time
                logger.info(f"Ingestion completed in {ingestion_time:.2f} seconds")
                stats['using_cached_index'] = False

                # Log ingestion stats to wandb
                if self.wandb_logger:
                    self.wandb_logger.log_index_creation(
                        stats,
                        config.embedding_model,
                        config.to_dict(),
                        ingestion_time,
                        persist_dir)

            except Exception as e:
                logger.error(f"Error during ingestion: {e!s}")
                raise
            
        return index, stats, retriever


    def _compute_overall_run_stats(self, query_results: list[dict[str, float]], graded_accuracies: list[float]) -> dict[str, float]:
        """
        Compute overall run statistics, including averages for all metrics matching '*@number',
        such as 'vector_precision@5', 'bm25_recall@10', etc., as well as existing metrics.
        """
        ret: dict[str, float] = {}

        ret['graded_accuracy'] = round(sum(graded_accuracies) / len(graded_accuracies) if graded_accuracies else 0.0, 4)
        ret['source_score'] = round(sum(result.get('source_score', 0.0) for result in query_results) / len(query_results) if query_results else 0.0, 4)
        ret['precision'] = round(sum(result.get('precision', 0.0) for result in query_results) / len(query_results) if query_results else 0.0, 4)
        ret['recall'] = round(sum(result.get('recall', 0.0) for result in query_results) / len(query_results) if query_results else 0.0, 4)
        ret['mrr'] = round(sum(result.get('rr', 0.0) for result in query_results) / len(query_results) if query_results else 0.0, 4)

        # Generalize: Compute averages for all keys matching '*@number'
        at_k_metrics: dict[str, list[float]] = defaultdict(list)
        at_k_pattern = re.compile(r"^.+@\d+$")

        for result in query_results:
            for key, value in result.items():
                if at_k_pattern.match(key) and isinstance(value, (int, float)):
                    at_k_metrics[key].append(float(value))

        for key, vals in at_k_metrics.items():
            if vals:
                # Convert any _rr@k to _mrr@k, and rr@k to mrr@k
                if re.search(r"_rr@\d+$", key):
                    output_key = re.sub(r"_rr@(\d+)$", r"_mrr@\1", key)
                    ret[output_key] = sum(vals) / len(vals)
                elif re.match(r"^(.*\.)?rr@(\d+)$", key):
                    mrr_match = re.match(r"^(.*\.)?rr@(\d+)$", key)
                    prefix = mrr_match.group(1) or ""
                    k = mrr_match.group(2)
                    output_key = f"{prefix}mrr@{k}"
                    ret[output_key] = sum(vals) / len(vals)
                else:
                    ret[key] = sum(vals) / len(vals)

        return ret


    def run_this_configuration(self, config: cbfk.experiment_config.ExperimentConfig) -> dict:
        """
        Evaluate a specific parameter configuration against ground truth.
        
        Args:
            config: Experiment configuration
            
        Returns:
            Dictionary of evaluation results
        """

        import cbfk.evaluate
        import cbfk.ground_truth
        import cbfk.index_manager
        import cbfk.llm_prompter
        import cbfk.models

        start_time = time.time()
        graded_accuracies, query_results, index_responses = [], [], []
        index_stats = {}
        ground_truth, queries = self.load_ground_truth()

        try:

            _, index_stats, retriever = self.ingest_and_index(config)
            indexed_at_time = time.time()
            
            # Run batch queries on the index to get the relevant documents
            index_responses, retrieved_nodes_dicts = cbfk.index_manager.batch_query_index(
                retriever=retriever,
                queries=queries,
                config=config,
            )
            queried_at_time = time.time()
            
            # Create prompts for LLM
            prompts = cbfk.llm_prompter.batch_build_rag_prompts(
                rag_prompt=config.rag_prompt,
                ground_truth=ground_truth,
                index_responses=index_responses
            )
            prompted_at_time = time.time()
            
            # Query LLM with generated prompts
            if not config.skip_llm:
                llm_responses, llm_results = cbfk.llm_prompter.batch_prompt_ollama(
                    prompts=prompts,
                    model=config.llm_model,
                    temperature=config.llm_temperature,
                    max_tokens=config.llm_max_tokens
                )
            else:
                llm_responses = ["(skipped)"] * len(ground_truth)
            prompted_at_time = time.time()

            # Evaluate results against ground truth
            graded_accuracies, query_results = cbfk.evaluate.evaluate_against_ground_truth(
                ground_truth=ground_truth,
                index_responses=index_responses,
                retrieved_nodes_dicts=retrieved_nodes_dicts,
                llm_responses=llm_responses,
                llm_results=llm_results,
                config=config,
            )
            evaluated_at_time = time.time()

        except ChunkSizeTooLargeError as e:
            logger.warning(f"{LC.HI}{e}{LC.RESET}")
            graded_accuracies = [0] * len(ground_truth)
            query_results = []

        overall_run_stats = self._compute_overall_run_stats(query_results, graded_accuracies)
        prefixed_index_stats = {f"index.{k}": v for k, v in index_stats.items()}

        results = {
            "experiment_id": config.experiment_id,
            #**{f"parameters.{k}s": v for k, v in config.to_dict().items()},
            **overall_run_stats,
            **prefixed_index_stats,
            "query_results": query_results, 
            "prompts": prompts,
            "index_responses": index_responses,  # Include index_responses for HTML report
            "using_cached_index": index_stats.get("using_cached_index", False),
            "time.index_sec": round(indexed_at_time - start_time, 3),
            "time.query_sec": round(queried_at_time - indexed_at_time, 3),
            "time.prompt_sec": round(prompted_at_time - queried_at_time, 3),
            "time.evaluate_sec": round(evaluated_at_time - prompted_at_time, 3),
            "time.experiment_sec": round(evaluated_at_time - start_time, 3),
        }

        return results
 
