#!/usr/bin/env python3
"""
Evaluation utilities for the chatbot system.
"""

import logging
import re
import string
import sys
import time
from typing import Any, Tuple

from llama_index.core.base.response.schema import Response
from llama_index.core.schema import NodeWithScore
from tqdm import tqdm

import cbfk.experiment_config
import cbfk.llm_prompter

# Configure logging using LogConfig
from cbfk.log.log_config import LogConfig as LC

LC.setup_logging(log_level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_source_score(llm_response: str, ground_truth_sources: list[dict]) -> float:
    cited = re.findall(r"\[(.*?) § (.*?)\]", llm_response)
    cited_set = set((file_name.strip(), chapter.strip()) for file_name, chapter in cited)
    truth_set = set((s["file_name"], s["chapter"]) for s in ground_truth_sources)

    if not cited_set: return 0.0  # No citation

    truth_files = [file_name for file_name, _ in truth_set]
    cited_files = [file_name for file_name, _ in cited_set]

    truth_unique = set(truth_files)
    cited_set = set(cited_files)

    matched = len(truth_unique & cited_set)
    total = len(truth_unique)
    if matched == 0: return 0.1 # all citations wrong gets 0.1 for effort
    return round(matched / total, 4)

def parse_grading_accuracy_response(response: str, recall: float = 1.0) -> tuple[float, str]:
    """
    Extract a score (float between 0 and 1) from the response string based on a rating label (Wrong, Poor, Fair, Good, Perfect),
    followed by a justification. Removes all <think>...</think> blocks from the response for both parsing and return.
    Returns (score, justification).
    """
    rating_map = {
        'wrong': 0.0,
        'poor': 0.25,
        'fair': 0.5,
        'good': 0.75,
        'perfect': 1.0,
    }

    response = re.sub(r'\n', ' ', response).strip()

    # Special handling for fallback cases
    if 'no answer found in provided documents' in response.lower():
        if recall == 0:
            return .5, response
        else:
            return 0.0, response

    # Check for perfect match indicators
    if re.search(r'matches\s+(?:the\s+)?(?:ground\s+)?truth\s+perfectly', response, re.IGNORECASE):
        return 1.0, response

    # Extract the rating and justification
    match = re.search(r'\b(Wrong|Poor|Fair|Good|Perfect)\b\s*[:\-–]?\s*(.*)', response, re.IGNORECASE)  # noqa: RUF001 ambiguous dash
    if match:
        rating = match.group(1).lower()
        justification = match.group(2).strip()
        score = rating_map.get(rating, -1.0)
        return score, justification

    # If rating not found, return -1.0 and the response
    return 0.0, response


def evaluate_llm_response(model: str, question: str, truth: str, 
                            llm_response: str, recall: float, 
                            temperature: float, 
                            max_tokens: int) -> Tuple[float, str, float]:  
        
    start_time = time.time()
    result = cbfk.llm_prompter.prompt_grading_accuracy(
        question=question, 
        truth=truth, 
        llm_response=llm_response, 
        model=model, 
        temperature=temperature, 
        max_tokens=max_tokens)
    response = result.nothink_text
    score, response = parse_grading_accuracy_response(response, recall)
    elapsed_time_sec = time.time() - start_time
    return score, response, elapsed_time_sec
    

def calculate_rr(truth_quotes: list[str], response_quotes: list[str]) -> tuple[float, int]:
    """Reciprocal Rank calculation, measuring how high up the first relevant doc appears in the results.
    Returns the reciprocal rank of the highest-ranked truth quote found and the rank of the best match.
    """
    best_rank:int = sys.maxsize
    if not truth_quotes or not response_quotes:
        return 0.0, best_rank
    
    for truth_quote in truth_quotes:
        for i, response_quote in enumerate(response_quotes):
            # Check if the truth quote is contained within the response quote (case-insensitive)
            if truth_quote.lower() in response_quote.lower():
                rank = i + 1
                best_rank = min(best_rank, rank)
                break
    
    # Return the reciprocal of the best rank found, or 0.0 if no matches
    rr: float = round(1.0 / best_rank if best_rank != 0 else 0.0, 4)
    return rr, best_rank



def _evaluate_index_response(truth_quotes: list[str], 
                            response_quotes: list[str], 
                            at_k: int )                 -> tuple[float, float, float, int]:
    """
    Evaluate the response from the index against ground truth quotes.
    
    Args:
        truth_quotes: List of ground truth quotes
        response_quotes: List of response quotes
    Returns:
        Tuple of (precision, recall, rr, best_rank)
    """

    # calculate Reciprocal Rank
    rq_at_k = response_quotes[:at_k].copy()
    rr, best_rank = calculate_rr(truth_quotes, rq_at_k)

    def strip_trailing_punct(s: str) -> str:
        return s.rstrip(string.whitespace + string.punctuation)
        
    found_quotes = [
        strip_trailing_punct(truth_quote)
        for truth_quote in truth_quotes
        if any(
            strip_trailing_punct(truth_quote) in strip_trailing_punct(rq)
            for rq in rq_at_k
        )
    ]
    found_quotes_count = len(found_quotes)
    
    # calculate index score, precision, recall
    precision = round(found_quotes_count / len(rq_at_k), 4) if rq_at_k else 0.0
    recall = round(found_quotes_count / len(truth_quotes), 4) if truth_quotes else 0.0

    return precision, recall, rr, best_rank


def evaluate_index_responses(truth: dict[str, str|list[str]], 
                            index_response: Response, 
                            vector_retriever_source: list[NodeWithScore] | None = None,
                            bm25_retriever_source: list[NodeWithScore] | None = None,
                            combined_retriever_source: list[NodeWithScore] | None = None, 
                            at_ks: list[int] | None = None) -> dict[str, Any]:  
    """
    Evaluate the response from the index against ground truth quotes.
    
    Args:
        truth: Dictionary containing ground truth quotes
        index_response: Object containing index response
        vector_retriever_source: Object containing vector retriever source
        bm25_retriever_source: Object containing bm25 retriever source
        combined_retriever_source: Object containing combined retriever source - this should be the same as the index_response
        at_ks: List of k values to evaluate at for precision, recall, mrr
    Returns:
        Dictionary containing evaluation results
    """
    ret: dict[str, Any] = dict()
    if at_ks is None: at_ks = [5]

    # Extract quotes from ground truth and index response
    truth_quotes = [quote['quote'] for quote in truth['sources']]
    response_quotes = [node.get_content() for node in index_response.source_nodes]
    precision, recall, rr, best_rank = _evaluate_index_response(truth_quotes, response_quotes, len(response_quotes))

    # check for watchword (for debugging)
    # watchword = "AI4ICU"
    # truth_contains_watchword = any(watchword in quote for quote in truth_quotes)
    # response_contains_watchword = any(watchword in quote for quote in response_quotes)
    # if truth_contains_watchword or response_contains_watchword:
    #     print(f"Breakpoint for {watchword}") # set breakpoint here
    
    # record which mechanism yielded the best result
    if best_rank != float('inf') \
        and len(index_response.source_nodes) >= best_rank \
        and index_response.source_nodes[best_rank - 1].metadata is not None:
            successful_retriever_source = index_response.source_nodes[best_rank - 1].metadata.get('retriever_source', 'unknown')
            successful_query_origin = index_response.source_nodes[best_rank - 1].metadata.get('query_origin', 'unknown')
    else:
        successful_retriever_source = 'none'
        successful_query_origin = 'none'  

    ret['truth_quotes'] = truth_quotes
    ret['successful_retriever_source'] = successful_retriever_source
    ret['successful_query_origin'] = successful_query_origin

    ret['precision'] = precision
    ret['recall'] = recall
    ret['rr'] = rr


    for at_k in at_ks:
        precision, recall, rr, _ = _evaluate_index_response(truth_quotes, response_quotes, at_k)
        ret[f'precision@{at_k}'] = precision
        ret[f'recall@{at_k}'] = recall
        ret[f'rr@{at_k}'] = rr

        if vector_retriever_source is not None:
            vector_response_quotes = [node.get_content() for node in vector_retriever_source]
            vector_precision, vector_recall, vector_rr, _ = _evaluate_index_response(truth_quotes, vector_response_quotes, at_k)
            ret[f'vector_precision@{at_k}'] = vector_precision
            ret[f'vector_recall@{at_k}'] = vector_recall
            ret[f'vector_rr@{at_k}'] = vector_rr

        if bm25_retriever_source is not None:
            bm25_response_quotes = [node.get_content() for node in bm25_retriever_source]
            bm25_precision, bm25_recall, bm25_rr, _ = _evaluate_index_response(truth_quotes, bm25_response_quotes, at_k)
            ret[f'bm25_precision@{at_k}'] = bm25_precision
            ret[f'bm25_recall@{at_k}'] = bm25_recall
            ret[f'bm25_rr@{at_k}'] = bm25_rr

        if combined_retriever_source is not None:
            combined_response_quotes = [node.get_content() for node in combined_retriever_source]
            combined_precision, combined_recall, combined_rr, _ = _evaluate_index_response(truth_quotes, combined_response_quotes, at_k)
            ret[f'combined_precision@{at_k}'] = combined_precision
            ret[f'combined_recall@{at_k}'] = combined_recall
            ret[f'combined_rr@{at_k}'] = combined_rr

    return ret

def evaluate_against_ground_truth(
    ground_truth: list[dict[str, object]],
    index_responses: list[NodeWithScore],
    retrieved_nodes_dicts: list[dict[str, list[NodeWithScore]]],
    llm_responses: list[str],
    llm_results: list[object],
    config: cbfk.experiment_config.ExperimentConfig) -> Tuple[list[float], list[dict[str, Any]]]:
    """
    Evaluate index and LLM responses against ground truth.
    
    Args:
        ground_truth: List of ground truth queries with sources and answers
        index_responses: List of index responses
        retrieved_nodes: Dict of retrieved nodes for each retriever
        llm_responses: List of LLM responses
        config: ExperimentConfig
        
    Returns:
        Tuple of (graded_accuracies, per_query_results)
        per_query_results includes precision and recall metrics
    """
    graded_accuracies = []
    query_results = []

    # iterate over ground truth and responses
    display_text = f" with {config.evaluating_model}" if config.evaluate_graded_accuracy else " without scoring llm"
    for i, truth in enumerate(tqdm(ground_truth, desc=f"Evaluating {display_text}", unit="query")):
        
        # Evaluate Retrieval
        index_response = index_responses[i]       
        vector_retriever_source = retrieved_nodes_dicts[i]['vector'] if 'vector' in retrieved_nodes_dicts[i] else None
        bm25_retriever_source = retrieved_nodes_dicts[i]['bm25'] if 'bm25' in retrieved_nodes_dicts[i] else None
        combined_retriever_source = retrieved_nodes_dicts[i]['combined'] if 'combined' in retrieved_nodes_dicts[i] else None

        index_response_metrics: dict[str, Any] = evaluate_index_responses(
                                                    truth, 
                                                    index_response,
                                                    vector_retriever_source,
                                                    bm25_retriever_source,
                                                    combined_retriever_source,
                                                    at_ks = [5,8,13])

        # Evaluate Genration
        llm_response = llm_responses[i]
        if not config.evaluate_graded_accuracy:
            # In test mode, use direct string comparison for deterministic results
            graded_accuracy = 1.0 if truth['answer'] in llm_response else 0.0
            eval_response = f"Test mode evaluation: {'Correct' if graded_accuracy == 1.0 else 'Incorrect'}"
        else:
            # In normal mode, use LLM evaluation
            graded_accuracy, eval_response, _ = evaluate_llm_response(
                                                    question=truth['query'], 
                                                    truth=truth['answer'], 
                                                    llm_response=llm_response, 
                                                    recall=index_response_metrics['recall'], 
                                                    model=config.evaluating_model, 
                                                    temperature=config.evaluating_temperature, 
                                                    max_tokens=config.evaluating_max_tokens)
        
        graded_accuracies.append(graded_accuracy)

        source_score = compute_source_score(llm_response, truth['sources'])

        # record per-query results
        query_result = {
            "query": truth["query"],
            "expected_answer": truth["answer"],
            "llm_response": llm_response,
            "llm_prompt_tokens": llm_results[i].prompt_tokens,
            "llm_completion_tokens": llm_results[i].completion_tokens,
            "graded_accuracy": graded_accuracy,
            "eval_response": eval_response,
            "source_score": source_score,
        }
        query_result.update(index_response_metrics)
        query_results.append(query_result)
    
    return graded_accuracies, query_results