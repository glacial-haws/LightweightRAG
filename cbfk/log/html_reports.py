# cbfk/logging/visualization.py
import datetime
import html
import re
from pathlib import Path
from typing import Any, Dict, List, TypedDict

from jinja2 import Environment, FileSystemLoader, select_autoescape
from llama_index.core.base.response.schema import Response


class SourceNodeInfo(TypedDict):
    """Type definition for source node information."""
    id: int
    source: str
    score: float
    text: str


class QueryResult(TypedDict, total=False):
    """Type definition for query result information."""
    query: str
    response: str
    expected_answer: str
    recall: float
    graded_accuracy: float
    llm_response: str
    sources: List[Dict[str, Any]]
    source_nodes: List[Dict[str, Any]]
    response_time: float
    experiment_name: str
    experiment_params: Dict[str, Any]


class ProcessedQueryResult(TypedDict, total=False):
    """Type definition for processed query result information."""
    query: str
    response_length: int
    sources_count: int
    response_time: float
    experiment_name: str
    experiment_params: Dict[str, Any]


def _get_jinja_env() -> Environment:
    """
    Get a Jinja2 environment configured with the templates directory.
    
    Returns:
        Environment: A configured Jinja2 environment
    """
    templates_dir = Path(__file__).parent / "templates"
    env = Environment(
        loader=FileSystemLoader(templates_dir),
        autoescape=select_autoescape(['html', 'xml']),
        trim_blocks=True,
        lstrip_blocks=True
    )
    
    # Add custom filters
    env.filters['format_number'] = lambda n: f"{n:,}"
    
    # Add global functions
    env.globals['enumerate'] = enumerate
    
    return env


def _get_generation_time() -> str:
    """
    Get the current time formatted for HTML reports.
    
    Returns:
        str: Formatted current time
    """
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _render_template(template_name: str, **kwargs: Any) -> str:
    """
    Render a Jinja2 template with the given context.
    
    Args:
        template_name: Name of the template file
        **kwargs: Context variables for the template
        
    Returns:
        str: Rendered HTML content
    """
    env = _get_jinja_env()
    template = env.get_template(template_name)
    
    # Add generation time if not provided
    if 'generation_time' not in kwargs:
        kwargs['generation_time'] = _get_generation_time()
    
    return template.render(**kwargs)


def _highlight_text(text: str, highlights: list[str]) -> str:
    """
    Highlight occurrences of specific text within a larger text.
    Case-insensitive matching while preserving the original case in the output.
    
    Args:
        text: The text to highlight within
        highlights: List of strings to highlight
        
    Returns:
        str: Text with highlighted portions
    """
    highlighted_text = text
    for highlight in highlights:
        if not highlight:  # Skip empty highlights
            continue
            
        # Case-insensitive search using regex
        pattern = re.compile(re.escape(highlight), re.IGNORECASE)
        matches = pattern.finditer(highlighted_text)
        
        # Process matches in reverse to avoid offset issues when replacing
        matches = list(matches)
        for match in reversed(matches):
            start, end = match.span()
            original_match = highlighted_text[start:end]  # Preserve original case
            replacement = f'<span class="highlight">{html.escape(original_match)}</span>'
            highlighted_text = highlighted_text[:start] + replacement + highlighted_text[end:]
            
    return highlighted_text


def create_index_report_html(stats: Dict[str, Any], model_name: str, params: Dict[str, Any], ingestion_time: float, persist_dir: Path) -> str:
    """
    Create a standardized HTML report for index statistics.
    
    Args:
        stats: Dictionary containing index statistics
        model_name: Name of the embedding model used
        params: Dictionary of parameters used for index creation
        ingestion_time: Time taken to create the index
        persist_dir: Directory where the index is persisted
    Returns:
        str: HTML string representation of the index report
    """
    # Sort sources by count (descending)
    sorted_sources = []
    if "source_distribution" in stats:
        sorted_sources = sorted(
            stats["source_distribution"].items(),
            key=lambda x: x[1],
            reverse=True
        )
    
    return _render_template(
        'index_report.jinja2',
        stats=stats,
        model_name=model_name,
        stored_params=params,
        sorted_sources=sorted_sources,
        ingestion_time=ingestion_time,
        persist_dir=persist_dir
    )


def create_experiment_run_report(experiment_id: str, recall: float, mrr: float, graded_accuracy: float, source_score: float, query_results: List[QueryResult], index_responses: List[Response]) -> str:
    """
    Create a summary HTML table of all queries and their scores.
    
    Args:
        experiment_id: ID of the experiment
        recall: Recall metric
        mrr: Reciprocal Rank metric
        graded_accuracy: LLM graded accuracy
        query_results: List of query result dictionaries    
        index_responses: List of index response objects
        
    Returns:
        str: HTML string representation of the summary table
    """
    # Add query rewrite to query results if available
    has_query_rewrite = False
    if len(index_responses) > 0 and hasattr(index_responses[0], 'query_rewrite') and index_responses[0].query_rewrite is not None:
        has_query_rewrite = True
        query_results = [query_result | {"query_rewrite": index_response.query_rewrite} 
                        for query_result, index_response in zip(query_results, index_responses, strict=False)]

    return _render_template('experiment_run_report.jinja2', 
                            experiment_id=experiment_id, 
                            recall = recall,
                            mrr = mrr,
                            graded_accuracy = graded_accuracy,
                            source_score = source_score,
                            query_results=query_results,
                            has_query_rewrite=has_query_rewrite)


def create_html_question_report(experiment_id: str, query_result: QueryResult, index_response: Response, prompt: str) -> str:
    """
    Create an HTML visualization of an index response with highlighted ground truth quotes.
    
    Args:
        query_result: Dictionary containing query result information
        index_response: The index response object
        prompt: The prompt used for the query

    Returns:
        str: HTML string representation of the response
    """
    # Extract ground truth sources for highlighting
    truth_quotes = query_result["truth_quotes"]

    # Highlight ground truth in source nodes
    highlighted_source_nodes = []
    for node in index_response.source_nodes:
        n = {
            'score': node.score,
            'text': _highlight_text(node.get_content(), truth_quotes),
            'source': node.metadata.get('file_name', 'unknown'),
            'chapter': node.metadata.get('chapter', 'unknown'),
            'retriever_source': node.metadata.get('retriever_source', 'unknown'),
            'retriever_score': node.metadata.get('retriever_score', '0.0'),
            'origin': node.metadata.get('query_origin', 'unknown')
        }
        highlighted_source_nodes.append(n)

    # Add query rewrite to query results if available
    has_query_rewrite = False
    query_rewrite = ""
    if hasattr(index_response, 'query_rewrite') and index_response.query_rewrite is not None:
        has_query_rewrite = True
        query_rewrite = index_response.query_rewrite

    # Only include eval_response if it contains more than one word
    eval_response = query_result.get("eval_response", "")
    llm_response_raw: str = query_result.get("llm_response", "")
    llm_response_html: str = _format_llm_response_for_html(llm_response_raw)
    template_kwargs = dict(
        experiment_id=experiment_id,
        query=query_result["query"],
        expected_answer=query_result["expected_answer"],
        recall=query_result["recall"],
        source_score=query_result.get("source_score", 0),
        rr=query_result.get("rr", query_result.get("mrr", 0.0)),  # Use mrr if rr is not present
        graded_accuracy=query_result.get("graded_accuracy", 0),
        llm_response=llm_response_html,
        highlighted_source_nodes=highlighted_source_nodes,
        prompt=prompt,
        has_query_rewrite=has_query_rewrite,
        query_rewrite=query_rewrite
    )
    if len(eval_response.split()) > 1:
        # sometimes, the evaluating llm provides an explanation to the score (behind the first word)
        template_kwargs["eval_response"] = eval_response
    return _render_template(
        'question_report.jinja2',
        **template_kwargs
    )


def _format_llm_response_for_html(llm_response: str) -> str:
    """
    Format LLM response for HTML, wrapping <think>...</think> sections in a special div.
    Escapes all text outside <think>...</think> and inside, preserving line breaks.
    """
    import html
    import re
    if not llm_response:
        return ""
    # Pattern to find <think>...</think> (non-greedy)
    pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
    parts: list[str] = []
    last_end = 0
    for match in pattern.finditer(llm_response):
        # Escape and append text before <think>
        before = llm_response[last_end:match.start()]
        if before:
            parts.append(html.escape(before).replace("\n", "<br>"))
        # Escape and wrap <think> content
        think_content = match.group(1)
        think_html = f'<div class="think-section">{html.escape(think_content).replace("\n", "<br>")}</div>'
        parts.append(think_html)
        last_end = match.end()
    # Escape and append any text after last </think>
    after = llm_response[last_end:]
    if after:
        parts.append(html.escape(after).replace("\n", "<br>"))
    return ''.join(parts)

