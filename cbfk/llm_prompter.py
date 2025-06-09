#!/usr/bin/env python3
"""
Script to query a LlamaIndex index and get responses from a local LLM.
Usage: python query.py "Your query here"
"""

import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Any

import requests
from llama_index.core.base.response.schema import Response
from tqdm import tqdm

import cbfk.models

# Configure logging using LogConfig
from cbfk.log.log_config import LogConfig as LC

LC.setup_logging(log_level=logging.INFO)
logger = logging.getLogger(__name__)

_default_ollama_url: str = f"{os.environ.get('OLLAMA_URL', 'http://localhost:11434')}/v1/chat/completions"


class OllamaException(requests.exceptions.RequestException):
    pass


@dataclass
class OllamaPromptResult:
    text: str
    nothink_text: str
    prompt_length_char: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    status_code: int
    raw_response: dict[str, Any]
    retries: int
    response_time_sec: float
    ollama_url: str
    model: str
    temperature: float
    max_tokens: int
    prompt: str

def prompt_ollama(
    prompt: str,
    ollama_url: str = _default_ollama_url,
    model: str = "gemma3:1b",
    temperature: float = 0.7,
    max_tokens: int = 4096,
    num_ctx: int = 8192,    ) -> OllamaPromptResult:
    """
    Query the LLM using the /v1/chat/completions endpoint (OpenAI-compatible chat API).

    Args:
        prompt: The prompt text to send to the LLM
        ollama_url: The URL of the Ollama API endpoint
        model: The model name to use for generation
        temperature: The sampling temperature (higher = more creative, lower = more deterministic)
        max_tokens: Maximum number of tokens to generate for each response

    Returns:
        OllamaPromptResult: Dataclass containing generated text, token counts, status code, and raw response.
    """

    if not cbfk.models.get_model_registry().is_repo_id(model):
        model = cbfk.models.get_model_registry().get_repo_id(model)

    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "options": {
            "num_ctx": num_ctx
        }        
    }
    max_retries: int = 6
    backoff_base: int = 10  # seconds
    last_exception: Exception | None = None
    response = None
    attempt = 0
    response_time_sec = 0.0
    for attempt in range(max_retries):
        try:
            # Connect timeout (60 seconds), Read timeout (600 seconds)
            start_time = time.time()
            response = requests.post(ollama_url, json=payload, timeout=(60, 600))
            response_time_sec = time.time() - start_time
            break  # Success
        except requests.exceptions.ReadTimeout as e:
            last_exception = e
            logger.warning(f"Request to {model} failed ({attempt+1}/{max_retries}): {LC.AMBER}{e}{LC.RESET}")
            if attempt < max_retries - 1:
                backoff = backoff_base * (attempt + 1)
                logger.info(f"Retrying after {backoff} seconds...")
                time.sleep(backoff)
        except requests.exceptions.ConnectionError as e:
            last_exception = e
            logger.error(f"Connection error to {model}: {LC.RED}{e}{LC.RESET}")
            raise e
    if response is None:
        raise RuntimeError(
            f"Request to {model} failed after {max_retries} attempts: {last_exception.__class__} {last_exception}"
        )

    try:
        result = response.json()
        content = result["choices"][0]["message"]["content"]
        # Try to extract token usage info (OpenAI-compatible format)
        usage = result.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        total_tokens = usage.get("total_tokens")

        # Fallback: estimate tokens if not provided
        def estimate_tokens(text: str) -> int:
            # Simple whitespace-based fallback; replace with model-specific tokenizer if needed
            return len(text.split())

        if prompt_tokens is None:
            prompt_tokens = estimate_tokens(prompt)
        if completion_tokens is None:
            completion_tokens = estimate_tokens(content)
        if total_tokens is None:
            total_tokens = prompt_tokens + completion_tokens

        # Remove all <think>...</think> blocks and strip leading/trailing whitespace and quotes
        nothink_text = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip().strip('"').strip("'")

        return OllamaPromptResult(
            status_code=response.status_code,
            raw_response=result,
            text=content.strip(),
            nothink_text=nothink_text.strip(),
            prompt_length_char=len(prompt),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            retries=attempt,
            response_time_sec=response_time_sec,
            ollama_url=ollama_url,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            prompt=prompt,
        )
    except Exception as e:
        logger.error(f"Failed to parse {model} response: {LC.AMBER}{e.__class__}{LC.RESET}\n{LC.AMBER}{e}{LC.RESET}\nResponse text: {response.text}")
        raise RuntimeError(f"Failed to parse {model} response: {e.__class__} {e}\nResponse text: {response.text}")  # noqa: B904 will raise "from error"



def batch_prompt_ollama(prompts: str, 
                         ollama_url: str = _default_ollama_url, 
                         model: str = "gemma3:1b", 
                         temperature: float = 0.7, 
                         max_tokens: int = 512) -> list[str, OllamaPromptResult]:
    """Query the LLM with multiple prompts, one for each query, and get responses.
    
    This function sends multiple prompts to the Ollama API in sequence and collects
    all responses. Unlike the streaming version, this doesn't print responses in real-time.
    
    Args:
        prompts: List of prompt texts to send to the LLM
        model: The model name to use for generation
        temperature: The sampling temperature (higher = more creative, lower = more deterministic)
        max_tokens: Maximum number of tokens to generate for each response
        
    Returns:
        List of text responses corresponding to each prompt
    """

    responses: list[str] = []
    results: list[OllamaPromptResult] = []
    max_retries = 1
    
    for i, prompt_text in enumerate(tqdm(prompts, desc=f"Batch prompting {model}", unit="prompt")):
        retries = 0
        success = False
        last_status = None
        last_response = None
        while retries < max_retries:
            try:
                result: OllamaPromptResult = prompt_ollama(prompt_text, ollama_url, model, temperature, max_tokens)
                last_status = result.status_code
                last_response = result.text
                if last_status == 200:
                    responses.append(last_response)
                    results.append(result)
                    success = True
                    break
            except Exception as e:
                last_status = 500
                last_response = str(e)
                logger.warning(f"Error processing prompt {i+1}: {e!s} (retry {retries+1}/3)")
                # On last retry, do not raise; let the loop finish and append error to responses

            retries += 1
        if not success:
            logger.warning(f"Batch LLM prompting max retires, status {last_status} {last_response}")
            responses.append(f"Error: Error processing prompt {i+1}: Status {last_status}: {last_response} (retry {retries}/3)")
    return responses, results




_prompts = {
    "rag_no_repeat_question_md": {
        "prompt":
            "You are a helpful personal assistant. Answer the question **strictly based on the provided sources**.\n\n"
            "### Question\n"
            "{query_text}\n\n"
            "### Relevant Sources\n"
            "{formatted_sources}\n\n"
            "### Instructions\n"
            "- Only use the information from the sources above.\n"
            "- **Do not use any external knowledge.**\n"
            "- Do not repeat yourself.\n"
            "- Be concise, because the reader has access to the full text of all the documents and can read the whole thing if they want to.\n"
            "- Cite only the sources you directly refer to.\n"
            "- Use this format for citations: `[file_name § chapter]` or `[file_name § ]` if no chapter exists.\n"
            "- **Example citation** (do not copy this unless relevant): `[Master Service Agreement.md § 17.1]`\n"
            "- If the answer is not found in the provided sources, reply with: `No answer found in provided sources.`\n\n"
            "### Your Answer"
    },        
    "grading_accuracy_md": {
        "prompt":
            "You are an expert evaluator. Your task is to rate the quality of a candidate answer to a user query, using a reference answer (ground truth) for comparison.\n\n"
            "Evaluate the candidate answer based on the following criteria:\n\n"
            "- **Factual Accuracy**: Are the facts correct and consistent with the reference answer?\n"
            "- **Completeness**: Does the candidate include the key information found in the reference answer?\n"
            "- **Relevance**: Does the candidate directly address the question, without going off-topic?\n\n"
            "Choose one of the following five ratings:\n\n"
            "- **Wrong**: The answer is mostly or completely incorrect, irrelevant, or misleading.\n"
            "- **Poor**: The answer has some relevance but contains factual inaccuracies or omits key information.\n"
            "- **Fair**: The answer is mostly correct but may lack completeness or precision. Or the answer says 'No answer found in provided sources.'\n"
            "- **Good**: The answer is accurate and relevant with only minor omissions or imprecisions.\n"
            "- **Perfect**: The answer is factually accurate, complete, and fully relevant to the question.\n\n"
            "---\n\n"
            "### User Query\n\n"
            "{question}\n\n"
            "### Reference Answer\n\n"
            "{truth}\n\n"
            "### Candidate Answer\n\n"
            "{llm_response}\n\n"
            "---\n\n"
            "Please respond with:\n\n"
            "- The selected **rating** (one of: Wrong, Poor, Fair, Good, Perfect)\n"
            "- A brief **justification** (1-2 sentences)\n\n"
            "Your response: \n",
    },
    "rewrite_query_simple": {
        "prompt":
            "You are rewriting user search queries to make them clearer and more complete for a document search system.\n\n"
            "You may paraphrase, add synonyms, or explain terms.\n"
            "You may not ask for additional information.\n"
            "Only return the rewritten query, do not add any additional text.\n"
            "Do not return the original query.\n"
            "Rewrite the query below to make it clearer and more complete.\n\n"
            "### Original Query \n"
            "{user_query}\n\n"
            "### Rewritten Query\n",
    },
    "augmenting_question_simple": {
        "prompt":
            "You are generating augmenting questions for a IT sercvice management contract search system.  "
            "Given the following document text, generate questions that could be answered by this text. "
            "Be concise. Cover all the main ideas, key facts, and important details. "
            "Ask for different question types (who, what, how, why, etc.) if relevance allows."
            "Focus on semantic relevance, not just repeating keywords from the chunk."
            "Only output the questions as a numbered list. "
            "Do not include any additional text, never start with \"Here are the questions:\" or similar.\n\n"
            "### Document Text:\n\n"
            "---\n\n"
            "{chunk_text}\n\n"
            "---\n\n"
            "### Questions:\n1.",
    },
}


def build_rag_prompt(rag_prompt: str, query_text: str, rag_response: Response) -> str:
    """Build a prompt for the RAG system based on the given query and response.
    Uses metadata.original_source that does not contain autmenting questions if it is there. 
    """
    sources = rag_response.source_nodes
    formatted_sources = ''.join(
        f"[{source.node.metadata.get('file_name', 'UnknownFile')} § {source.node.metadata.get('chapter', '')}]: "
        f"{source.node.metadata.get('original_text', source.node.get_content())}\n\n"
        for source in sources
    )
     
    prompt = _prompts[rag_prompt]['prompt'].format(
        query_text=query_text,
        formatted_sources=formatted_sources
    )
    return prompt


def batch_build_rag_prompts(rag_prompt: str, ground_truth: list[dict[str, str]], index_responses: list[Response]) -> list[str]:    
    prompts = []
    for i, truth in enumerate(ground_truth): 
        rag_response = index_responses[i]
        prompt = build_rag_prompt(rag_prompt, truth['query'], rag_response)
        prompts.append(prompt)
    return prompts


def prompt_grading_accuracy(
    question: str, 
    truth: str, 
    llm_response: str, 
    model: str, 
    temperature: float, 
    max_tokens: int)            -> OllamaPromptResult:
    """Grade the LLM response using the LLM."""
    prompt = _prompts['grading_accuracy_md']['prompt'].format(
                                                            question=question,
                                                            truth=truth,
                                                            llm_response=llm_response)
    result = prompt_ollama(
                        prompt=prompt,
                        model=model, 
                        temperature=temperature, 
                        max_tokens=max_tokens)    
    
    if result.status_code != 200:
        raise OllamaException(f"Error processing eval prompt: Status {result.status_code}: {result.text}")
    
    return result
    

def prompt_rewrite_query(user_query: str, model: str = "gemma3:1b", temperature: float = 0.7, max_tokens: int = 4096) -> str:
    """Rewrite the query using the LLM."""
    prompt = _prompts['rewrite_query_simple']['prompt'].format(user_query=user_query)
    result = prompt_ollama(
                        prompt=prompt,
                        model=model, 
                        temperature=temperature, 
                        max_tokens=max_tokens)    
    
    if result.status_code != 200:
        raise OllamaException(f"Error processing prompt for rewrite: Status {result.status_code}: {result.text}")
            
    return result.nothink_text
    

def prompt_augmenting_question(chunk_text: str, model: str = "gemma3:1b", temperature: float = 0.7, max_tokens: int = 4096) -> str:
    """Augment the question using the LLM."""

    prompt = _prompts['augmenting_question_simple']['prompt'].format(chunk_text=chunk_text)
    result = prompt_ollama(
                        prompt=prompt,
                        model=model, 
                        temperature=temperature, 
                        max_tokens=max_tokens)    
    
    if result.status_code != 200:
        raise OllamaException(f"Error processing chunk for question: Status {result.status_code}: {result.text}")
    
    return result.nothink_text
