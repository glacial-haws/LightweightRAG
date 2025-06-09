import logging
from dataclasses import dataclass
from enum import Enum

from cbfk.log.log_config import LogConfig as LC

# Configure logging using LogConfig
LC.setup_logging(log_level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelType(Enum):
    EMB = "emb"
    LLM = "llm"

@dataclass(frozen=True, slots=True)
class ModelInfo:
    # find embedding model max sequence length with "uv run cbfk/models.py <short_name>""
    # find ollama llm model context length "ollama show <repo_id>", 
    # but ollama has its own max sequence length too: 
    # https://github.com/ollama/ollama/blob/main/docs/faq.md#how-can-i-specify-the-context-window-size
    type: ModelType
    short_name: str
    repo_id: str
    revision: str | None = None
    max_seq_len: int = 512
    query_instruction: str | None = None
    text_instruction: str | None = None


_global_models: list[ModelInfo] = [

    # embedding models
    ModelInfo(ModelType.EMB, 'bge-small', 'BAAI/bge-small-en-v1.5', 
                                                    revision='5c38ec7', 
                                                    max_seq_len=512),
    ModelInfo(ModelType.EMB, 'nomic-moe', 'nomic-ai/nomic-embed-text-v2-moe', 
                                                    revision='45301cc', 
                                                    max_seq_len=2048, 
                                                    query_instruction="search_query: ", 
                                                    text_instruction="search_document: "),
    ModelInfo(ModelType.EMB, 'mxbai-large', 'mixedbread-ai/mxbai-embed-large-v1',
                                                    revision='db9d1fe0f31addb4978201b2bf3e577f3f8900d2',
                                                    max_seq_len=512,
                                                    query_instruction='Represent this sentence for searching relevant passages: '),
    # ModelInfo(ModelType.EMB, 'stella-400m', 'NovaSearch/stella_en_400M_v5',
    #                                                 revision='dcae70d3f2b4aaee36afc3cde638ca4614497aec',
    #       stella requires cuda                      max_seq_len=8192,
    #                                                 query_instruction='Instruct: Given a web search query, retrieve relevant passages that answer the query.\nQuery: ',
    #                                                 text_instruction='Instruct: Retrieve semantically similar text.\nQuery: '),
    # ModelInfo(ModelType.EMB, 'jinaai-v3', 'jinaai/jina-embeddings-v3',
    #                                                 revision='f1944de8402dcd5f2b03f822a4bc22a7f2de2eb9',
    #        No module named 'custom_st'              max_seq_len=8192,
    #                                                 query_instruction=''),
    ModelInfo(ModelType.EMB, 'gte-base', 'Alibaba-NLP/gte-multilingual-base',
                                                    revision='9fdd4ee8bba0e2808a34e0e739576f6740d2b225',
                                                    max_seq_len=8192,
                                                    query_instruction=''),                                                    
    
    # LLM, ollama does not support revision pinning currently
    ModelInfo(ModelType.LLM, 'gemma3:1b', 'gemma3:1b', max_seq_len=32768),
    ModelInfo(ModelType.LLM, 'gemma3:4b', 'gemma3:4b', max_seq_len=131072),
    ModelInfo(ModelType.LLM, 'gemma3:12b', 'gemma3:12b', max_seq_len=131072),
    ModelInfo(ModelType.LLM, 'qwen3:0.6b', 'qwen3:0.6b', max_seq_len=40960),
    ModelInfo(ModelType.LLM, 'qwen3:1.7b', 'qwen3:1.7b', max_seq_len=40960),
    ModelInfo(ModelType.LLM, 'qwen3:4b', 'qwen3:4b', max_seq_len=40960),
    ModelInfo(ModelType.LLM, 'qwen3:8b', 'qwen3:8b', max_seq_len=40960),                                                        
    ModelInfo(ModelType.LLM, 'mistral:7b', 'mistral:7b', max_seq_len=32768),
    ModelInfo(ModelType.LLM, 'mistral-nemo:12b', 'mistral-nemo:12b', max_seq_len=1024000),
    ModelInfo(ModelType.LLM, 'deepseek-r1:1.5b', 'deepseek-r1:1.5b', max_seq_len=131072),
    ModelInfo(ModelType.LLM, 'deepseek-r1:7b', 'deepseek-r1:7b', max_seq_len=131072),
    ModelInfo(ModelType.LLM, 'deepseek-r1:8b', 'deepseek-r1:8b', max_seq_len=131072),
    ModelInfo(ModelType.LLM, 'llama3.2:1b', 'llama3.2:1b', max_seq_len=131072),
    ModelInfo(ModelType.LLM, 'llama3.2:3b', 'llama3.2:3b', max_seq_len=131072),
]   

class ModelRegistry:
    def __init__(self, model_list: list[ModelInfo]) -> None:
        self._models = model_list
        self._by_short_name = {m.short_name: m for m in model_list}
        self._by_repo_id = {m.repo_id: m for m in model_list}

    def list_short_names(self, model_type: ModelType | None = None) -> list[str]:
        """Return all model short names, optionally filtered by type."""
        if model_type is None:
            return [m.short_name for m in self._models]
        return [m.short_name for m in self._models if m.type == model_type]

    def is_repo_id(self, model: str) -> bool:
        """Return True if the given model is a repo_id, False if it is a short name."""
        return model in self._by_repo_id

    def get_repo_id(self, short_name: str) -> str:
        """Return the full model name for a given short name."""
        if short_name not in self._by_short_name:
            raise ValueError(f"Unknown model short name: {short_name}")
        return self._by_short_name[short_name].repo_id

    def get_revision(self, short_name: str) -> str | None:
        """Return the revision for a given short name."""
        if short_name not in self._by_short_name:
            raise ValueError(f"Unknown model short name: {short_name}")
        return self._by_short_name[short_name].revision

    def get_max_sequence_length(self, short_name: str) -> int:
        """Return the max sequence length for a given short name."""
        if short_name not in self._by_short_name:
            raise ValueError(f"Unknown model short name: {short_name}")
        return self._by_short_name[short_name].max_seq_len

    def get_model_info(self, short_name: str) -> ModelInfo:
        """Return the full model info for a given short name."""
        if short_name not in self._by_short_name:
            raise ValueError(f"Unknown model short name: {short_name}")
        return self._by_short_name[short_name]


def get_model_registry() -> ModelRegistry:
    """
    Returns the singleton instance of the ModelRegistry.
    """
    if not hasattr(get_model_registry, "_instance"):
        get_model_registry._instance = ModelRegistry(_global_models)
    return get_model_registry._instance


def get_embedding_model_max_length(short_name: str, show_info: bool = False) -> int:
    """
    Determine the maximum sequence length for a given embedding model.
    short_name: Short name of the embedding model
    Returns the maximum sequence length the model can handle
    """
    registry = get_model_registry()
    repo_id = registry.get_repo_id(short_name)
    revision: str | None = None
    if '@' in repo_id:
        repo_id, revision = repo_id.split('@', 1)
    # Also allow specifying revision in short_name directly
    if '@' in short_name:
        repo_id, revision = short_name.split('@', 1)
    if show_info:
        print(f"Resolved repo_id: {repo_id}")
        print(f"Revision: {revision if revision else 'default/latest'}")
    try:
        from transformers import AutoConfig, AutoTokenizer
        
        # Extract repo_id and revision if needed
        revision = None
        if '@' in short_name:
            repo_id, revision = short_name.split('@', 1)
        
        # First try to get config without loading the full model
        config_kwargs: dict[str, object] = {"trust_remote_code": True}
        if revision:
            config_kwargs["revision"] = revision
            
        config = AutoConfig.from_pretrained(repo_id, **config_kwargs)
        
        # Most models store max length in config.max_position_embeddings
        if hasattr(config, "max_position_embeddings"):
            return config.max_position_embeddings
            
        # If not in config, load the tokenizer which often has this information
        tokenizer_kwargs: dict[str, object] = {"trust_remote_code": True}
        if revision:
            tokenizer_kwargs["revision"] = revision
            
        tokenizer = AutoTokenizer.from_pretrained(repo_id, **tokenizer_kwargs)
        
        # Check common attributes where max length might be stored
        if hasattr(tokenizer, "model_max_length") and tokenizer.model_max_length != 10000000000000:
            return tokenizer.model_max_length
        
        # Default fallback if we can't determine
        logger.warning(f"{LC.AMBER}Could not determine max length for {short_name}, using default of 512{LC.RESET}")
        return 512
        
    except Exception as e:
        logger.error(f"Error determining model max length: {e!s}")
        # Return a conservative default
        return -1


if __name__ == "__main__":

    # use "ollama show <model_name>" to get context length for llms
    
    import sys
    if len(sys.argv) > 1:
        EMBEDDING_MODEL = sys.argv[1]
        max_length = get_embedding_model_max_length(EMBEDDING_MODEL, show_info=True)
        if max_length < 0:
            print(f"{LC.AMBER}Could not determine max length for {EMBEDDING_MODEL}{LC.RESET}")
            exit(1)
        print(f"Max length for {EMBEDDING_MODEL}: {max_length}")
    else:
        print("LLM models to pull:")
        models: list[str] = get_model_registry().list_short_names(ModelType.LLM)
        for m in models:
            print(f'ollama pull {m}')



# Example:

# from cbfk.models import get_model_registry, ModelType

# registry = get_model_registry()
# llm_names = registry.list_short_names(ModelType.LLM)
# print("Available LLMs:", llm_names)
