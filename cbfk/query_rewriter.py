
import json
import logging

from cbfk.experiment_config import ExperimentConfig
from cbfk.llm_prompter import prompt_rewrite_query

# Configure logging using LogConfig
from cbfk.log.log_config import LogConfig as LC
from cbfk.models import get_model_registry

LC.setup_logging(log_level=logging.INFO)
logger = logging.getLogger(__name__)


class QueryRewriter:

    query_cache = None
    query_cache_file = "index/experiments/query_rewriter_cache.json"


    def __persist_query_cache():
        with open(QueryRewriter.query_cache_file, "w") as f:
            json.dump(QueryRewriter.query_cache, f, indent=4)


    def __restore_query_cache():
        try:
            with open(QueryRewriter.query_cache_file, "r") as f:
                QueryRewriter.query_cache = json.load(f)
        except FileNotFoundError:
            QueryRewriter.query_cache = {}
            pass


    def __key(query: str, config: ExperimentConfig) -> str:
        return f"{config.query_rewrite_model}_{config.query_rewrite_temperature}_{config.query_rewrite_max_tokens}\n{query}"


    def rewrite(query: str, config: ExperimentConfig) -> str:
        
        if QueryRewriter.query_cache is None:
            QueryRewriter.__restore_query_cache()

        if QueryRewriter.__key(query, config) in QueryRewriter.query_cache:
            return QueryRewriter.query_cache[QueryRewriter.__key(query, config)]

        model = get_model_registry().get_repo_id(config.query_rewrite_model)
        rewritten_query = prompt_rewrite_query(
                                query, 
                                model = model,
                                temperature = config.query_rewrite_temperature,
                                max_tokens = config.query_rewrite_max_tokens)
        #logger.info(f"Query {LC.HI}{query}{LC.RESET} rewrite {LC.HI}{rewritten_query}{LC.RESET} with model {LC.HI}{config.query_rewrite_model}{LC.RESET}")
        QueryRewriter.query_cache[QueryRewriter.__key(query, config)] = rewritten_query
        QueryRewriter.__persist_query_cache()
        return rewritten_query

