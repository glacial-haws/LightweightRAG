#!/usr/bin/env python3
"""
Weights & Biases sweep configuration for parameter space exploration.
"""

import argparse
import sys
from pathlib import Path

import commentjson as json
import wandb
import yaml

from cbfk.log.log_config import LogConfig as LC


def get_sweep_config() -> dict:
    """
    Define the sweep configuration for hyperparameter optimization.
    
    Returns:
        Dictionary containing the sweep configuration
    """
    sweep_config = {
        'method': 'grid',
        'parameters': {

            # Indexing parameters          
            'embedding_model': {
                'values': [
                    # 'bge-small',  
                    'nomic-moe',
                    # 'mxbai-large',
                    # 'gte-base',
                ]
            },
            'splitter_type': {
                'values': [
                    #'TOKEN',
                    'SENTENCE',
                    # 'SEMANTIC',
                ]
            },
            'chunk_size': {
                'values': [1024] #[256, 384, 512, 768, 1024, 1280]  
            },
            'chunk_overlap_pct': {
                'values': [0.1] 
            },
            'use_cached_index': {  
                'values': [True]
            },

            # Augmenting parameters
            'augment_chunks': {
                'values': [True]    
            },
            'augmenting_model': {
                'values': [
                    'gemma3:4b', 
                ]
            },
            'augmenting_temperature': {
                'values': [0.7]
            },
            'augmenting_max_tokens': {
                'values': [1024] 
            },

            # Retrieval parameters
            'similarity_top_k': {   
                'values': [8] 
            },
            'vector_top_k': {   
                'values': [13] 
            }, 
            'bm25_top_k': {   
                'values': [8] 
            },
            'rag_prompt': {
                'values': [
                    'rag_no_repeat_question_md'
                ]
            },

            # Hybrid retriever parameters
            'deduplicate': {
                'values': [True]
            },
            'rerank': {
                'values': [True]
            },
            'crossencoder_model': {
                'values': [
                    'cross-encoder/ms-marco-MiniLM-L6-v2',
                ]
            },

            # Query rewrite parameters
            'query_rewrite': {
                'values': [True]
            },
            'query_rewrite_model': {
                'values': [
                    'gemma3:4b',
                ]
            },
            'query_rewrite_temperature': {
                'values': [0.7]
            },
            'query_rewrite_max_tokens': {
                'values': [950] 
            },

            # Chatbot LLM parameters
            'llm_model': {
                'values': [
                    # 'llama3.2:1b',
                    # 'llama3.2:3b',
                    # 'gemma3:1b', 
                    # 'gemma3:4b', 
                    # 'gemma3:12b',
                    # 'qwen3:0.6b',
                    'qwen3:1.7b',
                    'qwen3:4b', 
                    'qwen3:8b', 
                    # 'mistral:7b',
                    # 'mistral-nemo:12b',
                    # 'deepseek-r1:1.5b',
                    # 'deepseek-r1:7b',
                    'deepseek-r1:8b',
                ]
            },
            'llm_temperature': {
                'values': [0.1]
            },
            'llm_max_tokens': {
                'values': [10000]
            },            

            # Evaluation
            'evaluate_graded_accuracy': {
                'values': [True]
            },
            'evaluating_model': {
                'values': [
                    'gemma3:4b', 
                ]
            },
            'evaluating_temperature': {
                'values': [0.1]
            },
            'evaluating_max_tokens': {
                'values': [4096] 
            },

            # Corpus
            'corpus_path': {
                'values': [
                    # 'corpus/Contract_Subset'
                    'corpus/Contract'
                ]
            },

            # Loop values for debugging to keep the sweep going
            # 'loop_values': {
            #     'values': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 
            #     26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
            #     51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75,
            #     76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100,
            #     ]
            # },   
        }
    }
    
    with open('temp/sweep_config.json', 'w', encoding='utf-8') as f:
        json.dump(sweep_config, f, ensure_ascii=False, indent=4)
    with open('temp/sweep_config.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(sweep_config, f, indent=4, default_flow_style=False, width=float('inf'))


    return sweep_config


def get_total_runs_in_sweep() -> int:
    sweep_config = get_sweep_config()
    param_values = [v['values'] for v in sweep_config['parameters'].values()]
    from math import prod
    return prod([len(v) for v in param_values])


def create_sweep(project: str, entity: str | None = None, sweep_config: dict | None = None) -> str:
    """
    Create a new sweep in Weights & Biases.
    
    Args:
        project: Name of the wandb project
        entity: Name of the wandb entity/team (optional)
        
    Returns:
        Sweep ID
    """
    if not sweep_config:
        sweep_config = get_sweep_config()
    sweep_id = wandb.sweep(sweep_config, project=project, entity=entity)
    return sweep_id


def update_launch_json(sweep_id: str) -> None:
    launch_path = Path(__file__).parent.parent / ".vscode" / "launch.json"
    try:
        # Update the sweep id in the launch.json file
        with launch_path.open("r", encoding="utf-8") as f:
            launch_data = json.load(f)
        changed = False
        for config in launch_data.get("configurations", []):
            config_args = config.get("args")
            if config_args and "--sweep-id" in config_args:
                idx = config_args.index("--sweep-id")
                if idx + 1 < len(config_args) and config_args[idx+1] != sweep_id:
                    config_args[idx+1] = sweep_id
                    changed = True
        if changed:
            with launch_path.open("w", encoding="utf-8") as f:
                json.dump(launch_data, f, indent=4)
            print(f"Updated {LC.HI}{launch_path}{LC.RESET} with new sweep id: {LC.HI}{sweep_id}{LC.RESET}")
    except Exception as e:
        print(f"Could not update {launch_path}: {e}")


def parse_args() -> argparse.ArgumentParser:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run a wandb sweep agent")
    parser.add_argument(
        "--wandb-project",
        type=str,
        default='cbfk-debug',
        help="Weights & Biases project name"
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="Weights & Biases team/entity name"
    )
    parser.add_argument(
        "--config-file",
        type=str,
        default=None,
        help="Path to the sweep config file (json or yaml)"
    )
    return parser


def main():
    # Create a sweep when run direct
    parser = parse_args()
    args = parser.parse_args() 
    sweep_config: dict | None = None
    print(sys.argv)
        
    # Create the sweep
    sweep_id: str = create_sweep(project=args.wandb_project, entity=args.wandb_entity, sweep_config=sweep_config)

    # Update .vscode/launch.json with new sweep id for debugging project
    if args.wandb_project == "cbfk-debug":
        update_launch_json(sweep_id)

    print(f"Total runs in grid search: {LC.HI}{get_total_runs_in_sweep()}{LC.RESET}\n")
    print(f"\nRun the sweep with: {LC.HI}uv run cbfk/sweep_agent.py --wandb-project {args.wandb_project} --sweep-id {sweep_id}{LC.RESET}")



if __name__ == "__main__":
    main()
