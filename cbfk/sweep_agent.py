#!/usr/bin/env python3
"""
Weights & Biases sweep agent for parameter space exploration.
"""

import argparse
import gc
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any, Dict

import yaml

import cbfk.experiment_config
from cbfk.log.log_config import LogConfig as LC
from cbfk.log.wandb_logger import WandbLogger
from cbfk.models import get_model_registry
from cbfk.param_explorer import ParameterExplorer

# Set the TOKENIZERS_PARALLELISM environment variable to avoid forking warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configure logging
LC.setup_logging(log_level=logging.INFO)
logger = logging.getLogger(__name__)


def get_git_commit_hash() -> str:
    """
    Get the git hash of the current commit.
    """
    commit_hash = subprocess.check_output(['git', 'rev-parse', '--short=7', 'HEAD']).decode('utf-8').strip()
    return commit_hash


def is_invalid_config(config: Dict[str, Any]) -> str:
    """
    Check if a configuration is invalid.
    Returns None if the configuration is valid, otherwise returns a message.
    """

    embedding_model = config['embedding_model']
    model_max_sequence_length = get_model_registry().get_max_sequence_length(embedding_model)
    
    if config['chunk_size'] > model_max_sequence_length:
        return f"Chunk size {config['chunk_size']} is larger than {embedding_model} max sequence length {model_max_sequence_length}"

    return None


def train_with_config(config: Dict[str, Any], 
                      corpus_dir: Path, 
                      results_dir: Path, 
                      persist_dir: Path,
                      wandb_logger: WandbLogger) -> None:
    """
    Run a single experiment with the given configuration.
    
    Args:
        config: Configuration from wandb sweep
        corpus_dir: Directory containing documents to ingest
        results_dir: Directory to store experiment results
        persist_dir: Base directory to store indices
        wandb_logger: WandbLogger instance with an active run
    """
    # Let WandbLogger manage the heartbeat - no need for manual heartbeat thread

    # Check for invalid hyperparameter combinations
    invalid_config = is_invalid_config(config)
    if invalid_config:
        logger.info(f"{LC.HI}Skipping invalid hyperparameter combo: {invalid_config}{LC.RESET}")
        result = config
        result["is_valid_config"] = False   
    else:           

        # Fallback: set corpus_dir from config if not provided
        if corpus_dir is None and "corpus_path" in config:
            corpus_dir = Path(config["corpus_path"])

        # Create experiment config from wandb config in a pythonic way
        from dataclasses import fields
        config_fields: set[str] = {f.name for f in fields(cbfk.experiment_config.ExperimentConfig)}
        # Log a warning for each config key not in ExperimentConfig
        for k in config:
            if k not in config_fields:
                logger.warning(f"Config key '{k}' is not a field in ExperimentConfig and will be ignored.")
        filtered_config: dict[str, object] = {k: v for k, v in config.items() if k in config_fields}
        # Convert splitter_type from string to enum if needed
        if isinstance(filtered_config.get("splitter_type"), str):
            filtered_config["splitter_type"] = cbfk.experiment_config.SplitterType.from_str(filtered_config["splitter_type"])
        exp_config = cbfk.experiment_config.ExperimentConfig(**filtered_config)
        
        logger.info(f"Starting experiment with config: {LC.AMBER}{exp_config.experiment_id}{LC.RESET}")
        
        # Initialize parameter explorer
        explorer = ParameterExplorer(
            corpus_dir=corpus_dir,
            results_dir=results_dir,
            base_persist_dir=persist_dir,
            wandb_logger=wandb_logger
        )
        
        # Evaluate the configuration
        result = explorer.run_this_configuration(exp_config)
        result['is_valid_config'] = True

        # Log the html reports
        wandb_logger.log_question_report(result)

    # Log experiment result, even if invalid config
    wandb_logger.log_experiment_results(result)
    return


def run_config_from_file(args: argparse.Namespace, config_from_file: dict[str, Any]) -> str:
    """
    Run a single configuration from a file that will be part of the sweep.
    """
    # Rather than trying to use internal APIs, we'll create a standard run
    # and try to manually connect it to the sweep
    logger.info(f"Initializing manual run with config from file for sweep {args.sweep_id}")
    
    # Create a run with appropriate tagging to associate with the sweep
    import wandb
    run = wandb.init(
        project=args.wandb_project, 
        entity=args.wandb_entity,
        group=args.sweep_id,     # Group this run with others in the sweep
        job_type="rerun",        # Mark as a rerun
        name=f"rerun-{wandb.util.generate_id()}",
        tags=[args.sweep_id],     # Tag with sweep ID
        config=config_from_file   # Use the config from file
    )
    
    # Try to manually associate with the sweep by updating config
    run.config.update({"_wandb": {"sweep": {"id": args.sweep_id}}}, allow_val_change=True)
    
    logger.info(f"Created run {run.id} for config rerun")
    
    # Create an experiment-specific logger
    experiment_logger = WandbLogger(
        project=args.wandb_project,
        entity=args.wandb_entity,
        is_sweep_agent=True,
        enabled=True
    )
    experiment_logger.run = run  # Set the run directly on this logger
    
    # Get the config from wandb
    config = dict(run.config)
    
    # Run the experiment
    train_with_config(
        config=config,
        corpus_dir=args.corpus_dir,
        results_dir=args.results_dir,
        persist_dir=args.persist_dir,
        wandb_logger=experiment_logger,
    )
    
    # Finish the run
    run.finish()    
    gc.collect()
    return run.id
    
    
def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run a wandb sweep agent")
    parser.add_argument(
        "--sweep-id",
        type=str,
        required=False,
        help="Sweep ID to run, create sweep ID with sweep_config.py"
    )
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=None,
        help="Directory containing documents to ingest"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory to store experiment results"
    )
    parser.add_argument(
        "--persist-dir",
        type=Path,
        default=Path("index/experiments"),
        help="Base directory to store indices"
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="Weights & Biases project name"
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="Weights & Biases team/entity name"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="Number of runs to execute (default: unlimited)"
    )
    parser.add_argument(
        "--config-yaml",
        type=Path,
        default=None,
        help="Path to config.yaml file from a failed run to rerun (requires --sweep-id)"
    )
    parser.add_argument(
        "--config-json",
        type=Path,
        default=None,
        help="Path to config.json file from a failed run to rerun (requires --sweep-id)"
    )
    return parser


def extract_sweep_id_from_launch_json(config_name: str = "sweep_agent") -> str | None:
    """
    Extract the sweep_id from .vscode/launch.json for the given config name.
    """
    import json
    from pathlib import Path

    launch_path = Path(__file__).parent.parent / ".vscode" / "launch.json"
    if not launch_path.exists():
        logger.error(f"Could not find {launch_path}")
        return None
    with launch_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    configs = data.get("configurations", [])
    for conf in configs:
        if conf.get("name") == config_name:
            args = conf.get("args", [])
            # Look for --sweep-id followed by the value
            for i, val in enumerate(args):
                if val == "--sweep-id" and i + 1 < len(args):
                    return args[i + 1]
    logger.error(f"Could not find sweep_id in {launch_path} for config '{config_name}'")
    return None


def load_config_from_yaml(config_path: Path) -> Dict[str, Any]:
    """Load configuration from a YAML file."""
    logger.info(f"Loading configuration from YAML file: {LC.HI}{config_path}{LC.RESET}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_config_from_json(config_path: Path) -> Dict[str, Any]:
    """Load configuration from a JSON file."""
    logger.info(f"Loading configuration from JSON file: {LC.HI}{config_path}{LC.RESET}")
    with open(config_path, "r") as f:
        return json.load(f)


def main() -> None:
    """Main function to run the sweep agent."""
    parser = parse_args()
    args = parser.parse_args()
    
    # Check if a config file is provided
    config_from_file = None
    if args.config_yaml:
        if not args.config_yaml.exists():
            raise FileNotFoundError(f"Config YAML file not found: {args.config_yaml}")
        config_from_file = load_config_from_yaml(args.config_yaml)
    elif args.config_json:
        if not args.config_json.exists():
            raise FileNotFoundError(f"Config JSON file not found: {args.config_json}")
        config_from_file = load_config_from_json(args.config_json)

    # If wandb-project is cbfk-debug and sweep-id is missing, extract from launch.json
    if args.wandb_project == "cbfk-debug" and not args.sweep_id:
        sweep_id = extract_sweep_id_from_launch_json()
        logger.info(f"Extracted sweep_id from launch.json: {LC.HI}{sweep_id}{LC.RESET}")
        if not sweep_id:
            raise RuntimeError("--sweep-id not provided and could not be found in .vscode/launch.json for config 'sweep_agent'")
        args.sweep_id = sweep_id

    # Ensure directories exist
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.persist_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Starting sweep agent for sweep ID: {args.sweep_id}")
    logger.info(f"Corpus directory:  {LC.HI}{args.corpus_dir}{LC.RESET}")
    logger.info(f"Results directory: {LC.HI}{args.results_dir}{LC.RESET}")
    logger.info(f"Persist directory: {LC.HI}{args.persist_dir}{LC.RESET}")

    # Create a WandbLogger instance for the agent itself
    wandb_logger = WandbLogger(
        project=args.wandb_project,
        entity=args.wandb_entity,
        is_sweep_agent=False,  # This is the parent logger, not the sweep agent
        enabled=True
    )

    # Make sure we have a sweep ID in all cases
    if not args.sweep_id:
        raise ValueError("--sweep-id is required for both sweep agent and config rerun")
        
    logger.info(f"Using sweep ID: {LC.HI}{args.sweep_id}{LC.RESET}")
    
    # Handle config file rerun differently from regular sweep agent
    if config_from_file:

        logger.info("Running a single configuration from config file")
        try:

            run_id = run_config_from_file(args, config_from_file)
            logger.info(f"Run completed with ID: {LC.HI}{run_id}{LC.RESET}")

        except Exception as e:
            logger.error(f"Error in rerun: {LC.RED}{e!s}{LC.RESET}")
            logger.error(f'Exception class: {LC.RED}{e.__class__!s}{LC.RESET}')
            import traceback
            traceback.print_exc()
    else:
        # Start the normal sweep agent for regular sweep runs
        logger.info("Running normal sweep agent")

        # Define the train function for regular sweep runs (not config file reruns)
        def train() -> None:
            """
            Training function called by the wandb agent for each run.
            This function creates a new WandbLogger instance for each experiment.
            """
            import wandb  # Import wandb here to ensure it's available in the function scope

            try:
                # Initialize wandb first
                run = wandb.init()
                if run is None:
                    logger.error("Failed to initialize wandb run")
                    return
                logger.info(f"Successfully initialized wandb run: {run.id}")

                # Get the sweep parameters from run.config
                sweep_params: dict[str, object] = dict(run.config)

                # Tag this run with the sweep_id
                sweep_id: str = args.sweep_id
                if sweep_id not in run.tags:
                    run.tags = [*list(run.tags), sweep_id]
                    run.save()
                logger.info(f"Tagged run {run.id} with sweep_id: {sweep_id}")

                # Create a NEW logger specific to this experiment and bind the correct run
                experiment_logger = WandbLogger(
                    project=args.wandb_project,
                    entity=args.wandb_entity,
                    is_sweep_agent=True,
                    enabled=True
                )
                experiment_logger.run = run  # Use the run from the original wandb.init()

                # Use the sweep_params (with original keys) for the experiment logic
                train_with_config(
                    config=sweep_params,
                    corpus_dir=args.corpus_dir,
                    results_dir=args.results_dir,
                    persist_dir=args.persist_dir,
                    wandb_logger=experiment_logger,  # Use the experiment-specific logger
                )
            except Exception as e:
                logger.error(f"Error in train function: {LC.RED}{e!s}{LC.RESET}")
                logger.error(f'Exception class: {LC.RED}{e.__class__}{LC.RESET}')
                import traceback
                traceback.print_exc()
                raise e

        # Start the sweep agent with train() as the function
        import wandb
        wandb.agent(
            sweep_id=args.sweep_id,
            project=args.wandb_project,
            entity=args.wandb_entity,
            function=train,
            count=args.count
        )

    # Finish the parent logger
    wandb_logger.finish()
    
    logger.info("Sweep agent completed")

    # Create the sentinel file to signal sweep completion
    sentinel_path = Path(__file__).parent.parent / "temp" / "sweep_done.txt"
    sentinel_path.parent.mkdir(parents=True, exist_ok=True)
    sentinel_path.touch()


if __name__ == "__main__":
    main()
