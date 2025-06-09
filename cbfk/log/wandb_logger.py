# cbfk/wandb_logger.py
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Optional, Union

import cbfk.log.html_reports
import wandb

logger = logging.getLogger(__name__)


class WandbLogger:
    """Centralized class for managing wandb logging across the codebase."""
    
    def __init__(
        self,
        project: str = "rag-parameter-explorer",
        entity: Optional[str] = None,
        is_sweep_agent: bool = False,
        enabled: bool = True):

        self.project = project
        self.entity = entity
        self.is_sweep_agent = is_sweep_agent
        self.enabled = enabled
        self.run = None
        
        # Heartbeat mechanism
        self._heartbeat_thread = None
        self._stop_heartbeat = threading.Event()
        self._heartbeat_active = False
        
    def _start_heartbeat(self) -> None:
        """Start a heartbeat thread to keep the wandb run active."""
        if self._heartbeat_active:
            return
            
        # Reset the stop event
        self._stop_heartbeat.clear()
        
        # Define the heartbeat function
        def heartbeat_func():
            heartbeat_count = 0
            while not self._stop_heartbeat.is_set():
                try:
                    # Check if run is still active
                    if self.run is not None:
                        # Log a heartbeat to keep the run active
                        self.run.log({"heartbeat": heartbeat_count, "timestamp": time.time()})
                        heartbeat_count += 1
                    else:
                        # Try to reconnect to the run
                        logger.warning("Wandb run not found in heartbeat thread, attempting to reconnect")
                        self.run = wandb.run
                except Exception as e:
                    logger.error(f"Error in wandb heartbeat: {e!s}")
                    
                # Sleep for 15 seconds
                time.sleep(15)
        
        # Start the heartbeat thread
        self._heartbeat_thread = threading.Thread(target=heartbeat_func, daemon=True)
        self._heartbeat_thread.start()
        self._heartbeat_active = True
        logger.info("Started wandb heartbeat thread to keep run active")
    
    def _stop_heartbeat_thread(self) -> None:
        """Stop the heartbeat thread if it's running."""
        if self._heartbeat_active and self._heartbeat_thread is not None:
            self._stop_heartbeat.set()
            self._heartbeat_thread.join(timeout=1.0)
            self._heartbeat_active = False
            logger.info("Stopped wandb heartbeat thread")
            # Do NOT call wandb.finish() here as it could end the run prematurely
                

    def init(self, run_name: Optional[str] = None, config: Optional[dict[str, Any]] = None, reinit: bool = False) -> bool:
        """Initialize a wandb run if enabled."""
        if not self.enabled:
            logger.info("Wandb logging is disabled")
            return False
            
        try:
            if self.is_sweep_agent and not reinit:
                # For sweep agents, wandb.init is called by the agent
                # Just check if we already have a run set
                if self.run is not None:
                    logger.info(f"Using existing wandb run in sweep agent mode: {self.run.name}")
                    # Start heartbeat to keep run active
                    self._start_heartbeat()
                    return True
                    
                # Try to get the current run
                self.run = wandb.run
                if self.run is None:
                    logger.warning("wandb.run is None in sweep agent mode. This may cause issues.")
                    return False
                    
                logger.info(f"Using sweep agent wandb run: {self.run.name}")
                # Start heartbeat to keep run active
                self._start_heartbeat()
                return True
                
            # For non-sweep agent mode or explicit reinit
            self.run = wandb.init(
                project=self.project,
                entity=self.entity,
                name=run_name,
                config=config,
                reinit=reinit
            )
            logger.info(f"Initialized wandb run: {run_name}")

            # Monkey patch to suppress `parameters/` noise
            if self.run is not None:
                try:
                    self.run._config._sanitize_callable = lambda x: x
                    self.run._config._as_dict = lambda *args, **kwargs: {}
                    logger.info("Monkey-patched wandb config to suppress `parameters/` logging")
                except Exception as e:
                    logger.warning(f"Could not patch wandb config: {e}")            
            
            # Start heartbeat to keep run active
            self._start_heartbeat()
            return True

        except Exception as e:
            logger.error(f"Failed to initialize wandb: {e}")
            self.enabled = False
            return False

    
    def log_artifact(self, artifact_name: str, artifact_type: str, path: Union[str, Path]) -> bool:
        """Log an artifact file to wandb."""
        if not self.enabled or not self.run:
            return False
            
        try:
            artifact = wandb.Artifact(name=artifact_name, type=artifact_type)
            artifact.add_file(str(path))
            wandb.log_artifact(artifact)
            return True
        except Exception as e:
            logger.error(f"Error logging artifact to wandb: {e}")
            return False
    
    def log_html(self, name: str, html_content: str) -> bool:
        """Log HTML content to wandb."""
        if not self.enabled:
            return False
            
        try:
            # Always try to get the latest run reference first
            if self.run is None:
                self.run = wandb.run
                
            # Check if we have a valid run
            if self.run is None:
                logger.error("Cannot log HTML to wandb: No active run")
                return False
                
            # Log the HTML content
            self.run.log({name: wandb.Html(html_content)})
            return True
        except Exception as e:
            logger.error(f"Error logging HTML to wandb: {e}")
            return False
            
    def log(self, data: dict[str, Any]) -> bool:
        """Log metrics and data to wandb."""
        if not self.enabled:
            return False
            
        try:
            # Always try to get the latest run reference first
            if self.run is None:
                self.run = wandb.run
                
            # Check if we have a valid run
            if self.run is None:
                logger.error("Cannot log to wandb: No active run")
                return False
                
            # Try to log the data
            try:
                self.run.log(data)
                return True
            except Exception as inner_e:
                # Check if the run is finished
                if "is finished" in str(inner_e):
                    logger.warning(f"Wandb run is finished, attempting to reconnect: {inner_e!s}")
                    # Try to reconnect to the current run
                    if wandb.run is not None and wandb.run.id != self.run.id:
                        logger.info(f"Reconnecting to active wandb run: {wandb.run.name}")
                        self.run = wandb.run
                        try:
                            self.run.log(data)
                            # Restart heartbeat with the new run
                            self._start_heartbeat()
                            return True
                        except Exception as retry_e:
                            logger.error(f"Error logging to wandb after reconnect: {retry_e!s}")
                else:
                    # Some other error occurred
                    logger.error(f"Error logging to wandb: {inner_e!s}")
                    
            return False
        except Exception as e:
            logger.error(f"Error logging to wandb: {e}")
            return False
    
    def finish(self) -> None:
        """Finish the current wandb run."""
        # Stop the heartbeat thread if it's running
        self._stop_heartbeat_thread()
        
        if self.enabled and not self.is_sweep_agent and wandb.run:
            try:
                wandb.finish()
            except Exception as e:
                logger.error(f"Error finishing wandb run: {e}")
                
        
    def log_question_report(self, result: dict[str, Any]) -> bool:
        """Log HTML visualizations of index responses to wandb.
        
        Args:
            experiment_id: ID of the experiment
            query_results: List of query result dictionaries
            index_responses: List of index response objects
            
        Returns:
            True if successful, False otherwise
        """
        if not self.enabled: return False

        experiment_id = result["experiment_id"]
        query_results = result["query_results"]
        index_responses = result["index_responses"]
        prompts = result["prompts"]

        try:
            
            # Create a summary table of all queries and scores
            try:
                summary_html = cbfk.log.html_reports.create_experiment_run_report(
                                                        experiment_id, 
                                                        recall = result["recall"],
                                                        mrr = result["mrr"],
                                                        graded_accuracy = result["graded_accuracy"],
                                                        source_score = result["source_score"],
                                                        query_results = query_results,
                                                        index_responses = index_responses,
                                                        )
                self.log_html("experiment_run_report", summary_html)
            except Exception as e:
                logger.error(f"Error creating experiment run report: {e!s}")
            
            # Log individual visualizations - use range instead of enumerate for safety
            success_count = 0
            for i in range(len(query_results)):
                if i < len(index_responses):
                    try:
                        query_result = query_results[i]
                        index_response = index_responses[i]
                        prompt = prompts[i]
                            
                        # Create HTML visualization
                        html_content = cbfk.log.html_reports.create_html_question_report(experiment_id, query_result, index_response, prompt)
                        if not html_content:
                            logger.warning(f"Empty HTML content for query {i}")
                            continue
                        
                        # Log HTML to wandb using self.log_html instead of direct wandb.log
                        self.log_html(f"question_{i:02d}.report", html_content)
                        
                        success_count += 1
                    except Exception as e:
                        logger.error(f"Error logging visualization for query {i}: {e!s}")
                
            return success_count > 0
        except Exception as e:
            logger.error(f"Error logging HTML visualizations to wandb: {e!s}")
            return False


    def log_index_creation(
        self,
        index_stats: dict[str, Any],
        model_name: str,
        params: dict[str, Any],
        ingestion_time: float,
        persist_dir: Path, ) -> None:
        """
        Log index creation details with a standardized pattern.
        """
        if not self.enabled: return

        html_content = cbfk.log.html_reports.create_index_report_html(index_stats, model_name, params, ingestion_time, persist_dir)
        self.log_html("index_report", html_content)
        with open(persist_dir / "index_report.html", "w") as f:
            f.write(html_content)


    def log_experiment_results(
        self,
        result: dict[str, Any],  ) -> None:
        """
        Log experiment results with a standardized pattern.
        Centralizes all experiment result logging to avoid duplication.
        Args:
            result: Dictionary containing experiment results
        """
        if not self.enabled: return
        
        def try_json_dump(v):
            try:
                return json.dumps(v)
            except Exception:
                return v

        # convert lists and dics to json strings
        rdj = {k: try_json_dump(v) for k, v in result.items()}

        # Filter out non-serializable values from result
        rd = {k: v for k, v in rdj.items() if isinstance(v, (int, float, str, bool, type(None)))}

        # Add question results for each question
        if "query_results" in result:
            qrs = result["query_results"].copy()
            for i, query_result in enumerate(qrs):
                rd[f"question_{i:02d}.question"] = query_result["query"]
                del query_result["query"]
                rd[f"question_{i:02d}.answer"] = query_result["llm_response"]
                del query_result["llm_response"]
                for k, v in query_result.items():
                    rd[f"question_{i:02d}.{k}"] = v

        # Remove too large keys
        rd.pop("prompts", None)
        rd.pop("index.sample_chunks", None)
        rd.pop("query_results", None)
        rd.pop("index.bm25", None)

        self.log(rd)


