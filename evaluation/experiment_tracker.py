"""
Experiment tracking and results management.
Handles experiment logging, results storage, and analysis.
"""

from typing import Dict, Any, List, Optional
import json
import pickle
from pathlib import Path
from datetime import datetime
import logging
import pandas as pd
import numpy as np

from config.experiment_configs import ExperimentConfig

logger = logging.getLogger(__name__)

class ExperimentTracker:
    """Track and manage ASR experiments with detailed logging."""
    
    def __init__(self, results_dir: str = "experiments/results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.results_dir / "experiment_logs").mkdir(exist_ok=True)
        (self.results_dir / "predictions").mkdir(exist_ok=True)
        (self.results_dir / "metrics").mkdir(exist_ok=True)
        
        self.current_experiment = None
        
    def start_experiment(self, config: ExperimentConfig) -> str:
        """Start a new experiment and return experiment ID."""
        
        self.current_experiment = {
            "config": config,
            "start_time": datetime.now().isoformat(),
            "status": "running",
            "results": {},
            "predictions": [],
            "metrics": {},
            "logs": []
        }
        
        experiment_id = config.experiment_id
        
        # Save initial experiment config
        self._save_experiment_config(experiment_id, config)
        
        logger.info(f"Started experiment: {experiment_id}")
        logger.info(f"Description: {config.description}")
        
        return experiment_id
    
    def log_batch_results(
        self,
        batch_predictions: List[str],
        batch_references: List[str],
        batch_hypotheses: List[List[str]],
        batch_metadata: Dict[str, Any],
        batch_metrics: Dict[str, Any]
    ) -> None:
        """Log results from a batch of samples."""
        
        if self.current_experiment is None:
            raise RuntimeError("No active experiment. Call start_experiment() first.")
        
        # Store predictions
        for i, (pred, ref, hyps) in enumerate(zip(batch_predictions, batch_references, batch_hypotheses)):
            prediction_entry = {
                "sample_idx": len(self.current_experiment["predictions"]) + i,
                "prediction": pred,
                "reference": ref,
                "hypotheses": hyps,
                "metadata": {
                    "utterance_id": batch_metadata.get("utterance_id", [None])[i],
                    "domain": batch_metadata.get("domain", [None])[i],
                    "voice": batch_metadata.get("voice", [None])[i],
                    "asr_difficulty": batch_metadata.get("asr_difficulty", [None])[i],
                },
                "timestamp": datetime.now().isoformat()
            }
            self.current_experiment["predictions"].append(prediction_entry)
        
        # Store batch metrics
        batch_id = f"batch_{len(self.current_experiment['results'])}"
        self.current_experiment["results"][batch_id] = {
            "metrics": batch_metrics,
            "sample_count": len(batch_predictions),
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Logged batch with {len(batch_predictions)} samples")
    
    def update_experiment_metrics(self, final_metrics: Dict[str, Any]) -> None:
        """Update experiment with final aggregated metrics."""
        
        if self.current_experiment is None:
            raise RuntimeError("No active experiment. Call start_experiment() first.")
        
        self.current_experiment["metrics"] = final_metrics
        logger.info("Updated experiment with final metrics")
    
    def finish_experiment(self, success: bool = True, error_message: str = None) -> str:
        """Finish the current experiment and save all results."""
        
        if self.current_experiment is None:
            raise RuntimeError("No active experiment. Call start_experiment() first.")
        
        experiment_id = self.current_experiment["config"].experiment_id
        
        # Update experiment status
        self.current_experiment["end_time"] = datetime.now().isoformat()
        self.current_experiment["status"] = "completed" if success else "failed"
        
        if error_message:
            self.current_experiment["error_message"] = error_message
        
        # Calculate experiment duration
        start_time = datetime.fromisoformat(self.current_experiment["start_time"])
        end_time = datetime.fromisoformat(self.current_experiment["end_time"])
        duration = (end_time - start_time).total_seconds()
        self.current_experiment["duration_seconds"] = duration
        
        # Save complete experiment results
        self._save_experiment_results(experiment_id, self.current_experiment)
        
        # Save predictions separately
        if self.current_experiment["config"].save_predictions:
            self._save_predictions(experiment_id, self.current_experiment["predictions"])
        
        # Save detailed metrics
        if self.current_experiment["config"].save_detailed_metrics:
            self._save_detailed_metrics(experiment_id, self.current_experiment["metrics"])
        
        logger.info(f"Finished experiment: {experiment_id}")
        logger.info(f"Duration: {duration:.2f} seconds")
        logger.info(f"Status: {self.current_experiment['status']}")
        
        # Clear current experiment
        experiment_summary = self.current_experiment.copy()
        self.current_experiment = None
        
        return experiment_id
    
    def _save_experiment_config(self, experiment_id: str, config: ExperimentConfig) -> None:
        """Save experiment configuration."""
        
        config_path = self.results_dir / "experiment_logs" / f"{experiment_id}_config.json"
        
        with open(config_path, 'w') as f:
            json.dump(config.to_dict(), f, indent=2)
    
    def _save_experiment_results(self, experiment_id: str, experiment_data: Dict[str, Any]) -> None:
        """Save complete experiment results."""
        
        results_path = self.results_dir / "experiment_logs" / f"{experiment_id}_results.json"
        
        # Create a copy without predictions (save separately)
        results_copy = experiment_data.copy()
        results_copy["predictions"] = f"See {experiment_id}_predictions.json"
        
        with open(results_path, 'w') as f:
            json.dump(results_copy, f, indent=2, default=str)
    
    def _save_predictions(self, experiment_id: str, predictions: List[Dict[str, Any]]) -> None:
        """Save predictions separately."""
        
        predictions_path = self.results_dir / "predictions" / f"{experiment_id}_predictions.json"
        
        with open(predictions_path, 'w') as f:
            json.dump(predictions, f, indent=2)
    
    def _save_detailed_metrics(self, experiment_id: str, metrics: Dict[str, Any]) -> None:
        """Save detailed metrics."""
        
        metrics_path = self.results_dir / "metrics" / f"{experiment_id}_metrics.json"
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
    
    def load_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """Load a complete experiment by ID."""
        
        results_path = self.results_dir / "experiment_logs" / f"{experiment_id}_results.json"
        config_path = self.results_dir / "experiment_logs" / f"{experiment_id}_config.json"
        predictions_path = self.results_dir / "predictions" / f"{experiment_id}_predictions.json"
        metrics_path = self.results_dir / "metrics" / f"{experiment_id}_metrics.json"
        
        experiment_data = {}
        
        # Load results
        if results_path.exists():
            with open(results_path, 'r') as f:
                experiment_data.update(json.load(f))
        
        # Load config
        if config_path.exists():
            with open(config_path, 'r') as f:
                experiment_data["config"] = json.load(f)
        
        # Load predictions
        if predictions_path.exists():
            with open(predictions_path, 'r') as f:
                experiment_data["predictions"] = json.load(f)
        
        # Load detailed metrics
        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                experiment_data["detailed_metrics"] = json.load(f)
        
        return experiment_data
    
    def list_experiments(self) -> List[Dict[str, Any]]:
        """List all experiments with summary information."""
        
        experiments = []
        
        results_dir = self.results_dir / "experiment_logs"
        for results_file in results_dir.glob("*_results.json"):
            experiment_id = results_file.stem.replace("_results", "")
            
            try:
                with open(results_file, 'r') as f:
                    data = json.load(f)
                
                summary = {
                    "experiment_id": experiment_id,
                    "description": data.get("config", {}).get("description", ""),
                    "model": data.get("config", {}).get("model", {}).get("name", "unknown"),
                    "decoding": data.get("config", {}).get("decoding", {}).get("strategy", "unknown"),
                    "dataset": data.get("config", {}).get("dataset", {}).get("name", "unknown"),
                    "status": data.get("status", "unknown"),
                    "start_time": data.get("start_time", "unknown"),
                    "duration_seconds": data.get("duration_seconds", 0),
                    "wer_1": self._extract_metric(data, "wer_1"),
                    "wer_5": self._extract_metric(data, "wer_5"),
                    "ser_1": self._extract_metric(data, "ser_1"),
                    "total_samples": data.get("metrics", {}).get("total_samples", 0)
                }
                
                experiments.append(summary)
                
            except Exception as e:
                logger.warning(f"Error loading experiment {experiment_id}: {e}")
        
        # Sort by start time (newest first)
        experiments.sort(key=lambda x: x["start_time"], reverse=True)
        
        return experiments
    
    def _extract_metric(self, experiment_data: Dict[str, Any], metric_name: str) -> Optional[float]:
        """Extract a metric value from experiment data."""
        
        try:
            overall_metrics = experiment_data.get("metrics", {}).get("overall", {})
            metric_data = overall_metrics.get(metric_name, {})
            
            if isinstance(metric_data, dict):
                return metric_data.get("mean", None)
            else:
                return metric_data
        except:
            return None
    
    def compare_experiments(self, experiment_ids: List[str]) -> pd.DataFrame:
        """Compare multiple experiments in a DataFrame."""
        
        comparison_data = []
        
        for exp_id in experiment_ids:
            try:
                exp_data = self.load_experiment(exp_id)
                
                row = {
                    "experiment_id": exp_id,
                    "model": exp_data.get("config", {}).get("model", {}).get("name", "unknown"),
                    "decoding": exp_data.get("config", {}).get("decoding", {}).get("strategy", "unknown"),
                    "dataset": exp_data.get("config", {}).get("dataset", {}).get("name", "unknown"),
                    "batch_size": exp_data.get("config", {}).get("processing", {}).get("batch_size", 0),
                    "duration_seconds": exp_data.get("duration_seconds", 0),
                    "total_samples": exp_data.get("metrics", {}).get("total_samples", 0)
                }
                
                # Add metrics
                overall_metrics = exp_data.get("metrics", {}).get("overall", {})
                for metric_name in ["wer_1", "wer_5", "ser_1", "ser_5"]:
                    metric_value = overall_metrics.get(metric_name, {})
                    if isinstance(metric_value, dict):
                        row[f"{metric_name}_mean"] = metric_value.get("mean", None)
                        row[f"{metric_name}_std"] = metric_value.get("std", None)
                    else:
                        row[f"{metric_name}_mean"] = metric_value
                
                comparison_data.append(row)
                
            except Exception as e:
                logger.warning(f"Error loading experiment {exp_id} for comparison: {e}")
        
        return pd.DataFrame(comparison_data)
    
    def generate_experiment_report(self, experiment_id: str) -> Dict[str, Any]:
        """Generate a comprehensive report for an experiment."""
        
        exp_data = self.load_experiment(experiment_id)
        
        if not exp_data:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        # Basic info
        report = {
            "experiment_id": experiment_id,
            "description": exp_data.get("config", {}).get("description", ""),
            "start_time": exp_data.get("start_time", "unknown"),
            "end_time": exp_data.get("end_time", "unknown"),
            "duration_seconds": exp_data.get("duration_seconds", 0),
            "status": exp_data.get("status", "unknown")
        }
        
        # Configuration summary
        config = exp_data.get("config", {})
        report["configuration"] = {
            "model": config.get("model", {}),
            "decoding": config.get("decoding", {}),
            "dataset": config.get("dataset", {}),
            "processing": config.get("processing", {})
        }
        
        # Metrics summary
        metrics = exp_data.get("metrics", {})
        report["metrics"] = {
            "overall": metrics.get("overall", {}),
            "domain_breakdown": metrics.get("domain_breakdown", {}),
            "voice_breakdown": metrics.get("voice_breakdown", {}),
            "total_samples": metrics.get("total_samples", 0)
        }
        
        # Performance statistics
        if exp_data.get("duration_seconds", 0) > 0 and metrics.get("total_samples", 0) > 0:
            report["performance"] = {
                "samples_per_second": metrics["total_samples"] / exp_data["duration_seconds"],
                "seconds_per_sample": exp_data["duration_seconds"] / metrics["total_samples"]
            }
        
        return report
    
    def cleanup_old_experiments(self, keep_days: int = 30) -> int:
        """Clean up experiment files older than specified days."""
        
        cutoff_date = datetime.now() - pd.Timedelta(days=keep_days)
        deleted_count = 0
        
        for results_file in (self.results_dir / "experiment_logs").glob("*_results.json"):
            try:
                with open(results_file, 'r') as f:
                    data = json.load(f)
                
                start_time = datetime.fromisoformat(data.get("start_time", ""))
                
                if start_time < cutoff_date:
                    experiment_id = results_file.stem.replace("_results", "")
                    
                    # Delete all related files
                    for suffix in ["_results.json", "_config.json"]:
                        file_path = self.results_dir / "experiment_logs" / f"{experiment_id}{suffix}"
                        if file_path.exists():
                            file_path.unlink()
                    
                    # Delete predictions and metrics
                    pred_path = self.results_dir / "predictions" / f"{experiment_id}_predictions.json"
                    if pred_path.exists():
                        pred_path.unlink()
                    
                    metrics_path = self.results_dir / "metrics" / f"{experiment_id}_metrics.json"
                    if metrics_path.exists():
                        metrics_path.unlink()
                    
                    deleted_count += 1
                    logger.info(f"Deleted old experiment: {experiment_id}")
                    
            except Exception as e:
                logger.warning(f"Error processing {results_file}: {e}")
        
        return deleted_count
