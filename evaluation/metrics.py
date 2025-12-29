"""
Metrics calculation for ASR evaluation.
Supports WER@N, SER, and detailed error analysis.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
from dataclasses import dataclass
import re
import string
from collections import defaultdict
import jiwer
import logging

logger = logging.getLogger(__name__)

@dataclass
class ErrorAnalysis:
    """Detailed error analysis for a single prediction."""
    substitutions: int
    deletions: int
    insertions: int
    total_errors: int
    reference_length: int
    wer: float
    alignment: List[Tuple[str, str]]  # (reference_word, hypothesis_word)

class WERCalculator:
    """Word Error Rate calculator with detailed analysis."""
    
    def __init__(
        self,
        normalize_text: bool = True,
        remove_punctuation: bool = True,
        lowercase: bool = True
    ):
        self.normalize_text = normalize_text
        self.remove_punctuation = remove_punctuation
        self.lowercase = lowercase
        
    def _normalize_text(self, text: str) -> str:
        """Normalize text for consistent WER calculation."""
        
        if not self.normalize_text:
            return text
        
        # Convert to lowercase
        if self.lowercase:
            text = text.lower()
        
        # Remove punctuation
        if self.remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        return text
    
    def calculate_wer(
        self,
        reference: str,
        hypothesis: str,
        return_details: bool = False
    ) -> Union[float, Tuple[float, ErrorAnalysis]]:
        """Calculate WER between reference and hypothesis."""
        
        # Normalize texts
        ref_normalized = self._normalize_text(reference)
        hyp_normalized = self._normalize_text(hypothesis)
        
        # Calculate WER using jiwer
        try:
            wer_score = jiwer.wer(ref_normalized, hyp_normalized)
        except Exception as e:
            logger.warning(f"Error calculating WER: {e}")
            wer_score = 1.0  # Worst case
        
        if not return_details:
            return wer_score
        
        # Get detailed alignment for error analysis  
        try:
            alignment = jiwer.process_words(ref_normalized, hyp_normalized)
            
            error_analysis = ErrorAnalysis(
                substitutions=alignment.substitutions,
                deletions=alignment.deletions,
                insertions=alignment.insertions,
                total_errors=alignment.substitutions + alignment.deletions + alignment.insertions,
                reference_length=alignment.reference_length,
                wer=wer_score,
                alignment=list(zip(
                    ref_normalized.split(),
                    hyp_normalized.split()
                ))
            )
            
        except Exception as e:
            logger.warning(f"Error in detailed analysis: {e}")
            ref_words = ref_normalized.split()
            error_analysis = ErrorAnalysis(
                substitutions=0,
                deletions=0,
                insertions=0,
                total_errors=0,
                reference_length=len(ref_words),
                wer=wer_score,
                alignment=[]
            )
        
        return wer_score, error_analysis
    
    def calculate_wer_at_n(
        self,
        reference: str,
        hypotheses: List[str],
        n: int = 5
    ) -> Dict[str, float]:
        """Calculate WER@N (oracle WER from top-N hypotheses)."""
        
        if not hypotheses:
            return {"wer_1": 1.0, f"wer_{n}": 1.0}
        
        # Calculate WER for each hypothesis
        wer_scores = []
        for hyp in hypotheses[:n]:
            wer_score = self.calculate_wer(reference, hyp)
            wer_scores.append(wer_score)
        
        results = {
            "wer_1": wer_scores[0] if wer_scores else 1.0,
            f"wer_{n}": min(wer_scores) if wer_scores else 1.0  # Oracle (best)
        }
        
        return results

class SERCalculator:
    """Sentence Error Rate calculator."""
    
    def __init__(
        self,
        normalize_text: bool = True,
        remove_punctuation: bool = True,
        lowercase: bool = True
    ):
        self.normalize_text = normalize_text
        self.remove_punctuation = remove_punctuation
        self.lowercase = lowercase
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for consistent SER calculation."""
        
        if not self.normalize_text:
            return text
        
        # Convert to lowercase
        if self.lowercase:
            text = text.lower()
        
        # Remove punctuation
        if self.remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        return text
    
    def calculate_ser(self, reference: str, hypothesis: str) -> float:
        """Calculate SER (0 if exact match, 1 if different)."""
        
        ref_normalized = self._normalize_text(reference)
        hyp_normalized = self._normalize_text(hypothesis)
        
        return 0.0 if ref_normalized == hyp_normalized else 1.0
    
    def calculate_ser_at_n(
        self,
        reference: str,
        hypotheses: List[str],
        n: int = 5
    ) -> Dict[str, float]:
        """Calculate SER@N (oracle SER from top-N hypotheses)."""
        
        if not hypotheses:
            return {"ser_1": 1.0, f"ser_{n}": 1.0}
        
        # Check if any hypothesis matches exactly
        ref_normalized = self._normalize_text(reference)
        
        ser_1 = 1.0
        ser_n = 1.0
        
        for i, hyp in enumerate(hypotheses[:n]):
            hyp_normalized = self._normalize_text(hyp)
            
            if ref_normalized == hyp_normalized:
                if i == 0:
                    ser_1 = 0.0
                ser_n = 0.0
                break
        
        return {"ser_1": ser_1, f"ser_{n}": ser_n}

class ASRMetrics:
    """Comprehensive ASR metrics calculator."""
    
    def __init__(
        self,
        normalize_text: bool = True,
        remove_punctuation: bool = True,
        lowercase: bool = True
    ):
        self.wer_calculator = WERCalculator(normalize_text, remove_punctuation, lowercase)
        self.ser_calculator = SERCalculator(normalize_text, remove_punctuation, lowercase)
        
    def calculate_sample_metrics(
        self,
        reference: str,
        hypotheses: List[str],
        n_values: List[int] = [1, 5]
    ) -> Dict[str, Any]:
        """Calculate all metrics for a single sample."""
        
        metrics = {}
        
        # WER metrics
        for n in n_values:
            wer_results = self.wer_calculator.calculate_wer_at_n(reference, hypotheses, n)
            metrics.update(wer_results)
        
        # SER metrics
        for n in n_values:
            ser_results = self.ser_calculator.calculate_ser_at_n(reference, hypotheses, n)
            metrics.update(ser_results)
        
        # Detailed analysis for best hypothesis
        if hypotheses:
            wer_score, error_analysis = self.wer_calculator.calculate_wer(
                reference, hypotheses[0], return_details=True
            )
            
            metrics.update({
                "substitutions": error_analysis.substitutions,
                "deletions": error_analysis.deletions,
                "insertions": error_analysis.insertions,
                "total_errors": error_analysis.total_errors,
                "reference_length": error_analysis.reference_length
            })
        
        return metrics
    
    def calculate_batch_metrics(
        self,
        references: List[str],
        hypotheses_list: List[List[str]],
        n_values: List[int] = [1, 5],
        domains: Optional[List[str]] = None,
        voices: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Calculate metrics for a batch of samples with breakdowns."""
        
        if len(references) != len(hypotheses_list):
            raise ValueError("References and hypotheses lists must have same length")
        
        # Calculate per-sample metrics
        sample_metrics = []
        for ref, hyps in zip(references, hypotheses_list):
            sample_metric = self.calculate_sample_metrics(ref, hyps, n_values)
            sample_metrics.append(sample_metric)
        
        # Aggregate overall metrics
        overall_metrics = self._aggregate_metrics(sample_metrics, n_values)
        
        # Domain-specific metrics
        domain_metrics = {}
        if domains:
            domain_groups = defaultdict(list)
            for i, domain in enumerate(domains):
                domain_groups[domain].append(sample_metrics[i])
            
            for domain, metrics_group in domain_groups.items():
                domain_metrics[domain] = self._aggregate_metrics(metrics_group, n_values)
        
        # Voice-specific metrics
        voice_metrics = {}
        if voices:
            voice_groups = defaultdict(list)
            for i, voice in enumerate(voices):
                voice_groups[voice].append(sample_metrics[i])
            
            for voice, metrics_group in voice_groups.items():
                voice_metrics[voice] = self._aggregate_metrics(metrics_group, n_values)
        
        results = {
            "overall": overall_metrics,
            "domain_breakdown": domain_metrics,
            "voice_breakdown": voice_metrics,
            "sample_metrics": sample_metrics,
            "total_samples": len(references)
        }
        
        return results
    
    def _aggregate_metrics(
        self,
        sample_metrics: List[Dict[str, Any]],
        n_values: List[int]
    ) -> Dict[str, Any]:
        """Aggregate metrics across samples."""
        
        if not sample_metrics:
            return {}
        
        aggregated = {}
        
        # Aggregate WER and SER metrics
        for n in n_values:
            wer_key = f"wer_{n}"
            ser_key = f"ser_{n}"
            
            if wer_key in sample_metrics[0]:
                wer_values = [m[wer_key] for m in sample_metrics]
                aggregated[wer_key] = {
                    "mean": np.mean(wer_values),
                    "std": np.std(wer_values),
                    "min": np.min(wer_values),
                    "max": np.max(wer_values),
                    "median": np.median(wer_values)
                }
            
            if ser_key in sample_metrics[0]:
                ser_values = [m[ser_key] for m in sample_metrics]
                aggregated[ser_key] = {
                    "mean": np.mean(ser_values),
                    "std": np.std(ser_values),
                    "min": np.min(ser_values),
                    "max": np.max(ser_values),
                    "median": np.median(ser_values)
                }
        
        # Aggregate error counts
        error_fields = ["substitutions", "deletions", "insertions", "total_errors", "reference_length"]
        for field in error_fields:
            if field in sample_metrics[0]:
                values = [m[field] for m in sample_metrics]
                aggregated[field] = {
                    "sum": np.sum(values),
                    "mean": np.mean(values),
                    "std": np.std(values)
                }
        
        return aggregated
    
    def generate_error_report(
        self,
        references: List[str],
        hypotheses_list: List[List[str]],
        utterance_ids: Optional[List[str]] = None,
        domains: Optional[List[str]] = None,
        voices: Optional[List[str]] = None,
        worst_n: int = 10
    ) -> Dict[str, Any]:
        """Generate detailed error analysis report."""
        
        errors = []
        
        for i, (ref, hyps) in enumerate(zip(references, hypotheses_list)):
            if not hyps:
                continue
                
            wer_score, error_analysis = self.wer_calculator.calculate_wer(
                ref, hyps[0], return_details=True
            )
            
            error_info = {
                "sample_idx": i,
                "utterance_id": utterance_ids[i] if utterance_ids else f"sample_{i}",
                "domain": domains[i] if domains else "unknown",
                "voice": voices[i] if voices else "unknown",
                "reference": ref,
                "hypothesis": hyps[0],
                "wer": wer_score,
                "substitutions": error_analysis.substitutions,
                "deletions": error_analysis.deletions,
                "insertions": error_analysis.insertions,
                "total_errors": error_analysis.total_errors,
                "reference_length": error_analysis.reference_length
            }
            
            errors.append(error_info)
        
        # Sort by WER (worst first)
        errors.sort(key=lambda x: x["wer"], reverse=True)
        
        # Common error patterns
        error_patterns = self._analyze_error_patterns(errors)
        
        report = {
            "worst_errors": errors[:worst_n],
            "error_patterns": error_patterns,
            "total_samples": len(errors),
            "samples_with_errors": sum(1 for e in errors if e["wer"] > 0),
            "perfect_samples": sum(1 for e in errors if e["wer"] == 0)
        }
        
        return report
    
    def _analyze_error_patterns(self, errors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze common error patterns."""
        
        # Group errors by domain and voice
        domain_errors = defaultdict(list)
        voice_errors = defaultdict(list)
        
        for error in errors:
            if error["wer"] > 0:  # Only consider samples with errors
                domain_errors[error["domain"]].append(error["wer"])
                voice_errors[error["voice"]].append(error["wer"])
        
        patterns = {
            "domain_error_rates": {
                domain: {
                    "mean_wer": np.mean(wers),
                    "error_count": len(wers),
                    "samples_with_errors": len([w for w in wers if w > 0])
                }
                for domain, wers in domain_errors.items()
            },
            "voice_error_rates": {
                voice: {
                    "mean_wer": np.mean(wers),
                    "error_count": len(wers),
                    "samples_with_errors": len([w for w in wers if w > 0])
                }
                for voice, wers in voice_errors.items()
            }
        }
        
        return patterns
