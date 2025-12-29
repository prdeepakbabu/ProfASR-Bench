"""
Decoding configuration definitions for ASR experiments.
Supports various decoding strategies including n-best generation.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from enum import Enum

class DecodingStrategy(Enum):
    """Available decoding strategies."""
    GREEDY = "greedy"
    BEAM_SEARCH = "beam_search"
    NUCLEUS_SAMPLING = "nucleus_sampling"
    TOP_K_SAMPLING = "top_k_sampling"

@dataclass
class DecodingConfig:
    """Configuration for decoding parameters."""
    strategy: DecodingStrategy
    max_new_tokens: int
    num_beams: int
    temperature: float
    top_p: float
    top_k: int
    do_sample: bool
    length_penalty: float
    repetition_penalty: float
    no_repeat_ngram_size: int
    early_stopping: bool
    num_return_sequences: int
    return_dict_in_generate: bool
    output_scores: bool
    output_attentions: bool
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "strategy": self.strategy.value,
            "max_new_tokens": self.max_new_tokens,
            "num_beams": self.num_beams,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "do_sample": self.do_sample,
            "length_penalty": self.length_penalty,
            "repetition_penalty": self.repetition_penalty,
            "no_repeat_ngram_size": self.no_repeat_ngram_size,
            "early_stopping": self.early_stopping,
            "num_return_sequences": self.num_return_sequences,
            "return_dict_in_generate": self.return_dict_in_generate,
            "output_scores": self.output_scores,
            "output_attentions": self.output_attentions
        }
    
    def get_generation_kwargs(self) -> Dict[str, Any]:
        """Get kwargs for model.generate() method."""
        kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "num_beams": self.num_beams,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "do_sample": self.do_sample,
            "length_penalty": self.length_penalty,
            "repetition_penalty": self.repetition_penalty,
            "no_repeat_ngram_size": self.no_repeat_ngram_size,
            "early_stopping": self.early_stopping,
            "num_return_sequences": self.num_return_sequences,
            "return_dict_in_generate": self.return_dict_in_generate,
            "output_scores": self.output_scores,
            "output_attentions": self.output_attentions
        }
        
        # Remove parameters that are 0 or False to avoid conflicts
        filtered_kwargs = {}
        for key, value in kwargs.items():
            if key in ["temperature", "top_p", "top_k"] and not self.do_sample:
                continue
            if key == "num_beams" and value == 1 and self.strategy == DecodingStrategy.GREEDY:
                continue
            filtered_kwargs[key] = value
            
        return filtered_kwargs

# Predefined decoding configurations

# Standard configurations for WER@1 evaluation
DECODING_CONFIGS = {
    "greedy": DecodingConfig(
        strategy=DecodingStrategy.GREEDY,
        max_new_tokens=440,  # Reduced to leave room for start tokens
        num_beams=1,
        temperature=0.0,
        top_p=1.0,
        top_k=50,
        do_sample=False,
        length_penalty=1.0,
        repetition_penalty=1.0,
        no_repeat_ngram_size=0,
        early_stopping=True,
        num_return_sequences=1,
        return_dict_in_generate=True,
        output_scores=True,
        output_attentions=False
    ),
    
    "beam_search_5": DecodingConfig(
        strategy=DecodingStrategy.BEAM_SEARCH,
        max_new_tokens=440,  # Reduced to leave room for start tokens
        num_beams=5,
        temperature=0.0,
        top_p=1.0,
        top_k=50,
        do_sample=False,
        length_penalty=1.0,
        repetition_penalty=1.0,
        no_repeat_ngram_size=0,
        early_stopping=True,
        num_return_sequences=5,  # Return all beams for n-best analysis
        return_dict_in_generate=True,
        output_scores=True,
        output_attentions=False
    ),
    
    "beam_search_10": DecodingConfig(
        strategy=DecodingStrategy.BEAM_SEARCH,
        max_new_tokens=440,  # Reduced to leave room for start tokens
        num_beams=10,
        temperature=0.0,
        top_p=1.0,
        top_k=50,
        do_sample=False,
        length_penalty=1.0,
        repetition_penalty=1.0,
        no_repeat_ngram_size=0,
        early_stopping=True,
        num_return_sequences=10,  # Return all beams for n-best analysis
        return_dict_in_generate=True,
        output_scores=True,
        output_attentions=False
    ),
    
    # Temperature-based sampling for diversity
    "nucleus_sampling": DecodingConfig(
        strategy=DecodingStrategy.NUCLEUS_SAMPLING,
        max_new_tokens=440,  # Reduced to leave room for start tokens
        num_beams=1,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        do_sample=True,
        length_penalty=1.0,
        repetition_penalty=1.0,
        no_repeat_ngram_size=0,
        early_stopping=True,
        num_return_sequences=5,  # Multiple samples for diversity
        return_dict_in_generate=True,
        output_scores=True,
        output_attentions=False
    ),
    
    # Conservative beam search with length penalty
    "beam_conservative": DecodingConfig(
        strategy=DecodingStrategy.BEAM_SEARCH,
        max_new_tokens=440,  # Reduced to leave room for start tokens
        num_beams=5,
        temperature=0.0,
        top_p=1.0,
        top_k=50,
        do_sample=False,
        length_penalty=0.8,  # Favor shorter sequences
        repetition_penalty=1.1,  # Discourage repetition
        no_repeat_ngram_size=2,
        early_stopping=True,
        num_return_sequences=5,
        return_dict_in_generate=True,
        output_scores=True,
        output_attentions=False
    ),
    
    # Aggressive beam search with length penalty
    "beam_aggressive": DecodingConfig(
        strategy=DecodingStrategy.BEAM_SEARCH,
        max_new_tokens=440,  # Reduced to leave room for start tokens
        num_beams=5,
        temperature=0.0,
        top_p=1.0,
        top_k=50,
        do_sample=False,
        length_penalty=1.2,  # Favor longer sequences
        repetition_penalty=1.0,
        no_repeat_ngram_size=0,
        early_stopping=True,
        num_return_sequences=5,
        return_dict_in_generate=True,
        output_scores=True,
        output_attentions=False
    )
}

# Default configuration for quick testing
DEFAULT_DECODING = "greedy"

def get_decoding_config(name: str) -> DecodingConfig:
    """Get decoding configuration by name."""
    if name not in DECODING_CONFIGS:
        raise ValueError(f"Unknown decoding config: {name}. Available configs: {list(DECODING_CONFIGS.keys())}")
    return DECODING_CONFIGS[name]

def list_available_decoding_configs() -> List[str]:
    """List all available decoding configurations."""
    return list(DECODING_CONFIGS.keys())

def create_custom_decoding_config(
    strategy: str = "beam_search",
    num_beams: int = 5,
    temperature: float = 0.0,
    num_return_sequences: int = 5,
    **kwargs
) -> DecodingConfig:
    """Create a custom decoding configuration."""
    
    base_config = DECODING_CONFIGS["beam_search_5"]
    
    # Override with provided parameters
    custom_config = DecodingConfig(
        strategy=DecodingStrategy(strategy),
        max_new_tokens=kwargs.get("max_new_tokens", base_config.max_new_tokens),
        num_beams=num_beams,
        temperature=temperature,
        top_p=kwargs.get("top_p", base_config.top_p),
        top_k=kwargs.get("top_k", base_config.top_k),
        do_sample=kwargs.get("do_sample", temperature > 0.0),
        length_penalty=kwargs.get("length_penalty", base_config.length_penalty),
        repetition_penalty=kwargs.get("repetition_penalty", base_config.repetition_penalty),
        no_repeat_ngram_size=kwargs.get("no_repeat_ngram_size", base_config.no_repeat_ngram_size),
        early_stopping=kwargs.get("early_stopping", base_config.early_stopping),
        num_return_sequences=num_return_sequences,
        return_dict_in_generate=kwargs.get("return_dict_in_generate", True),
        output_scores=kwargs.get("output_scores", True),
        output_attentions=kwargs.get("output_attentions", False)
    )
    
    return custom_config

# Configuration sets for systematic evaluation
EVALUATION_SETS = {
    "quick_test": ["greedy"],
    "standard_eval": ["greedy", "beam_search_5"],
    "comprehensive_eval": ["greedy", "beam_search_5", "beam_search_10"],
    "diversity_eval": ["greedy", "beam_search_5", "nucleus_sampling"],
    "full_eval": list(DECODING_CONFIGS.keys())
}

def get_evaluation_set(name: str) -> List[str]:
    """Get a predefined set of decoding configurations for evaluation."""
    if name not in EVALUATION_SETS:
        raise ValueError(f"Unknown evaluation set: {name}. Available sets: {list(EVALUATION_SETS.keys())}")
    return EVALUATION_SETS[name]
