"""
Model configuration definitions for ASR experiments.
Optimized for Apple Silicon M3 Pro with 36GB unified memory.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
import torch
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for ASR models."""
    name: str
    model_id: str
    device: str
    torch_dtype: torch.dtype
    use_flash_attention: bool
    trust_remote_code: bool
    low_cpu_mem_usage: bool
    load_in_8bit: bool
    load_in_4bit: bool
    attn_implementation: Optional[str]
    local_path: Optional[str] = None  # Path to local model directory
    prefer_local: bool = True         # Try local first, then remote
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        config = {
            "name": self.name,
            "model_id": self.model_id,
            "device": self.device,
            "torch_dtype": str(self.torch_dtype),
            "use_flash_attention": self.use_flash_attention,
            "trust_remote_code": self.trust_remote_code,
            "low_cpu_mem_usage": self.low_cpu_mem_usage,
            "load_in_8bit": self.load_in_8bit,
            "load_in_4bit": self.load_in_4bit,
            "attn_implementation": self.attn_implementation
        }
        return config

# Detect device automatically
def get_optimal_device() -> str:
    """Detect the best available device for Apple Silicon."""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

# Get optimal dtype for the device
def get_optimal_dtype(device: str) -> torch.dtype:
    """Get optimal dtype for the device."""
    if device == "mps":
        return torch.float16  # MPS works well with float16
    elif device == "cuda":
        return torch.float16
    else:
        return torch.float32

# Predefined model configurations optimized for M3 Pro
OPTIMAL_DEVICE = get_optimal_device()
OPTIMAL_DTYPE = get_optimal_dtype(OPTIMAL_DEVICE)

# Model configurations
MODEL_CONFIGS = {
    "whisper-tiny": ModelConfig(
        name="whisper-tiny",
        model_id="openai/whisper-tiny",
        device=OPTIMAL_DEVICE,
        torch_dtype=OPTIMAL_DTYPE,
        use_flash_attention=False,  # Not available on MPS yet
        trust_remote_code=False,
        low_cpu_mem_usage=True,
        load_in_8bit=False,
        load_in_4bit=False,
        attn_implementation=None
    ),
    
    "whisper-base": ModelConfig(
        name="whisper-base",
        model_id="openai/whisper-base",
        device=OPTIMAL_DEVICE,
        torch_dtype=OPTIMAL_DTYPE,
        use_flash_attention=False,
        trust_remote_code=False,
        low_cpu_mem_usage=True,
        load_in_8bit=False,
        load_in_4bit=False,
        attn_implementation=None
    ),
    
    "whisper-small": ModelConfig(
        name="whisper-small",
        model_id="openai/whisper-small",
        device=OPTIMAL_DEVICE,
        torch_dtype=OPTIMAL_DTYPE,
        use_flash_attention=False,
        trust_remote_code=False,
        low_cpu_mem_usage=True,
        load_in_8bit=False,
        load_in_4bit=False,
        attn_implementation=None
    ),
    
    "whisper-medium": ModelConfig(
        name="whisper-medium",
        model_id="openai/whisper-medium",
        device=OPTIMAL_DEVICE,
        torch_dtype=OPTIMAL_DTYPE,
        use_flash_attention=False,
        trust_remote_code=False,
        low_cpu_mem_usage=True,
        load_in_8bit=False,
        load_in_4bit=False,
        attn_implementation=None
    ),
    
    # Distil-Whisper models (faster inference)
    "distil-whisper-small": ModelConfig(
        name="distil-whisper-small",
        model_id="distil-whisper/distil-small.en",
        device=OPTIMAL_DEVICE,
        torch_dtype=OPTIMAL_DTYPE,
        use_flash_attention=False,
        trust_remote_code=False,
        low_cpu_mem_usage=True,
        load_in_8bit=False,
        load_in_4bit=False,
        attn_implementation=None
    ),
    
    "distil-whisper-medium": ModelConfig(
        name="distil-whisper-medium",
        model_id="distil-whisper/distil-medium.en",
        device=OPTIMAL_DEVICE,
        torch_dtype=OPTIMAL_DTYPE,
        use_flash_attention=False,
        trust_remote_code=False,
        low_cpu_mem_usage=True,
        load_in_8bit=False,
        load_in_4bit=False,
        attn_implementation=None
    )
}

# Default model for quick testing
DEFAULT_MODEL = "whisper-small"

def get_model_config(name: str) -> ModelConfig:
    """Get model configuration by name."""
    if name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {name}. Available models: {list(MODEL_CONFIGS.keys())}")
    return MODEL_CONFIGS[name]

def list_available_models() -> list:
    """List all available model configurations."""
    return list(MODEL_CONFIGS.keys())

# Memory estimates for M3 Pro (36GB unified memory)
MODEL_MEMORY_ESTIMATES = {
    "whisper-tiny": 1.5,      # GB
    "whisper-base": 2.5,      # GB  
    "whisper-small": 4.0,     # GB
    "whisper-medium": 8.0,    # GB
    "distil-whisper-small": 3.0,   # GB
    "distil-whisper-medium": 6.0,  # GB
}

def check_memory_requirements(model_name: str, batch_size: int = 32) -> Dict[str, Any]:
    """Estimate memory requirements for a given model and batch size."""
    base_memory = MODEL_MEMORY_ESTIMATES.get(model_name, 5.0)
    
    # Estimate batch processing overhead (rough approximation)
    batch_memory = batch_size * 0.1  # ~100MB per sample in batch
    total_estimated = base_memory + batch_memory
    
    return {
        "model_memory_gb": base_memory,
        "batch_memory_gb": batch_memory,
        "total_estimated_gb": total_estimated,
        "fits_in_36gb": total_estimated < 30,  # Leave 6GB buffer
        "recommended_batch_size": min(batch_size, int((30 - base_memory) / 0.1)) if total_estimated >= 30 else batch_size
    }

# Local model detection utilities
def find_local_model_paths() -> Dict[str, str]:
    """Find local model paths by scanning common directories."""
    
    # Common directories where models might be stored
    search_dirs = [
        Path("models"),                          # experiments/models/
        Path("../models"),                       # models/ (from notebooks)
        Path("../../models"),                    # models/ (from deeper dirs)
        Path.cwd() / "models",                   # Current directory models/
        Path.cwd().parent / "models",            # Parent directory models/
        Path.home() / ".cache" / "huggingface" / "transformers",  # HF cache
        Path.home() / "models",                  # User home models/
    ]
    
    found_models = {}
    
    # Model name mappings (local name -> model config name)
    model_mappings = {
        "whisper-tiny": "whisper-tiny",
        "whisper-base": "whisper-base", 
        "whisper-small": "whisper-small",
        "whisper-medium": "whisper-medium",
        "distil-whisper-small": "distil-whisper-small",
        "distil-whisper-medium": "distil-whisper-medium",
        # Alternative names
        "openai--whisper-tiny": "whisper-tiny",
        "openai--whisper-base": "whisper-base",
        "openai--whisper-small": "whisper-small", 
        "openai--whisper-medium": "whisper-medium",
        "distil-whisper--distil-small.en": "distil-whisper-small",
        "distil-whisper--distil-medium.en": "distil-whisper-medium"
    }
    
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
            
        logger.debug(f"Searching for models in: {search_dir}")
        
        # Look for model directories
        for item in search_dir.iterdir():
            if not item.is_dir():
                continue
                
            # Check if this looks like a model directory
            if _is_valid_model_directory(item):
                # Map directory name to model config name
                dir_name = item.name
                if dir_name in model_mappings:
                    model_name = model_mappings[dir_name]
                    if model_name not in found_models:
                        found_models[model_name] = str(item)
                        logger.info(f"Found local model {model_name} at: {item}")
    
    return found_models

def _is_valid_model_directory(path: Path) -> bool:
    """Check if a directory contains a valid Whisper model."""
    
    # Required files for a Whisper model
    required_files = [
        "config.json",
        "tokenizer.json",
        "preprocessor_config.json"
    ]
    
    # At least one model file should exist
    model_files = [
        "pytorch_model.bin",
        "model.safetensors", 
        "pytorch_model.safetensors"
    ]
    
    # Check required files
    for req_file in required_files:
        if not (path / req_file).exists():
            return False
    
    # Check at least one model file exists
    has_model_file = any((path / model_file).exists() for model_file in model_files)
    
    return has_model_file

def update_model_configs_with_local_paths():
    """Update MODEL_CONFIGS with detected local paths."""
    
    logger.info("Scanning for local models...")
    local_models = find_local_model_paths()
    
    # Update configs with local paths
    for model_name, local_path in local_models.items():
        if model_name in MODEL_CONFIGS:
            MODEL_CONFIGS[model_name].local_path = local_path
            logger.info(f"Updated {model_name} with local path: {local_path}")
    
    return len(local_models)

def get_model_config_with_local_detection(name: str) -> ModelConfig:
    """Get model configuration with automatic local path detection."""
    
    # Get base config
    config = get_model_config(name)
    
    # If no local path set, try to find one
    if not config.local_path:
        local_models = find_local_model_paths()
        if name in local_models:
            config.local_path = local_models[name]
            logger.info(f"Auto-detected local path for {name}: {config.local_path}")
    
    return config

def list_available_local_models() -> Dict[str, str]:
    """List all locally available models."""
    return find_local_model_paths()

def verify_local_model(model_path: str) -> Dict[str, Any]:
    """Verify a local model directory and return information."""
    
    path = Path(model_path)
    
    if not path.exists():
        return {"valid": False, "error": "Path does not exist"}
    
    if not path.is_dir():
        return {"valid": False, "error": "Path is not a directory"}
    
    if not _is_valid_model_directory(path):
        return {"valid": False, "error": "Directory does not contain valid model files"}
    
    # Get model info
    try:
        import json
        
        # Read config file
        config_file = path / "config.json"
        with open(config_file, 'r') as f:
            model_config = json.load(f)
        
        # Calculate size
        total_size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
        size_mb = total_size / (1024 * 1024)
        
        return {
            "valid": True,
            "path": str(path),
            "size_mb": size_mb,
            "model_type": model_config.get("model_type", "unknown"),
            "vocab_size": model_config.get("vocab_size", "unknown"),
            "files": [f.name for f in path.iterdir() if f.is_file()]
        }
        
    except Exception as e:
        return {"valid": False, "error": f"Error reading model info: {e}"}
