"""
TTS Configuration for Kokoro 82M Text-to-Speech System
Handles model settings, audio parameters, and processing options
"""

import os
from pathlib import Path

class TTSConfig:
    """Configuration class for TTS system"""
    
    # Model Configuration
    MODEL_NAME = "hexgrad/Kokoro-82M"
    MODEL_CACHE_DIR = Path(__file__).parent.parent / "models"
    
    # Audio Settings
    SAMPLE_RATE = 22050  # Standard sample rate for TTS
    AUDIO_FORMAT = "wav"  # Output format (wav, mp3)
    BIT_DEPTH = 16  # Audio bit depth
    
    # Processing Settings
    BATCH_SIZE = 1  # Process one utterance at a time for quality
    MAX_LENGTH = 1000  # Maximum characters per utterance
    DEVICE = "cuda" if os.getenv("CUDA_AVAILABLE", "false").lower() == "true" else "cpu"
    
    # File Paths
    BASE_DIR = Path(__file__).parent.parent
    INPUT_DIR = BASE_DIR.parent  # Main project directory with JSON files
    OUTPUT_DIR = BASE_DIR / "output"
    SCRIPTS_DIR = BASE_DIR / "scripts"
    
    # Domain-specific output directories
    DOMAIN_DIRS = {
        "medical": OUTPUT_DIR / "medical",
        "financial": OUTPUT_DIR / "financial", 
        "legal": OUTPUT_DIR / "legal",
        "technical": OUTPUT_DIR / "technical"
    }
    
    # Input file patterns
    INPUT_FILES = {
        "medical": "medical_sample_utterances.json",
        "financial": "financial_sample_utterances.json",
        "legal": "legal_sample_utterances.json", 
        "technical": "technical_sample_utterances.json"
    }
    
    # Voice Settings (if Kokoro supports multiple voices)
    DEFAULT_VOICE = "default"
    VOICE_SPEED = 1.0  # Speech rate multiplier
    
    # Quality Settings
    NORMALIZE_AUDIO = True  # Normalize output volume
    ADD_SILENCE = 0.5  # Seconds of silence at start/end
    
    @classmethod
    def ensure_directories(cls):
        """Create necessary directories if they don't exist"""
        cls.MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        for domain_dir in cls.DOMAIN_DIRS.values():
            domain_dir.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_input_file_path(cls, domain):
        """Get full path to input JSON file for a domain"""
        return cls.INPUT_DIR / cls.INPUT_FILES.get(domain, f"{domain}_sample_utterances.json")
    
    @classmethod
    def get_output_dir(cls, domain):
        """Get output directory for a specific domain"""
        return cls.DOMAIN_DIRS.get(domain, cls.OUTPUT_DIR / domain)
