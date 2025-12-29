"""
Data loading utilities for ASR experiments.
Handles HuggingFace dataset loading with filtering and batching.
"""

from typing import Dict, Any, List, Optional, Iterator, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, load_from_disk, Dataset as HFDataset
import librosa
import soundfile as sf
from pathlib import Path
import logging

from config.experiment_configs import DatasetConfig, ProcessingConfig

logger = logging.getLogger(__name__)

class ASRDataset(Dataset):
    """PyTorch Dataset wrapper for HuggingFace ASR dataset."""
    
    def __init__(
        self,
        hf_dataset: HFDataset,
        processing_config: ProcessingConfig,
        cache_audio: bool = True
    ):
        self.hf_dataset = hf_dataset
        self.processing_config = processing_config
        self.cache_audio = cache_audio
        self._audio_cache = {} if cache_audio else None
        
    def __len__(self) -> int:
        return len(self.hf_dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single sample from the dataset."""
        sample = self.hf_dataset[idx]
        
        # Extract audio
        audio_data = self._get_audio(sample, idx)
        
        # Prepare output
        output = {
            "audio": audio_data,
            "text": sample["text"],
            "truth": sample["truth"],
            "prompt": sample["prompt"],
            "utterance_id": sample["utterance_id"],
            "domain": sample["domain"],
            "voice": sample["voice"],
            "asr_difficulty": sample.get("asr_difficulty", 0.0),
            "sample_rate": self.processing_config.target_sample_rate
        }
        
        return output
    
    def _get_audio(self, sample: Dict[str, Any], idx: int) -> np.ndarray:
        """Extract and process audio data."""
        
        # Check cache first
        if self.cache_audio and idx in self._audio_cache:
            return self._audio_cache[idx]
        
        # Get audio array from HF dataset
        if "audio" in sample and "array" in sample["audio"]:
            audio_array = sample["audio"]["array"]
            sample_rate = sample["audio"]["sampling_rate"]
        else:
            raise ValueError(f"No audio data found in sample {idx}")
        
        # Convert to numpy if needed
        if torch.is_tensor(audio_array):
            audio_array = audio_array.numpy()
        
        # Resample if needed
        if sample_rate != self.processing_config.target_sample_rate:
            audio_array = librosa.resample(
                audio_array,
                orig_sr=sample_rate,
                target_sr=self.processing_config.target_sample_rate
            )
        
        # Normalize if requested
        if self.processing_config.normalize_audio:
            audio_array = audio_array / np.max(np.abs(audio_array) + 1e-8)
        
        # Truncate if too long
        max_length_samples = int(
            self.processing_config.max_audio_length * self.processing_config.target_sample_rate
        )
        if len(audio_array) > max_length_samples:
            audio_array = audio_array[:max_length_samples]
        
        # Cache if enabled
        if self.cache_audio:
            self._audio_cache[idx] = audio_array
        
        return audio_array

class ASRDataLoader:
    """Data loader for ASR experiments with filtering and batching."""
    
    def __init__(
        self,
        dataset_config: DatasetConfig,
        processing_config: ProcessingConfig
    ):
        self.dataset_config = dataset_config
        self.processing_config = processing_config
        self.hf_dataset = None
        self.pytorch_dataset = None
        
    def load_dataset(self) -> HFDataset:
        """Load and filter the HuggingFace dataset."""
        
        logger.info(f"Loading dataset: {self.dataset_config.name}")
        
        try:
            # Try loading from HuggingFace Hub
            if self.dataset_config.use_auth_token:
                self.hf_dataset = load_dataset(
                    self.dataset_config.name,
                    use_auth_token=True,
                    split="train"  # Assuming single split
                )
            else:
                self.hf_dataset = load_dataset(
                    self.dataset_config.name,
                    split="train"
                )
                
        except Exception as e:
            logger.warning(f"Failed to load from HuggingFace Hub: {e}")
            
            # Try loading from local disk - check multiple possible paths
            possible_paths = [
                Path("final_hf_dataset/agentic_asr_hf_dataset"),  # From root directory
                Path("../final_hf_dataset/agentic_asr_hf_dataset"),  # From experiments directory
                Path("../../final_hf_dataset/agentic_asr_hf_dataset"),  # From notebooks directory
                Path.cwd().parent / "final_hf_dataset" / "agentic_asr_hf_dataset",  # Absolute resolution
                Path.cwd().parent.parent / "final_hf_dataset" / "agentic_asr_hf_dataset",  # From notebooks absolute
                Path.cwd() / "final_hf_dataset" / "agentic_asr_hf_dataset"  # Current directory fallback
            ]

            local_path = None
            for path in possible_paths:
                if path.exists():
                    local_path = path
                    logger.info(f"Found dataset at: {path}")
                    break
                else:
                    logger.debug(f"Dataset not found at: {path}")

            if local_path:
                logger.info(f"Loading from local path: {local_path}")
                self.hf_dataset = load_from_disk(str(local_path))
            else:
                raise RuntimeError(f"Could not find dataset at any of: {[str(p) for p in possible_paths]}")
        
        logger.info(f"Loaded dataset with {len(self.hf_dataset)} samples")
        
        # Apply filters
        self.hf_dataset = self._apply_filters(self.hf_dataset)
        
        logger.info(f"After filtering: {len(self.hf_dataset)} samples")
        
        return self.hf_dataset
    
    def _apply_filters(self, dataset: HFDataset) -> HFDataset:
        """Apply domain, voice, and sample filters."""
        
        filtered_dataset = dataset
        
        # Domain filter
        if self.dataset_config.domain_filter:
            logger.info(f"Filtering by domains: {self.dataset_config.domain_filter}")
            filtered_dataset = filtered_dataset.filter(
                lambda x: x["domain"] in self.dataset_config.domain_filter
            )
        
        # Voice filter  
        if self.dataset_config.voice_filter:
            logger.info(f"Filtering by voices: {self.dataset_config.voice_filter}")
            filtered_dataset = filtered_dataset.filter(
                lambda x: x["voice"] in self.dataset_config.voice_filter
            )
        
        # Shuffle if requested
        if self.dataset_config.shuffle:
            logger.info("Shuffling dataset")
            filtered_dataset = filtered_dataset.shuffle(seed=self.dataset_config.seed)
        
        # Limit samples if requested
        if self.dataset_config.max_samples:
            logger.info(f"Limiting to {self.dataset_config.max_samples} samples")
            filtered_dataset = filtered_dataset.select(range(min(
                self.dataset_config.max_samples,
                len(filtered_dataset)
            )))
        
        return filtered_dataset
    
    def create_pytorch_dataset(self) -> ASRDataset:
        """Create PyTorch dataset from HuggingFace dataset."""
        
        if self.hf_dataset is None:
            self.load_dataset()
        
        self.pytorch_dataset = ASRDataset(
            hf_dataset=self.hf_dataset,
            processing_config=self.processing_config,
            cache_audio=self.processing_config.cache_audio
        )
        
        return self.pytorch_dataset
    
    def _collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate function for batching variable-length audio."""
        
        # Separate fields
        audio_arrays = [item["audio"] for item in batch]
        texts = [item["text"] for item in batch]
        truths = [item["truth"] for item in batch]
        prompts = [item["prompt"] for item in batch]
        utterance_ids = [item["utterance_id"] for item in batch]
        domains = [item["domain"] for item in batch]
        voices = [item["voice"] for item in batch]
        difficulties = [item["asr_difficulty"] for item in batch]
        sample_rates = [item["sample_rate"] for item in batch]
        
        return {
            "audio": audio_arrays,  # Keep as list for variable lengths
            "text": texts,
            "truth": truths,
            "prompt": prompts,
            "utterance_id": utterance_ids,
            "domain": domains,
            "voice": voices,
            "asr_difficulty": difficulties,
            "sample_rate": sample_rates[0]  # Should be same for all
        }
    
    def create_dataloader(self, shuffle: bool = False) -> DataLoader:
        """Create PyTorch DataLoader with optimal settings."""
        
        if self.pytorch_dataset is None:
            self.create_pytorch_dataset()
        
        dataloader = DataLoader(
            dataset=self.pytorch_dataset,
            batch_size=self.processing_config.batch_size,
            shuffle=shuffle,
            num_workers=self.processing_config.num_workers,
            prefetch_factor=self.processing_config.prefetch_factor,
            pin_memory=self.processing_config.pin_memory,
            collate_fn=self._collate_fn,  # Now using class method instead of local function
            drop_last=False
        )
        
        return dataloader
    
    def get_dataset_stats(self) -> Dict[str, Any]:
        """Get statistics about the loaded dataset."""
        
        if self.hf_dataset is None:
            self.load_dataset()
        
        # Domain distribution
        domains = [sample["domain"] for sample in self.hf_dataset]
        domain_counts = {}
        for domain in domains:
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
        
        # Voice distribution
        voices = [sample["voice"] for sample in self.hf_dataset]
        voice_counts = {}
        for voice in voices:
            voice_counts[voice] = voice_counts.get(voice, 0) + 1
        
        # Audio length statistics
        audio_lengths = []
        for sample in self.hf_dataset:
            if "audio" in sample and "array" in sample["audio"]:
                duration = len(sample["audio"]["array"]) / sample["audio"]["sampling_rate"]
                audio_lengths.append(duration)
        
        stats = {
            "total_samples": len(self.hf_dataset),
            "domain_distribution": domain_counts,
            "voice_distribution": voice_counts,
            "audio_length_stats": {
                "mean": np.mean(audio_lengths) if audio_lengths else 0,
                "std": np.std(audio_lengths) if audio_lengths else 0,
                "min": np.min(audio_lengths) if audio_lengths else 0,
                "max": np.max(audio_lengths) if audio_lengths else 0
            }
        }
        
        return stats
    
    def preview_samples(self, n_samples: int = 5) -> List[Dict[str, Any]]:
        """Preview first n samples from the dataset."""
        
        if self.hf_dataset is None:
            self.load_dataset()
        
        preview = []
        for i in range(min(n_samples, len(self.hf_dataset))):
            sample = self.hf_dataset[i]
            preview_sample = {
                "utterance_id": sample["utterance_id"],
                "text": sample["text"],
                "truth": sample["truth"],
                "prompt": sample["prompt"],
                "domain": sample["domain"],
                "voice": sample["voice"],
                "asr_difficulty": sample.get("asr_difficulty", 0.0),
                "audio_length": len(sample["audio"]["array"]) / sample["audio"]["sampling_rate"] if "audio" in sample else 0
            }
            preview.append(preview_sample)
        
        return preview
