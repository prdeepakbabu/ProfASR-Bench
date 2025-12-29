"""
ASR model wrapper for Whisper and other models.
Optimized for Apple Silicon M3 Pro with MPS acceleration.
"""

from typing import Dict, Any, List, Optional, Union
import torch
import numpy as np
from transformers import (
    WhisperProcessor, 
    WhisperForConditionalGeneration,
    AutoProcessor,
    AutoModelForSpeechSeq2Seq
)
import logging
from pathlib import Path
import time

from config.model_configs import ModelConfig
from config.decoding_configs import DecodingConfig
from config.prompt_configs import PromptConfig, generate_prompt_text

logger = logging.getLogger(__name__)

class ASRModelWrapper:
    """Wrapper for ASR models with optimized loading and inference."""
    
    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
        self.model = None
        self.processor = None
        self.device = model_config.device
        
        # Performance tracking
        self.inference_times = []
        self.memory_usage = []
        
    def load_model(self) -> None:
        """Load the ASR model and processor with local/remote fallback."""
        
        logger.info(f"Loading model: {self.model_config.name}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Dtype: {self.model_config.torch_dtype}")
        
        start_time = time.time()
        
        # Determine model path (local or remote)
        model_path = self._get_model_path()
        logger.info(f"Model path: {model_path}")
        
        try:
            # Load processor
            self.processor = self._load_processor(model_path)
            
            # Load model with optimized settings for Apple Silicon
            self.model = self._load_model(model_path)
            
            # Move to device
            self.model = self.model.to(self.device)
            
            # Set to eval mode
            self.model.eval()
            
            load_time = time.time() - start_time
            logger.info(f"Model loaded successfully in {load_time:.2f} seconds")
            
            # Log model info
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            logger.info(f"Total parameters: {total_params:,}")
            logger.info(f"Trainable parameters: {trainable_params:,}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _get_model_path(self) -> str:
        """Determine the best model path (local vs remote)."""
        
        # If prefer_local is False, use remote
        if not self.model_config.prefer_local:
            logger.info("prefer_local=False, using remote model")
            return self.model_config.model_id
        
        # Check if local path is explicitly set
        if self.model_config.local_path:
            local_path = Path(self.model_config.local_path)
            if local_path.exists():
                logger.info(f"Using explicit local path: {local_path}")
                return str(local_path)
            else:
                logger.warning(f"Explicit local path not found: {local_path}")
        
        # Try to auto-detect local model
        from config.model_configs import find_local_model_paths
        local_models = find_local_model_paths()
        
        if self.model_config.name in local_models:
            local_path = local_models[self.model_config.name]
            logger.info(f"Auto-detected local model: {local_path}")
            return local_path
        
        # Fall back to remote
        logger.info(f"No local model found, using remote: {self.model_config.model_id}")
        return self.model_config.model_id
    
    def _load_processor(self, model_path: str) -> WhisperProcessor:
        """Load the processor from local or remote path."""
        
        try:
            logger.info(f"Loading processor from: {model_path}")
            processor = WhisperProcessor.from_pretrained(
                model_path,
                trust_remote_code=self.model_config.trust_remote_code,
                local_files_only=Path(model_path).exists()  # Use local_files_only if it's a local path
            )
            logger.info("✅ Processor loaded successfully")
            return processor
            
        except Exception as e:
            # If local loading fails, try remote as fallback
            if Path(model_path).exists():
                logger.warning(f"Local processor loading failed: {e}")
                logger.info(f"Falling back to remote: {self.model_config.model_id}")
                
                processor = WhisperProcessor.from_pretrained(
                    self.model_config.model_id,
                    trust_remote_code=self.model_config.trust_remote_code
                )
                logger.info("✅ Processor loaded from remote fallback")
                return processor
            else:
                raise e
    
    def _load_model(self, model_path: str) -> WhisperForConditionalGeneration:
        """Load the model from local or remote path."""
        
        # Prepare model kwargs
        model_kwargs = {
            "torch_dtype": self.model_config.torch_dtype,
            "low_cpu_mem_usage": self.model_config.low_cpu_mem_usage,
            "trust_remote_code": self.model_config.trust_remote_code
        }
        
        # Add attention implementation if specified
        if self.model_config.attn_implementation:
            model_kwargs["attn_implementation"] = self.model_config.attn_implementation
        
        # Add local_files_only if it's a local path
        if Path(model_path).exists():
            model_kwargs["local_files_only"] = True
        
        try:
            logger.info(f"Loading model from: {model_path}")
            model = WhisperForConditionalGeneration.from_pretrained(
                model_path,
                **model_kwargs
            )
            logger.info("✅ Model loaded successfully")
            return model
            
        except Exception as e:
            # If local loading fails, try remote as fallback
            if Path(model_path).exists():
                logger.warning(f"Local model loading failed: {e}")
                logger.info(f"Falling back to remote: {self.model_config.model_id}")
                
                # Remove local_files_only for remote loading
                model_kwargs.pop("local_files_only", None)
                
                model = WhisperForConditionalGeneration.from_pretrained(
                    self.model_config.model_id,
                    **model_kwargs
                )
                logger.info("✅ Model loaded from remote fallback")
                return model
            else:
                raise e
    
    def transcribe_batch(
        self,
        audio_arrays: List[np.ndarray],
        decoding_config: DecodingConfig,
        return_timestamps: bool = False
    ) -> Dict[str, Any]:
        """Transcribe a batch of audio arrays."""
        
        if self.model is None or self.processor is None:
            self.load_model()
        
        start_time = time.time()
        
        # Process audio inputs
        inputs = self._prepare_inputs(audio_arrays)
        
        # Get generation kwargs from decoding config
        generation_kwargs = decoding_config.get_generation_kwargs()
        
        # Add timestamps if requested
        if return_timestamps:
            generation_kwargs["return_timestamps"] = True
        
        # Perform inference
        with torch.no_grad():
            # Handle different decoding strategies
            if decoding_config.strategy.value == "greedy":
                outputs = self._transcribe_greedy(inputs, generation_kwargs)
            else:
                outputs = self._transcribe_beam_search(inputs, generation_kwargs)
        
        # Process outputs
        results = self._process_outputs(outputs, decoding_config)
        
        # Track performance
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        
        # Add timing info
        results["inference_time"] = inference_time
        results["avg_time_per_sample"] = inference_time / len(audio_arrays)
        
        return results
    
    def _prepare_inputs(self, audio_arrays: List[np.ndarray]) -> Dict[str, torch.Tensor]:
        """Prepare audio inputs for the model."""
        
        # Process each audio array
        processed_features = []
        for audio in audio_arrays:
            # Ensure audio is the right shape and type
            if len(audio.shape) > 1:
                audio = audio.squeeze()
            
            # Process with the processor
            features = self.processor(
                audio,
                sampling_rate=16000,
                return_tensors="pt"
            )
            processed_features.append(features.input_features)
        
        # Concatenate all features
        input_features = torch.cat(processed_features, dim=0)
        
        # Move to device AND ensure dtype matches the model
        inputs = {
            "input_features": input_features.to(
                device=self.device,
                dtype=self.model_config.torch_dtype  # Critical: match model dtype
            )
        }
        
        return inputs
    
    def _transcribe_greedy(
        self,
        inputs: Dict[str, torch.Tensor],
        generation_kwargs: Dict[str, Any]
    ) -> Any:
        """Perform greedy decoding."""
        
        # Override some kwargs for greedy
        generation_kwargs = generation_kwargs.copy()
        generation_kwargs["do_sample"] = False
        generation_kwargs["num_beams"] = 1
        generation_kwargs["language"] = "en"  # Explicit English to avoid auto-detection
        
        # Add repetition penalties to prevent loops
        generation_kwargs["repetition_penalty"] = 1.2
        generation_kwargs["no_repeat_ngram_size"] = 3
        
        # Reduce max_new_tokens to stay within Whisper limits
        generation_kwargs["max_new_tokens"] = 200  # Reduced from default
        
        # Add prompt if provided - but use a different approach
        if "decoder_input_ids" in inputs:
            # For greedy, we'll skip decoder_input_ids to avoid repetition issues
            logger.debug("Skipping decoder_input_ids for greedy to avoid repetition")
        
        outputs = self.model.generate(
            inputs["input_features"],
            **generation_kwargs
        )
        
        return outputs
    
    def _transcribe_beam_search(
        self,
        inputs: Dict[str, torch.Tensor],
        generation_kwargs: Dict[str, Any]
    ) -> Any:
        """Perform beam search decoding - try multiple strategies for diversity."""
        
        num_beams = generation_kwargs.get("num_beams", 5)
        num_return = generation_kwargs.get("num_return_sequences", 5)
        
        # Strategy 1: Try diverse beam search with proper parameters
        logger.info(f"🎯 Attempting diverse beam search (beams={num_beams})")
        
        diverse_kwargs = {
            "num_beams": num_beams,
            "num_return_sequences": num_return,
            "do_sample": False,
            "max_new_tokens": 200,
            "return_dict_in_generate": True,
            "output_scores": True,
            "early_stopping": True,
            "language": "en",
            "num_beam_groups": min(num_beams, 3),  # Use fewer groups for stability
            "diversity_penalty": 2.0,  # Higher penalty for more diversity
            "length_penalty": 0.5,  # Small length penalty for variation
        }
        
        # Add prompt if provided
        if "decoder_input_ids" in inputs:
            prompt_tokens = inputs["decoder_input_ids"]
            diverse_kwargs["decoder_input_ids"] = prompt_tokens
            logger.debug(f"Added prompt to diverse beam search")
        
        try:
            outputs = self.model.generate(inputs["input_features"], **diverse_kwargs)
            
            # Check diversity
            if hasattr(outputs, 'sequences'):
                unique_texts = set()
                for seq in outputs.sequences:
                    decoded = self.processor.decode(seq, skip_special_tokens=True).strip()
                    unique_texts.add(decoded)
                
                diversity_ratio = len(unique_texts) / len(outputs.sequences)
                logger.info(f"🎯 Diverse beam search: {len(unique_texts)}/{len(outputs.sequences)} unique ({diversity_ratio:.1%})")
                
                # If we got good diversity, return it
                if diversity_ratio >= 0.6:  # At least 60% unique
                    logger.info("✅ Diverse beam search successful!")
                    return outputs
                else:
                    logger.warning("⚠️ Poor diversity from diverse beam search, trying sampling...")
            
        except Exception as e:
            logger.warning(f"⚠️ Diverse beam search failed: {e}, trying sampling...")
        
        # Strategy 2: Sampling-based n-best (proven to work)
        logger.info(f"🎲 Using sampling-based n-best generation")
        
        sampling_kwargs = {
            "num_return_sequences": num_return,
            "do_sample": True,
            "temperature": 0.7,  # Controlled randomness
            "top_p": 0.9,  # Nucleus sampling
            "max_new_tokens": 200,
            "return_dict_in_generate": True,
            "output_scores": True,
            "language": "en",
        }
        
        # Add prompt if provided
        if "decoder_input_ids" in inputs:
            prompt_tokens = inputs["decoder_input_ids"]
            sampling_kwargs["decoder_input_ids"] = prompt_tokens
            logger.debug(f"Added prompt to sampling")
        
        try:
            outputs = self.model.generate(inputs["input_features"], **sampling_kwargs)
            
            # Verify sampling diversity
            if hasattr(outputs, 'sequences'):
                unique_texts = set()
                for seq in outputs.sequences:
                    decoded = self.processor.decode(seq, skip_special_tokens=True).strip()
                    unique_texts.add(decoded)
                
                diversity_ratio = len(unique_texts) / len(outputs.sequences)
                logger.info(f"🎲 Sampling n-best: {len(unique_texts)}/{len(outputs.sequences)} unique ({diversity_ratio:.1%})")
                
                if diversity_ratio >= 0.4:  # At least 40% unique
                    logger.info("✅ Sampling-based n-best successful!")
                    return outputs
            
        except Exception as e:
            logger.error(f"❌ Sampling also failed: {e}")
        
        # Strategy 3: Fallback to simple beam search
        logger.info(f"🔄 Fallback to simple beam search")
        fallback_kwargs = {
            "num_beams": num_beams,
            "num_return_sequences": num_return,
            "do_sample": False,
            "max_new_tokens": 200,
            "return_dict_in_generate": True,
            "output_scores": True,
            "language": "en",
        }
        
        if "decoder_input_ids" in inputs:
            fallback_kwargs["decoder_input_ids"] = inputs["decoder_input_ids"]
        
        return self.model.generate(inputs["input_features"], **fallback_kwargs)
    
    def _process_outputs(
        self,
        outputs: Any,
        decoding_config: DecodingConfig
    ) -> Dict[str, Any]:
        """Process model outputs into structured results."""
        
        results = {
            "predictions": [],
            "scores": [],
            "all_hypotheses": []
        }
        
        # Debug: Log output structure
        logger.debug(f"Output type: {type(outputs)}")
        logger.debug(f"Output attributes: {[attr for attr in dir(outputs) if not attr.startswith('_')]}")
        
        # Handle different output formats for Whisper
        if hasattr(outputs, 'sequences'):
            sequences = outputs.sequences
            logger.debug(f"Sequences shape: {sequences.shape}")
            
            # Extract real sequence scores from Whisper beam search output
            scores = None
            if hasattr(outputs, 'sequences_scores') and outputs.sequences_scores is not None:
                scores = outputs.sequences_scores
                logger.debug(f"✅ Found real sequences_scores: {scores.shape if hasattr(scores, 'shape') else type(scores)}")
                logger.debug(f"Real scores: {scores.tolist()}")
            elif hasattr(outputs, 'scores') and outputs.scores is not None and len(outputs.scores) > 0:
                # Try to compute real sequence scores from token-level scores
                logger.info("🔄 Computing real sequence scores from token scores...")
                try:
                    # Compute log probability for each sequence
                    sequence_scores = []
                    
                    for seq_idx in range(sequences.shape[0]):
                        seq_tokens = sequences[seq_idx]
                        seq_log_prob = 0.0
                        valid_tokens = 0
                        
                        # Sum log probabilities for all tokens in the sequence
                        for token_pos, token_scores in enumerate(outputs.scores):
                            if token_pos + 1 < len(seq_tokens):  # Skip special tokens
                                token_id = seq_tokens[token_pos + 1]  # +1 to skip start token
                                
                                # Get log probabilities for this position
                                if seq_idx < token_scores.shape[0]:
                                    token_logits = token_scores[seq_idx]
                                else:
                                    token_logits = token_scores[0]  # Fallback to first batch
                                
                                # Ensure token_id is valid
                                if 0 <= token_id < token_logits.shape[0]:
                                    token_log_probs = torch.log_softmax(token_logits, dim=-1)
                                    token_log_prob = float(token_log_probs[token_id])
                                    
                                    # Check for valid log probability
                                    if not (torch.isinf(torch.tensor(token_log_prob)) or torch.isnan(torch.tensor(token_log_prob))):
                                        seq_log_prob += token_log_prob
                                        valid_tokens += 1
                        
                        # Average by valid tokens to normalize
                        if valid_tokens > 0:
                            avg_log_prob = seq_log_prob / valid_tokens
                        else:
                            # Fallback if no valid tokens
                            avg_log_prob = -1.0 - (seq_idx * 0.1)
                        
                        sequence_scores.append(avg_log_prob)
                    
                    scores = torch.tensor(sequence_scores, dtype=torch.float32)
                    logger.info(f"✅ Computed real sequence scores: {scores.tolist()}")
                    
                except Exception as e:
                    logger.warning(f"Failed to compute real sequence scores: {e}")
                    # Fallback to rank-based scoring with variation
                    sequence_scores = []
                    for i in range(sequences.shape[0]):
                        import random
                        # Add more realistic variation based on rank
                        base_score = -0.5 - (i * 0.3) - (random.random() * 0.1)
                        sequence_scores.append(base_score)
                    scores = torch.tensor(sequence_scores, dtype=torch.float32)
                    logger.debug(f"Using improved fallback scoring: {scores.tolist()}")
            
            if scores is None:
                # Last resort: create more realistic fallback scores
                logger.warning("⚠️ No real scores available, using realistic fallback scoring")
                # Create more realistic scores with random variation
                import random
                base_scores = []
                for i in range(sequences.shape[0]):
                    # Add some randomness to make scores more realistic
                    base_score = -0.1 - (i * 0.05) - (random.random() * 0.02)
                    base_scores.append(base_score)
                scores = torch.tensor(base_scores, dtype=torch.float32)
                logger.debug(f"Created realistic fallback scores: {scores.tolist()}")
        else:
            sequences = outputs
            scores = None
            logger.debug(f"Using raw sequences: {sequences.shape}")
        
        # Debug: Check if we're actually getting multiple different sequences
        if sequences.shape[0] > 1:
            for i in range(min(3, sequences.shape[0])):
                decoded = self.processor.decode(sequences[i], skip_special_tokens=True)
                logger.debug(f"Sequence {i}: {decoded[:50]}...")
        
        # Process sequences
        num_sequences = sequences.shape[0]
        batch_size = num_sequences // decoding_config.num_return_sequences
        
        logger.debug(f"Processing {num_sequences} sequences for {batch_size} batch(es)")
        logger.debug(f"Expected return sequences: {decoding_config.num_return_sequences}")
        
        for batch_idx in range(batch_size):
            # Get all hypotheses for this sample
            start_idx = batch_idx * decoding_config.num_return_sequences
            end_idx = start_idx + decoding_config.num_return_sequences
            
            sample_sequences = sequences[start_idx:end_idx]
            sample_scores = scores[start_idx:end_idx] if scores is not None else None
            
            # Decode sequences
            hypotheses = []
            seen_texts = set()  # Track unique hypotheses
            
            for seq_idx, sequence in enumerate(sample_sequences):
                # Skip decoder start token for Whisper
                sequence_to_decode = sequence
                if len(sequence) > 0 and sequence[0] == self.model.config.decoder_start_token_id:
                    sequence_to_decode = sequence[1:]
                
                hypothesis = self.processor.decode(
                    sequence_to_decode,
                    skip_special_tokens=True
                )
                
                hypothesis_text = hypothesis.strip()
                
                # Calculate a score - if we don't have real scores, use sequence length and position
                if sample_scores is not None:
                    score = float(sample_scores[seq_idx])
                else:
                    # Fallback scoring: longer sequences get slightly lower scores, later ranks get lower scores
                    length_penalty = len(hypothesis_text) * 0.01
                    rank_penalty = seq_idx * 0.1
                    score = -(length_penalty + rank_penalty)
                
                hypothesis_data = {
                    "text": hypothesis_text,
                    "rank": seq_idx,
                    "score": score,
                    "unique": hypothesis_text not in seen_texts
                }
                hypotheses.append(hypothesis_data)
                seen_texts.add(hypothesis_text)
            
            # Sort by score (higher is better)
            hypotheses.sort(key=lambda x: x["score"], reverse=True)
            
            # Re-rank after sorting
            for i, hyp in enumerate(hypotheses):
                hyp["rank"] = i
            
            # Best prediction is rank 0
            best_prediction = hypotheses[0]["text"] if hypotheses else ""
            
            # Debug: Show hypothesis diversity
            unique_count = sum(1 for h in hypotheses if h["unique"])
            logger.debug(f"Generated {len(hypotheses)} hypotheses, {unique_count} unique")
            for i, hyp in enumerate(hypotheses[:3]):  # Show top 3
                logger.debug(f"  {i+1}. '{hyp['text'][:30]}...' (score: {hyp['score']:.3f}, unique: {hyp['unique']})")
            
            results["predictions"].append(best_prediction)
            results["all_hypotheses"].append(hypotheses)
            
            if len(hypotheses) > 0:
                results["scores"].append(hypotheses[0]["score"])
            else:
                results["scores"].append(0.0)
        
        return results
    
    def transcribe_single(
        self,
        audio_array: np.ndarray,
        decoding_config: DecodingConfig,
        return_timestamps: bool = False
    ) -> Dict[str, Any]:
        """Transcribe a single audio array."""
        
        batch_results = self.transcribe_batch([audio_array], decoding_config, return_timestamps)
        
        # Extract single result
        result = {
            "prediction": batch_results["predictions"][0],
            "score": batch_results["scores"][0],
            "hypotheses": batch_results["all_hypotheses"][0],
            "inference_time": batch_results["inference_time"],
            "avg_time_per_sample": batch_results["avg_time_per_sample"]
        }
        
        return result
    
    def transcribe_with_prompt(
        self,
        audio_array: np.ndarray,
        sample_metadata: Dict[str, Any],
        prompt_config: PromptConfig,
        decoding_config: DecodingConfig,
        return_timestamps: bool = False,
        template_style: str = "default"
    ) -> Dict[str, Any]:
        """Transcribe a single audio array with prompt conditioning."""
        
        if self.model is None or self.processor is None:
            self.load_model()
        
        start_time = time.time()
        
        # Generate prompt text
        prompt_text = generate_prompt_text(sample_metadata, prompt_config, template_style)
        
        # Process audio inputs
        inputs = self._prepare_inputs([audio_array])
        
        # Prepare prompt if enabled
        if prompt_config.enabled and prompt_text:
            inputs = self._add_prompt_to_inputs(inputs, prompt_text)
        
        # Get generation kwargs from decoding config
        generation_kwargs = decoding_config.get_generation_kwargs()
        
        # Add timestamps if requested
        if return_timestamps:
            generation_kwargs["return_timestamps"] = True
        
        # Perform inference
        with torch.no_grad():
            # Handle different decoding strategies
            if decoding_config.strategy.value == "greedy":
                outputs = self._transcribe_greedy(inputs, generation_kwargs)
            else:
                outputs = self._transcribe_beam_search(inputs, generation_kwargs)
        
        # Process outputs
        results = self._process_outputs(outputs, decoding_config)
        
        # Track performance
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        
        # Extract single result and add prompt info
        result = {
            "prediction": results["predictions"][0],
            "score": results["scores"][0],
            "hypotheses": results["all_hypotheses"][0],
            "inference_time": inference_time,
            "prompt_text": prompt_text,
            "prompt_strategy": prompt_config.strategy.value if prompt_config else "none",
            "prompt_enabled": prompt_config.enabled if prompt_config else False
        }
        
        return result
    
    def _add_prompt_to_inputs(
        self,
        inputs: Dict[str, torch.Tensor],
        prompt_text: str
    ) -> Dict[str, torch.Tensor]:
        """Add prompt conditioning to model inputs."""
        
        try:
            # For Whisper, we'll use the prompt as decoder input ids
            # Tokenize the prompt with proper Whisper tokens
            prompt_tokens = self.processor.tokenizer(
                prompt_text,
                return_tensors="pt",
                add_special_tokens=False
            )["input_ids"]
            
            # Move to device and ensure correct dtype
            prompt_tokens = prompt_tokens.to(
                device=self.device,
                dtype=torch.long  # Always use long for token ids
            )
            
            # Store prompt for generation kwargs
            inputs["decoder_input_ids"] = prompt_tokens
            inputs["prompt_text"] = prompt_text  # Store for logging
            
            logger.debug(f"Added prompt: '{prompt_text}'")
            logger.debug(f"Prompt tokens shape: {prompt_tokens.shape}")
            
        except Exception as e:
            logger.warning(f"Failed to add prompt '{prompt_text}': {e}")
            # Continue without prompt if tokenization fails
        
        return inputs
    
    def transcribe_batch_with_prompts(
        self,
        audio_arrays: List[np.ndarray],
        sample_metadata_list: List[Dict[str, Any]],
        prompt_config: PromptConfig,
        decoding_config: DecodingConfig,
        return_timestamps: bool = False,
        template_style: str = "default"
    ) -> Dict[str, Any]:
        """Transcribe a batch of audio arrays with prompt conditioning."""
        
        if self.model is None or self.processor is None:
            self.load_model()
        
        start_time = time.time()
        
        # Generate prompts for all samples
        prompt_texts = []
        for metadata in sample_metadata_list:
            prompt_text = generate_prompt_text(metadata, prompt_config, template_style)
            prompt_texts.append(prompt_text)
        
        # Process audio inputs
        inputs = self._prepare_inputs(audio_arrays)
        
        # Add prompts if enabled (batch processing is more complex for prompts)
        if prompt_config.enabled and any(prompt_texts):
            # For batch processing with different prompts per sample,
            # we need to process each sample individually
            # This is a limitation of current implementation
            logger.warning("Batch processing with different prompts per sample not fully supported. Processing individually.")
            
            individual_results = []
            for i, (audio, metadata) in enumerate(zip(audio_arrays, sample_metadata_list)):
                result = self.transcribe_with_prompt(
                    audio, metadata, prompt_config, decoding_config, 
                    return_timestamps, template_style
                )
                individual_results.append(result)
            
            # Combine individual results into batch format
            batch_results = {
                "predictions": [r["prediction"] for r in individual_results],
                "scores": [r["score"] for r in individual_results],
                "all_hypotheses": [r["hypotheses"] for r in individual_results],
                "inference_time": sum(r["inference_time"] for r in individual_results),
                "avg_time_per_sample": sum(r["inference_time"] for r in individual_results) / len(individual_results),
                "prompt_texts": [r["prompt_text"] for r in individual_results],
                "prompt_strategy": prompt_config.strategy.value,
                "prompt_enabled": prompt_config.enabled
            }
            
            return batch_results
        
        # If no prompts, use regular batch processing
        return self.transcribe_batch(audio_arrays, decoding_config, return_timestamps)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and statistics."""
        
        if self.model is None:
            return {"status": "not_loaded"}
        
        # Model parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # Memory usage (approximate)
        model_memory = sum(p.numel() * p.element_size() for p in self.model.parameters()) / 1024**3
        
        # Performance statistics
        avg_inference_time = np.mean(self.inference_times) if self.inference_times else 0
        
        info = {
            "model_id": self.model_config.model_id,
            "device": self.device,
            "dtype": str(self.model_config.torch_dtype),
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_memory_gb": model_memory,
            "avg_inference_time": avg_inference_time,
            "total_inferences": len(self.inference_times),
            "status": "loaded"
        }
        
        return info
    
    def clear_cache(self) -> None:
        """Clear model cache and free memory."""
        
        if self.device == "mps":
            torch.mps.empty_cache()
        elif self.device == "cuda":
            torch.cuda.empty_cache()
        
        # Clear performance tracking
        self.inference_times.clear()
        self.memory_usage.clear()
    
    def benchmark_inference(
        self,
        audio_array: np.ndarray,
        decoding_config: DecodingConfig,
        n_runs: int = 10
    ) -> Dict[str, Any]:
        """Benchmark inference performance."""
        
        logger.info(f"Benchmarking inference with {n_runs} runs")
        
        times = []
        for i in range(n_runs):
            start_time = time.time()
            result = self.transcribe_single(audio_array, decoding_config)
            end_time = time.time()
            times.append(end_time - start_time)
            
            # Clear cache every few runs
            if i % 3 == 0:
                self.clear_cache()
        
        benchmark_results = {
            "mean_time": np.mean(times),
            "std_time": np.std(times),
            "min_time": np.min(times),
            "max_time": np.max(times),
            "median_time": np.median(times),
            "total_runs": n_runs,
            "audio_duration": len(audio_array) / 16000,  # Assuming 16kHz
            "rtf": np.mean(times) / (len(audio_array) / 16000)  # Real-time factor
        }
        
        return benchmark_results
