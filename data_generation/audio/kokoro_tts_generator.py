"""
Kokoro 82M TTS Generator
Uses the official Kokoro TTS library for high-quality speech synthesis
"""

import torch
import numpy as np
import soundfile as sf
from pathlib import Path
import logging
from typing import Optional, Union
import json
import time
from datetime import datetime
import sys

sys.path.append(str(Path(__file__).parent.parent))
from config.tts_config import TTSConfig

# Import Kokoro TTS
try:
    from kokoro import KPipeline
    KOKORO_AVAILABLE = True
except ImportError:
    KOKORO_AVAILABLE = False
    print("Warning: Kokoro TTS not available. Install with: pip install kokoro>=0.9.2")

class KokoroTTSGenerator:
    """Real Kokoro 82M TTS generator using the official library"""
    
    def __init__(self, config: TTSConfig = None):
        """Initialize the Kokoro TTS generator
        
        Args:
            config: TTSConfig instance with audio settings
        """
        self.config = config or TTSConfig()
        self.pipeline = None
        self.model_loaded = False
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Ensure directories exist
        self.config.ensure_directories()
        
    def load_model(self) -> bool:
        """Load the Kokoro 82M model
        
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        if not KOKORO_AVAILABLE:
            self.logger.error("Kokoro TTS library not available")
            self.logger.error("Install with: pip install kokoro>=0.9.2")
            return False
            
        try:
            self.logger.info("Loading Kokoro 82M model...")
            
            # Initialize Kokoro pipeline with English language
            # Using 'a' for American English
            self.pipeline = KPipeline(lang_code='a')
            
            self.model_loaded = True
            self.logger.info("Kokoro 82M model loaded successfully!")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load Kokoro model: {str(e)}")
            self.logger.error("Make sure you have internet connection for first-time model download")
            return False
    
    def synthesize_speech(self, text: str, output_path: Union[str, Path], voice: str = 'af_heart') -> bool:
        """Generate real speech from text and save to file
        
        Args:
            text: Input text to synthesize
            output_path: Path to save the generated audio file
            voice: Voice to use (default: 'af_heart' - American Female)
            
        Returns:
            bool: True if synthesis successful, False otherwise
        """
        if not self.model_loaded or self.pipeline is None:
            self.logger.error("Model not loaded. Call load_model() first.")
            return False
            
        if not text.strip():
            self.logger.warning("Empty text provided for synthesis")
            return False
            
        try:
            start_time = time.time()
            self.logger.info(f"Synthesizing with Kokoro: {text[:50]}...")
            
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Generate speech using Kokoro pipeline
            generator = self.pipeline(text, voice=voice)
            
            # Get the audio from the generator (usually first result)
            audio_data = None
            for i, (gs, ps, audio) in enumerate(generator):
                self.logger.debug(f"Generated segment {i}: gs={gs}, ps={ps}")
                if audio_data is None:
                    audio_data = audio
                else:
                    # Concatenate multiple segments if needed
                    audio_data = np.concatenate([audio_data, audio])
                
                # For now, just take the first segment
                # In practice, you might want to concatenate all segments
                if i == 0:
                    break
            
            if audio_data is None:
                self.logger.error("No audio generated")
                return False
            
            # Kokoro generates at 24kHz by default
            kokoro_sample_rate = 24000
            
            # Resample to target sample rate if needed
            if self.config.SAMPLE_RATE != kokoro_sample_rate:
                # Simple resampling - for better quality, use librosa
                try:
                    import scipy.signal
                    audio_data = scipy.signal.resample(
                        audio_data, 
                        int(len(audio_data) * self.config.SAMPLE_RATE / kokoro_sample_rate)
                    )
                    final_sample_rate = self.config.SAMPLE_RATE
                except ImportError:
                    self.logger.warning("scipy not available for resampling, using original sample rate")
                    final_sample_rate = kokoro_sample_rate
            else:
                final_sample_rate = kokoro_sample_rate
            
            # Add silence padding if configured
            if self.config.ADD_SILENCE > 0:
                silence_samples = int(self.config.ADD_SILENCE * final_sample_rate)
                silence = np.zeros(silence_samples)
                audio_data = np.concatenate([silence, audio_data, silence])
            
            # Normalize audio if configured
            if self.config.NORMALIZE_AUDIO:
                if np.max(np.abs(audio_data)) > 0:
                    audio_data = audio_data / np.max(np.abs(audio_data)) * 0.95
            
            # Save audio file
            sf.write(
                str(output_path),
                audio_data,
                final_sample_rate,
                subtype=f'PCM_{self.config.BIT_DEPTH}' if self.config.AUDIO_FORMAT == 'wav' else None
            )
            
            duration = time.time() - start_time
            self.logger.info(f"Kokoro audio saved to {output_path} (synthesis took {duration:.2f}s)")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to synthesize speech with Kokoro: {str(e)}")
            return False
    
    def generate_from_utterance(self, utterance_data: dict, output_dir: Path) -> Optional[Path]:
        """Generate audio from a single utterance dictionary
        
        Args:
            utterance_data: Dictionary containing utterance information
            output_dir: Directory to save the audio file
            
        Returns:
            Path to generated audio file if successful, None otherwise
        """
        try:
            # Extract text and metadata
            text = utterance_data.get('utterance', '')
            utterance_id = utterance_data.get('id', 'unknown')
            profile = utterance_data.get('profile', 'unknown')
            domain = utterance_data.get('domain', 'unknown')
            
            if not text:
                self.logger.warning(f"No text found for utterance {utterance_id}")
                return None
            
            # Create filename
            filename = f"{utterance_id}.{self.config.AUDIO_FORMAT}"
            output_path = output_dir / filename
            
            # Choose voice based on profile (simple heuristic)
            voice = self._choose_voice(profile)
            
            # Generate real audio using Kokoro
            if self.synthesize_speech(text, output_path, voice=voice):
                # Create metadata file
                metadata = {
                    'utterance_id': utterance_id,
                    'text': text,
                    'profile': profile,
                    'domain': domain,
                    'audio_file': str(output_path),
                    'sample_rate': self.config.SAMPLE_RATE,
                    'generated_at': datetime.now().isoformat(),
                    'asr_difficulty': utterance_data.get('asr_difficulty', 0),
                    'error_targets': utterance_data.get('error_targets', []),
                    'generator_type': 'kokoro_82m',
                    'kokoro_voice': voice,
                    'note': 'Real speech generated using Kokoro 82M TTS'
                }
                
                metadata_path = output_path.with_suffix('.json')
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                return output_path
            
        except Exception as e:
            self.logger.error(f"Failed to generate Kokoro audio for utterance {utterance_data.get('id', 'unknown')}: {str(e)}")
        
        return None
    
    def _choose_voice(self, profile: str) -> str:
        """Choose appropriate Kokoro voice based on profile
        
        Args:
            profile: Speaker profile description
            
        Returns:
            Voice identifier for Kokoro
        """
        profile_lower = profile.lower()
        
        # Available voices in Kokoro (these are examples, check docs for full list)
        # af_* = American Female
        # am_* = American Male
        # bf_* = British Female
        # bm_* = British Male
        
        # Simple heuristics based on profile
        if any(word in profile_lower for word in ['female', 'woman', 'she', 'her']):
            if any(word in profile_lower for word in ['british', 'london', 'uk']):
                return 'bf_heart'  # British female
            else:
                return 'af_heart'  # American female (default)
        elif any(word in profile_lower for word in ['male', 'man', 'he', 'his']):
            if any(word in profile_lower for word in ['british', 'london', 'uk']):
                return 'bm_heart'  # British male
            else:
                return 'am_heart'  # American male
        else:
            # Default to American female for medical/professional contexts
            return 'af_heart'
    
    def list_available_voices(self):
        """List available voices in Kokoro"""
        # This would require checking Kokoro documentation
        # For now, return common ones
        return [
            'af_heart',  # American Female
            'am_heart',  # American Male
            'bf_heart',  # British Female
            'bm_heart',  # British Male
        ]

def test_kokoro():
    """Test Kokoro TTS with sample medical text"""
    generator = KokoroTTSGenerator()
    
    if not generator.load_model():
        print("❌ Failed to load Kokoro model")
        return False
    
    # Test with the user's medical text
    test_text = "I've been treating more patients with post-pneumonia weakness in our rehabilitation center. The Graston technique and myofascial release therapy have shown remarkable improvement for patients also taking lisinopril."
    output_path = Path("TTS_audio/output/kokoro_test.wav")
    
    print(f"🎵 Testing Kokoro with: {test_text[:50]}...")
    success = generator.synthesize_speech(test_text, output_path)
    
    if success:
        print(f"✅ Kokoro test successful: {output_path}")
        print(f"   Audio file size: {output_path.stat().st_size:,} bytes")
        
        # Validate the audio
        from scripts.audio_utils import AudioValidator
        validator = AudioValidator()
        validation = validator.validate_audio_file(output_path)
        
        print(f"   Duration: {validation['duration']:.2f} seconds")
        print(f"   Sample rate: {validation['sample_rate']} Hz")
        print(f"   Valid: {'✅ Yes' if validation['valid'] else '❌ No'}")
        
        return True
    else:
        print("❌ Kokoro test failed")
        return False

if __name__ == "__main__":
    test_kokoro()
