"""
Audio utilities for TTS processing
Handles audio analysis, validation, and post-processing
"""

import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Tuple, Optional, Dict, List
import json
import logging

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

class AudioValidator:
    """Validates generated audio files for quality and correctness"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def validate_audio_file(self, audio_path: Path) -> Dict:
        """Validate an audio file and return quality metrics
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with validation results and metrics
        """
        validation_result = {
            'valid': False,
            'file_exists': False,
            'file_size': 0,
            'duration': 0.0,
            'sample_rate': 0,
            'channels': 0,
            'max_amplitude': 0.0,
            'rms_level': 0.0,
            'has_silence': False,
            'errors': []
        }
        
        try:
            # Check if file exists
            if not audio_path.exists():
                validation_result['errors'].append("Audio file does not exist")
                return validation_result
            
            validation_result['file_exists'] = True
            validation_result['file_size'] = audio_path.stat().st_size
            
            # Load and analyze audio
            audio_data, sample_rate = sf.read(str(audio_path))
            
            if len(audio_data) == 0:
                validation_result['errors'].append("Audio file is empty")
                return validation_result
            
            # Basic metrics
            validation_result['sample_rate'] = sample_rate
            validation_result['duration'] = len(audio_data) / sample_rate
            validation_result['channels'] = 1 if audio_data.ndim == 1 else audio_data.shape[1]
            
            # Audio quality metrics
            validation_result['max_amplitude'] = float(np.max(np.abs(audio_data)))
            validation_result['rms_level'] = float(np.sqrt(np.mean(audio_data ** 2)))
            
            # Check for silence (very low RMS)
            validation_result['has_silence'] = validation_result['rms_level'] < 0.001
            
            # Quality checks
            if validation_result['duration'] < 0.1:
                validation_result['errors'].append("Audio too short (< 0.1s)")
            
            if validation_result['max_amplitude'] > 0.99:
                validation_result['errors'].append("Audio may be clipped")
            
            if validation_result['rms_level'] < 0.001:
                validation_result['errors'].append("Audio appears to be silent")
            
            # Mark as valid if no errors
            validation_result['valid'] = len(validation_result['errors']) == 0
            
        except Exception as e:
            validation_result['errors'].append(f"Error validating audio: {str(e)}")
        
        return validation_result
    
    def validate_batch(self, audio_dir: Path, domain: str) -> Dict:
        """Validate all audio files in a directory
        
        Args:
            audio_dir: Directory containing audio files
            domain: Domain name for context
            
        Returns:
            Dictionary with batch validation results
        """
        results = {
            'domain': domain,
            'total_files': 0,
            'valid_files': 0,
            'invalid_files': 0,
            'total_duration': 0.0,
            'avg_duration': 0.0,
            'file_results': {},
            'errors': []
        }
        
        try:
            # Find all audio files
            audio_files = list(audio_dir.glob("*.wav")) + list(audio_dir.glob("*.mp3"))
            results['total_files'] = len(audio_files)
            
            if results['total_files'] == 0:
                results['errors'].append("No audio files found")
                return results
            
            # Validate each file
            durations = []
            for audio_file in audio_files:
                file_result = self.validate_audio_file(audio_file)
                results['file_results'][audio_file.name] = file_result
                
                if file_result['valid']:
                    results['valid_files'] += 1
                    durations.append(file_result['duration'])
                else:
                    results['invalid_files'] += 1
            
            # Calculate statistics
            if durations:
                results['total_duration'] = sum(durations)
                results['avg_duration'] = results['total_duration'] / len(durations)
            
        except Exception as e:
            results['errors'].append(f"Error in batch validation: {str(e)}")
        
        return results

class AudioPostProcessor:
    """Post-processes generated audio for consistency and quality"""
    
    def __init__(self, target_sample_rate: int = 22050):
        self.target_sample_rate = target_sample_rate
        self.logger = logging.getLogger(__name__)
    
    def normalize_audio(self, audio: np.ndarray, target_rms: float = 0.2) -> np.ndarray:
        """Normalize audio to target RMS level
        
        Args:
            audio: Input audio array
            target_rms: Target RMS level
            
        Returns:
            Normalized audio array
        """
        current_rms = np.sqrt(np.mean(audio ** 2))
        if current_rms > 0:
            scaling_factor = target_rms / current_rms
            return audio * scaling_factor
        return audio
    
    def add_fade(self, audio: np.ndarray, fade_duration: float = 0.1, 
                 sample_rate: int = 22050) -> np.ndarray:
        """Add fade in/out to audio
        
        Args:
            audio: Input audio array
            fade_duration: Fade duration in seconds
            sample_rate: Audio sample rate
            
        Returns:
            Audio with fade applied
        """
        fade_samples = int(fade_duration * sample_rate)
        fade_samples = min(fade_samples, len(audio) // 4)  # Max 25% of audio length
        
        if fade_samples <= 0:
            return audio
        
        # Create fade curves
        fade_in = np.linspace(0, 1, fade_samples)
        fade_out = np.linspace(1, 0, fade_samples)
        
        # Apply fades
        result = audio.copy()
        result[:fade_samples] *= fade_in
        result[-fade_samples:] *= fade_out
        
        return result
    
    def remove_silence(self, audio: np.ndarray, sample_rate: int = 22050, 
                      threshold: float = 0.01) -> np.ndarray:
        """Remove silence from beginning and end of audio
        
        Args:
            audio: Input audio array
            sample_rate: Audio sample rate
            threshold: Silence threshold (RMS)
            
        Returns:
            Audio with silence removed
        """
        if not LIBROSA_AVAILABLE:
            self.logger.warning("librosa not available, skipping silence removal")
            return audio
        
        # Find non-silent intervals
        intervals = librosa.effects.split(audio, top_db=20)
        
        if len(intervals) == 0:
            return audio
        
        # Extract audio between first and last non-silent interval
        start = intervals[0][0]
        end = intervals[-1][1]
        
        return audio[start:end]
    
    def process_audio_file(self, input_path: Path, output_path: Path = None,
                          normalize: bool = True, add_fades: bool = True,
                          remove_silence: bool = True) -> bool:
        """Process an audio file with various enhancements
        
        Args:
            input_path: Path to input audio file
            output_path: Path for output (defaults to overwriting input)
            normalize: Whether to normalize audio
            add_fades: Whether to add fade in/out
            remove_silence: Whether to remove silence
            
        Returns:
            bool: True if processing successful
        """
        try:
            # Load audio
            audio, sample_rate = sf.read(str(input_path))
            
            # Apply processing steps
            if remove_silence:
                audio = self.remove_silence(audio, sample_rate)
            
            if normalize:
                audio = self.normalize_audio(audio)
            
            if add_fades:
                audio = self.add_fade(audio, sample_rate=sample_rate)
            
            # Save processed audio
            output_path = output_path or input_path
            sf.write(str(output_path), audio, sample_rate)
            
            self.logger.info(f"Processed audio saved to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to process audio file {input_path}: {str(e)}")
            return False

def create_audio_report(validation_results: Dict, output_path: Path):
    """Create a detailed audio validation report
    
    Args:
        validation_results: Results from batch validation
        output_path: Path to save the report
    """
    report = {
        'validation_timestamp': str(np.datetime64('now')),
        'summary': {
            'domain': validation_results['domain'],
            'total_files': validation_results['total_files'],
            'valid_files': validation_results['valid_files'],
            'invalid_files': validation_results['invalid_files'],
            'success_rate': validation_results['valid_files'] / max(validation_results['total_files'], 1) * 100,
            'total_duration_minutes': validation_results['total_duration'] / 60,
            'avg_duration_seconds': validation_results['avg_duration']
        },
        'detailed_results': validation_results['file_results'],
        'errors': validation_results['errors']
    }
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)

def main():
    """Test audio utilities"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Audio utilities for TTS validation")
    parser.add_argument('--validate', type=str, help='Directory to validate')
    parser.add_argument('--domain', type=str, default='test', help='Domain name')
    
    args = parser.parse_args()
    
    if args.validate:
        validator = AudioValidator()
        results = validator.validate_batch(Path(args.validate), args.domain)
        
        print(f"📊 Audio Validation Results for {args.domain}:")
        print(f"   Total files: {results['total_files']}")
        print(f"   Valid files: {results['valid_files']}")
        print(f"   Invalid files: {results['invalid_files']}")
        print(f"   Success rate: {results['valid_files'] / max(results['total_files'], 1) * 100:.1f}%")
        print(f"   Total duration: {results['total_duration']:.2f}s")
        print(f"   Average duration: {results['avg_duration']:.2f}s")
        
        if results['errors']:
            print(f"   Errors: {results['errors']}")

if __name__ == "__main__":
    main()
