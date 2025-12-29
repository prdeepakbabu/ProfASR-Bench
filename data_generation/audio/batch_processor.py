"""
Batch TTS Processor for Synthetic ASR Dataset
Processes JSON files containing utterances and generates audio files
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Optional
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from config.tts_config import TTSConfig
from scripts.kokoro_tts_generator import KokoroTTSGenerator

class BatchTTSProcessor:
    """Batch processor for converting utterance datasets to audio"""
    
    def __init__(self, config: TTSConfig = None):
        """Initialize batch processor
        
        Args:
            config: TTSConfig instance
        """
        self.config = config or TTSConfig()
        self.generator = KokoroTTSGenerator(self.config)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Statistics
        self.stats = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'start_time': None,
            'end_time': None
        }
    
    def load_utterances(self, json_file: Path) -> Optional[List[Dict]]:
        """Load utterances from JSON file
        
        Args:
            json_file: Path to JSON file containing utterances
            
        Returns:
            List of utterance dictionaries or None if failed
        """
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle different JSON structures
            if isinstance(data, dict) and 'utterances' in data:
                utterances = data['utterances']
                self.logger.info(f"Loaded {len(utterances)} utterances from {json_file}")
                return utterances
            elif isinstance(data, list):
                self.logger.info(f"Loaded {len(data)} utterances from {json_file}")
                return data
            else:
                self.logger.error(f"Unexpected JSON structure in {json_file}")
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to load utterances from {json_file}: {str(e)}")
            return None
    
    def process_single_utterance(self, utterance_data: Dict, output_dir: Path) -> bool:
        """Process a single utterance and generate audio
        
        Args:
            utterance_data: Dictionary containing utterance information
            output_dir: Directory to save the audio file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            result = self.generator.generate_from_utterance(utterance_data, output_dir)
            if result:
                self.stats['successful'] += 1
                return True
            else:
                self.stats['failed'] += 1
                return False
                
        except Exception as e:
            self.logger.error(f"Error processing utterance {utterance_data.get('id', 'unknown')}: {str(e)}")
            self.stats['failed'] += 1
            return False
    
    def process_domain(self, domain: str, max_utterances: Optional[int] = None) -> bool:
        """Process all utterances for a specific domain
        
        Args:
            domain: Domain name (medical, financial, legal, technical)
            max_utterances: Maximum number of utterances to process (None for all)
            
        Returns:
            bool: True if processing completed, False if failed
        """
        try:
            # Get input file path
            input_file = self.config.get_input_file_path(domain)
            if not input_file.exists():
                self.logger.error(f"Input file not found: {input_file}")
                return False
            
            # Load utterances
            utterances = self.load_utterances(input_file)
            if not utterances:
                return False
            
            # Limit utterances if specified
            if max_utterances and max_utterances < len(utterances):
                utterances = utterances[:max_utterances]
                self.logger.info(f"Processing first {max_utterances} utterances")
            
            # Get output directory
            output_dir = self.config.get_output_dir(domain)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Load model
            if not self.generator.load_model():
                self.logger.error("Failed to load TTS model")
                return False
            
            # Process utterances
            self.logger.info(f"Processing {len(utterances)} {domain} utterances...")
            self.stats['total_processed'] = len(utterances)
            self.stats['start_time'] = time.time()
            
            # Progress bar
            with tqdm(total=len(utterances), desc=f"Processing {domain}") as pbar:
                for utterance in utterances:
                    success = self.process_single_utterance(utterance, output_dir)
                    pbar.update(1)
                    pbar.set_postfix({
                        'Success': self.stats['successful'],
                        'Failed': self.stats['failed']
                    })
            
            self.stats['end_time'] = time.time()
            self._print_statistics(domain)
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to process domain {domain}: {str(e)}")
            return False
    
    def process_all_domains(self, max_per_domain: Optional[int] = None) -> Dict[str, bool]:
        """Process all available domains
        
        Args:
            max_per_domain: Maximum utterances per domain (None for all)
            
        Returns:
            Dictionary with domain names as keys and success status as values
        """
        results = {}
        
        for domain in self.config.DOMAIN_DIRS.keys():
            self.logger.info(f"Starting processing for domain: {domain}")
            
            # Reset stats for each domain
            self.stats = {
                'total_processed': 0,
                'successful': 0,
                'failed': 0,
                'start_time': None,
                'end_time': None
            }
            
            results[domain] = self.process_domain(domain, max_per_domain)
            
            if results[domain]:
                self.logger.info(f"✅ Successfully completed {domain} domain")
            else:
                self.logger.error(f"❌ Failed to process {domain} domain")
        
        return results
    
    def _print_statistics(self, domain: str):
        """Print processing statistics
        
        Args:
            domain: Domain name that was processed
        """
        duration = self.stats['end_time'] - self.stats['start_time']
        success_rate = (self.stats['successful'] / self.stats['total_processed']) * 100
        
        print(f"\n📊 {domain.title()} Domain Processing Statistics:")
        print(f"   Total utterances: {self.stats['total_processed']}")
        print(f"   Successful: {self.stats['successful']}")
        print(f"   Failed: {self.stats['failed']}")
        print(f"   Success rate: {success_rate:.1f}%")
        print(f"   Processing time: {duration:.2f} seconds")
        print(f"   Average time per utterance: {duration/self.stats['total_processed']:.2f}s")
        
        # Output directory info
        output_dir = self.config.get_output_dir(domain)
        audio_files = list(output_dir.glob(f"*.{self.config.AUDIO_FORMAT}"))
        metadata_files = list(output_dir.glob("*.json"))
        
        print(f"   Audio files generated: {len(audio_files)}")
        print(f"   Metadata files generated: {len(metadata_files)}")
        print(f"   Output directory: {output_dir}")

def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Batch TTS processor for synthetic ASR dataset")
    parser.add_argument(
        '--domain', 
        choices=['medical', 'financial', 'legal', 'technical', 'all'],
        default='all',
        help='Domain to process (default: all)'
    )
    parser.add_argument(
        '--max-utterances',
        type=int,
        help='Maximum number of utterances to process per domain'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run in test mode (process only 5 utterances per domain)'
    )
    
    args = parser.parse_args()
    
    # Create processor
    processor = BatchTTSProcessor()
    
    # Set test limits
    max_utterances = args.max_utterances
    if args.test:
        max_utterances = 5
        print("🧪 Running in test mode - processing 5 utterances per domain")
    
    # Process domains
    if args.domain == 'all':
        print("🎵 Starting batch TTS processing for all domains...")
        results = processor.process_all_domains(max_utterances)
        
        print("\n📋 Final Results:")
        for domain, success in results.items():
            status = "✅ Success" if success else "❌ Failed"
            print(f"   {domain.title()}: {status}")
            
    else:
        print(f"🎵 Starting batch TTS processing for {args.domain} domain...")
        success = processor.process_domain(args.domain, max_utterances)
        
        if success:
            print(f"✅ Successfully processed {args.domain} domain")
        else:
            print(f"❌ Failed to process {args.domain} domain")

if __name__ == "__main__":
    main()
