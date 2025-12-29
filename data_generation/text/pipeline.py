"""
Main generation pipeline for orchestrating memory bank utterance generation.
Coordinates all components for batch processing with rate limiting and progress tracking.
"""

import time
from typing import Dict, List, Any, Optional
from datetime import datetime

from bedrock_claude.client import BedrockClaudeClient
from .domains import get_domain_names, get_domain
from .profile_generator import ProfileGenerator, profile_generator
from .utterance_generator import UtteranceGenerator
from .storage import MemoryBankStorage, memory_bank_storage

class MemoryBankPipeline:
    """Main pipeline for generating and storing utterances across all domains."""
    
    def __init__(self, 
                 claude_client: BedrockClaudeClient = None,
                 domains: List[str] = None,
                 utterances_per_domain: int = 500,
                 delay_seconds: int = 10):
        """
        Initialize the memory bank pipeline.
        
        Args:
            claude_client: Bedrock Claude client instance
            domains: List of domains to generate for. If None, uses all domains.
            utterances_per_domain: Number of utterances per domain
            delay_seconds: Delay between API calls for rate limiting
        """
        self.claude_client = claude_client or BedrockClaudeClient()
        self.domains = domains or get_domain_names()
        self.utterances_per_domain = utterances_per_domain
        self.delay_seconds = delay_seconds
        
        # Initialize components
        self.profile_generator = profile_generator
        self.utterance_generator = UtteranceGenerator(self.claude_client)
        self.storage = memory_bank_storage
        
        # Pipeline statistics
        self.pipeline_stats = {
            'start_time': None,
            'end_time': None,
            'total_time': 0.0,
            'domains_completed': [],
            'domains_failed': [],
            'total_utterances_generated': 0,
            'total_api_calls': 0,
            'avg_generation_time': 0.0,
            'errors': []
        }
    
    def generate_all(self, save_batches: bool = True, batch_size: int = 50) -> Dict[str, Any]:
        """
        Generate utterances for all domains.
        
        Args:
            save_batches: Whether to save intermediate batches
            batch_size: Size of batches for intermediate saves
            
        Returns:
            Dictionary with generation results and statistics
        """
        self.pipeline_stats['start_time'] = datetime.utcnow()
        print(f"Starting memory bank generation for {len(self.domains)} domains")
        print(f"Target: {self.utterances_per_domain} utterances per domain")
        print(f"Total target: {len(self.domains) * self.utterances_per_domain} utterances")
        print(f"Rate limiting: {self.delay_seconds} seconds between API calls")
        print("-" * 60)
        
        results = {}
        
        for domain in self.domains:
            try:
                print(f"\n🔄 Starting generation for {domain.upper()} domain...")
                domain_result = self.generate_domain_utterances(
                    domain, 
                    save_batches=save_batches, 
                    batch_size=batch_size
                )
                results[domain] = domain_result
                self.pipeline_stats['domains_completed'].append(domain)
                
                print(f"✅ Completed {domain} domain: {len(domain_result['utterances'])} utterances")
                
            except Exception as e:
                error_msg = f"Failed to generate utterances for {domain}: {str(e)}"
                print(f"❌ {error_msg}")
                self.pipeline_stats['errors'].append(error_msg)
                self.pipeline_stats['domains_failed'].append(domain)
                results[domain] = {'utterances': [], 'error': str(e)}
        
        # Finalize pipeline statistics
        self.pipeline_stats['end_time'] = datetime.utcnow()
        self.pipeline_stats['total_time'] = (
            self.pipeline_stats['end_time'] - self.pipeline_stats['start_time']
        ).total_seconds()
        
        # Calculate final statistics
        self._calculate_final_stats(results)
        
        # Print summary
        self._print_generation_summary(results)
        
        return {
            'results': results,
            'statistics': self.pipeline_stats,
            'generation_completed': True
        }
    
    def generate_domain_utterances(self, 
                                 domain: str, 
                                 save_batches: bool = True,
                                 batch_size: int = 50) -> Dict[str, Any]:
        """
        Generate utterances for a single domain.
        
        Args:
            domain: Domain name
            save_batches: Whether to save intermediate batches
            batch_size: Size of batches for intermediate saves
            
        Returns:
            Dictionary with domain generation results
        """
        domain_start_time = time.time()
        utterances = []
        batch_count = 0
        
        print(f"Generating {self.utterances_per_domain} utterances for {domain}...")
        
        for i in range(self.utterances_per_domain):
            try:
                # Generate utterance
                utterance = self.utterance_generator.generate_utterance(domain)
                utterances.append(utterance)
                self.pipeline_stats['total_api_calls'] += 1
                
                # Progress reporting
                if (i + 1) % 10 == 0:
                    progress_pct = ((i + 1) / self.utterances_per_domain) * 100
                    print(f"  Progress: {i + 1}/{self.utterances_per_domain} ({progress_pct:.1f}%)")
                    print(f"  Latest: {utterance['utterance'][:80]}...")
                
                # Save intermediate batches
                if save_batches and len(utterances) % batch_size == 0:
                    batch_utterances = utterances[-batch_size:]
                    batch_count += 1
                    
                    batch_metadata = {
                        'batch_number': batch_count,
                        'batch_size': len(batch_utterances),
                        'total_in_session': len(utterances),
                        'session_target': self.utterances_per_domain
                    }
                    
                    self.storage.save_utterances(domain, batch_utterances, batch_metadata)
                    print(f"  💾 Saved batch {batch_count} ({batch_size} utterances)")
                
                # Rate limiting - wait between API calls
                if i < self.utterances_per_domain - 1:  # Don't wait after last call
                    time.sleep(self.delay_seconds)
                    
            except Exception as e:
                error_msg = f"Error generating utterance {i + 1} for {domain}: {str(e)}"
                print(f"  ⚠️  {error_msg}")
                self.pipeline_stats['errors'].append(error_msg)
                continue
        
        # Save any remaining utterances
        if save_batches and len(utterances) % batch_size != 0:
            remaining_utterances = utterances[-(len(utterances) % batch_size):]
            batch_count += 1
            
            batch_metadata = {
                'batch_number': batch_count,
                'batch_size': len(remaining_utterances),
                'total_in_session': len(utterances),
                'session_target': self.utterances_per_domain,
                'final_batch': True
            }
            
            self.storage.save_utterances(domain, remaining_utterances, batch_metadata)
            print(f"  💾 Saved final batch {batch_count} ({len(remaining_utterances)} utterances)")
        
        domain_time = time.time() - domain_start_time
        self.pipeline_stats['total_utterances_generated'] += len(utterances)
        
        print(f"Domain {domain} completed in {domain_time:.1f} seconds")
        print(f"Generated {len(utterances)}/{self.utterances_per_domain} utterances")
        
        return {
            'utterances': utterances,
            'domain': domain,
            'generated_count': len(utterances),
            'target_count': self.utterances_per_domain,
            'generation_time': domain_time,
            'batch_count': batch_count,
            'success_rate': len(utterances) / self.utterances_per_domain
        }
    
    def resume_generation(self, domain: str, target_additional: int) -> Dict[str, Any]:
        """
        Resume generation for a domain that was interrupted.
        
        Args:
            domain: Domain name
            target_additional: Number of additional utterances to generate
            
        Returns:
            Dictionary with resume results
        """
        print(f"Resuming generation for {domain} domain...")
        print(f"Target additional utterances: {target_additional}")
        
        # Temporarily set utterances_per_domain for this session
        original_target = self.utterances_per_domain
        self.utterances_per_domain = target_additional
        
        try:
            result = self.generate_domain_utterances(domain)
            return result
        finally:
            # Restore original target
            self.utterances_per_domain = original_target
    
    def _calculate_final_stats(self, results: Dict[str, Any]):
        """Calculate final pipeline statistics."""
        total_generated = sum(
            len(domain_result.get('utterances', [])) 
            for domain_result in results.values()
        )
        
        total_target = len(self.domains) * self.utterances_per_domain
        
        if self.pipeline_stats['total_api_calls'] > 0:
            self.pipeline_stats['avg_generation_time'] = (
                self.pipeline_stats['total_time'] / self.pipeline_stats['total_api_calls']
            )
        
        self.pipeline_stats.update({
            'total_utterances_generated': total_generated,
            'total_target_utterances': total_target,
            'overall_success_rate': total_generated / total_target if total_target > 0 else 0.0,
            'domains_attempted': len(self.domains),
            'domains_succeeded': len(self.pipeline_stats['domains_completed']),
            'domains_failed_count': len(self.pipeline_stats['domains_failed']),
            'total_errors': len(self.pipeline_stats['errors'])
        })
    
    def _print_generation_summary(self, results: Dict[str, Any]):
        """Print a summary of generation results."""
        print("\n" + "=" * 60)
        print("🎯 MEMORY BANK GENERATION SUMMARY")
        print("=" * 60)
        
        # Overall stats
        stats = self.pipeline_stats
        print(f"Total Time: {stats['total_time']:.1f} seconds ({stats['total_time']/60:.1f} minutes)")
        print(f"Total API Calls: {stats['total_api_calls']}")
        print(f"Average Time per Call: {stats['avg_generation_time']:.2f} seconds")
        print(f"Generated: {stats['total_utterances_generated']}/{stats['total_target_utterances']} utterances")
        print(f"Success Rate: {stats['overall_success_rate']:.1%}")
        
        # Per-domain breakdown
        print(f"\n📊 DOMAIN BREAKDOWN:")
        for domain in self.domains:
            domain_result = results.get(domain, {})
            utterance_count = len(domain_result.get('utterances', []))
            success_rate = utterance_count / self.utterances_per_domain
            status = "✅" if success_rate >= 0.9 else "⚠️" if success_rate >= 0.5 else "❌"
            
            print(f"  {status} {domain.capitalize()}: {utterance_count}/{self.utterances_per_domain} ({success_rate:.1%})")
        
        # Errors
        if stats['errors']:
            print(f"\n⚠️  ERRORS ENCOUNTERED ({len(stats['errors'])}):")
            for error in stats['errors'][-5:]:  # Show last 5 errors
                print(f"  - {error}")
            if len(stats['errors']) > 5:
                print(f"  ... and {len(stats['errors']) - 5} more errors")
        
        print("\n🎉 Generation pipeline completed!")
        print("Use create_master_dataset() to combine all domain data.")
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get current pipeline statistics."""
        return self.pipeline_stats.copy()
    
    def create_master_dataset(self, output_filename: str = None) -> str:
        """Create master dataset from all generated utterances."""
        print("Creating master dataset from all generated utterances...")
        return self.storage.create_master_dataset(output_filename)

# Convenience functions
def run_full_generation(utterances_per_domain: int = 500, 
                       delay_seconds: int = 10,
                       domains: List[str] = None) -> Dict[str, Any]:
    """
    Convenience function to run full generation pipeline.
    
    Args:
        utterances_per_domain: Number of utterances per domain
        delay_seconds: Delay between API calls
        domains: List of domains. If None, uses all domains.
        
    Returns:
        Generation results dictionary
    """
    pipeline = MemoryBankPipeline(
        utterances_per_domain=utterances_per_domain,
        delay_seconds=delay_seconds,
        domains=domains
    )
    
    return pipeline.generate_all()

def generate_sample_batch(domain: str, count: int = 10) -> List[Dict[str, Any]]:
    """
    Generate a small sample batch for testing.
    
    Args:
        domain: Domain name
        count: Number of utterances to generate
        
    Returns:
        List of generated utterances
    """
    pipeline = MemoryBankPipeline(utterances_per_domain=count, delay_seconds=1)
    result = pipeline.generate_domain_utterances(domain, save_batches=False)
    return result['utterances']
