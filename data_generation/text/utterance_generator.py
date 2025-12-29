"""
Utterance generator using Claude 3.7 via Bedrock for ASR research.
Generates realistic two-sentence utterances with ASR error-prone entities.
"""

import re
import time
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime

from bedrock_claude.client import BedrockClaudeClient, BedrockConfig
from .domains import get_domain, Domain
from .profile_generator import ProfileGenerator, profile_generator
from .spoken_forms import apply_spoken_forms

class ASRDifficultyAnalyzer:
    """Analyzes utterances for ASR error potential."""
    
    def __init__(self):
        self.error_patterns = [
            # Homophones
            r'\b(there|their|they\'re)\b',
            r'\b(to|too|two)\b', 
            r'\b(your|you\'re)\b',
            r'\b(its|it\'s)\b',
            r'\b(site|sight|cite)\b',
            
            # Confusable numbers
            r'\b(fifteen|fifty)\b',
            r'\b(thirteen|thirty)\b', 
            r'\b(fourteen|forty)\b',
            r'\b(sixteen|sixty)\b',
            r'\b(seventeen|seventy)\b',
            r'\b(eighteen|eighty)\b',
            r'\b(nineteen|ninety)\b',
            
            # Similar sounds
            r'\b[MN]\b',  # M vs N
            r'\b[BPD]\b',  # B vs P vs D
            r'\b[SF]\b',   # S vs F
            
            # Technical terms
            r'\b[A-Z]\s+[A-Z]\b',  # Spelled out acronyms
            r'\b\w+\s+and\s+\w+\b',  # "M and M" patterns
            
            # Numbers in spoken form
            r'\b(zero|one|two|three|four|five|six|seven|eight|nine)\b',
            r'\b(ten|eleven|twelve|thirteen|fourteen|fifteen)\b',
            r'\b(twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)\b',
            r'\b(hundred|thousand|million|billion)\b',
            
            # Medical/Technical terms prone to errors
            r'\b(milligrams|milliliters|centimeters)\b',
            r'\b(acetaminophen|ibuprofen|prednisone)\b',
            r'\b(hypertension|pneumonia|arrhythmia)\b',
        ]
    
    def calculate_difficulty_score(self, utterance: str) -> float:
        """Calculate ASR difficulty score (0.0 to 1.0)."""
        total_words = len(utterance.split())
        if total_words == 0:
            return 0.0
        
        error_prone_count = 0
        for pattern in self.error_patterns:
            matches = re.findall(pattern, utterance, re.IGNORECASE)
            error_prone_count += len(matches)
        
        # Base score from error-prone word ratio
        base_score = min(error_prone_count / total_words, 0.8)
        
        # Bonus for specific challenging patterns
        bonus = 0.0
        if re.search(r'\b[A-Z]\s+[A-Z]\b', utterance):  # Spelled acronyms
            bonus += 0.1
        if re.search(r'\b\w+\s+and\s+\w+\b', utterance):  # "X and Y" patterns
            bonus += 0.1
        if re.search(r'\b(fifteen|fifty|thirteen|thirty)\b', utterance, re.IGNORECASE):
            bonus += 0.15
        
        final_score = min(base_score + bonus, 1.0)
        return round(final_score, 2)
    
    def identify_error_targets(self, utterance: str) -> List[str]:
        """Identify specific error-prone segments in the utterance."""
        targets = []
        
        for pattern in self.error_patterns:
            matches = re.findall(pattern, utterance, re.IGNORECASE)
            targets.extend(matches)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_targets = []
        for target in targets:
            if target.lower() not in seen:
                seen.add(target.lower())
                unique_targets.append(target)
        
        return unique_targets

class UtteranceGenerator:
    """Main utterance generator using Claude 3.7."""
    
    def __init__(self, claude_client: BedrockClaudeClient = None):
        self.claude_client = claude_client or BedrockClaudeClient()
        self.profile_generator = profile_generator
        self.difficulty_analyzer = ASRDifficultyAnalyzer()
        
        # Generation statistics
        self.generation_stats = {
            'total_generated': 0,
            'by_domain': {},
            'avg_difficulty': 0.0,
            'generation_time': 0.0
        }
    
    def create_generation_prompt(self, domain: str, profile: str) -> str:
        """Create the prompt for Claude to generate utterances."""
        
        domain_obj = get_domain(domain)
        context_hints = self.profile_generator.get_profile_context_hints(domain, profile)
        
        # Get domain-specific examples
        vocab_examples = ', '.join(domain_obj.get_random_vocabulary(3))
        error_examples = ', '.join(domain_obj.get_error_prone_terms(3))
        
        system_prompt = f"""You are generating realistic spoken utterances for ASR (Automatic Speech Recognition) research. Your goal is to create natural speech patterns that will be challenging for ASR systems to transcribe correctly.

Domain: {domain.title()}
User Profile: {profile}

CRITICAL REQUIREMENTS:
1. Generate exactly 2 consecutive sentences
2. First sentence: Provides professional context relevant to the {domain} domain
3. Second sentence: MUST contain technical terms, proper nouns that are commonly mispronounced or confused by ASR systems

Focus on natural speech patterns that this specific user profile would actually say in their professional environment.
Avoid number based or acronym based error patterns. Focus on phonetically sounding errors of words instead.
Keep the sentence length moderate. do not make long sentences.

ASR ERROR-PRONE ELEMENTS TO INCLUDE:
- Homophones 
- Brand names
- Professional jargon specific to {domain}

DOMAIN VOCABULARY: {vocab_examples}

Remember: The speech should sound natural and conversational, as if this person is actually speaking in their work environment."""

        context_hint_text = ""
        if context_hints:
            context_hint_text = f"\n\nCONTEXT HINTS: This person {', '.join(context_hints)}"

        user_prompt = f"""Generate a realistic spoken utterance for a {profile} working in the {domain} domain.

Requirements:
- Exactly 2 sentences
- First sentence: Professional context setup
- Second sentence: Include ASR error-prone elements (numbers, technical terms, homophones, etc.)
- Sound natural and conversational
- Use terminology this person would actually use

{context_hint_text}

Generate the utterance now:"""

        return system_prompt, user_prompt
    
    def generate_utterance(self, domain: str, profile: str = None, max_retries: int = 3) -> Dict[str, Any]:
        """
        Generate a single utterance for the specified domain and profile.
        
        Args:
            domain: Domain name (medical, legal, financial, technical)
            profile: User profile string. If None, generates random profile.
            max_retries: Maximum retries if generation fails validation
            
        Returns:
            Dictionary containing utterance data
        """
        start_time = time.time()
        
        # Generate profile if not provided
        if profile is None:
            profile = self.profile_generator.generate_profile(domain)
        
        # Create prompt
        system_prompt, user_prompt = self.create_generation_prompt(domain, profile)
        
        for attempt in range(max_retries + 1):
            try:
                # Generate with Claude
                response = self.claude_client.send_prompt(
                    prompt=user_prompt,
                    system_prompt=system_prompt,
                    #model_id="us.anthropic.claude-3-5-sonnet-20241022-v2:0",  # Claude 3.5 Sonnet v2 (latest available)
                    model_id="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
                    max_tokens=300,
                    temperature=0.78
                )
                
                raw_utterance = response.content.strip()
                
                # Validate utterance structure
                if not self._validate_utterance(raw_utterance):
                    if attempt < max_retries:
                        continue
                    else:
                        raise ValueError("Generated utterance failed validation after max retries")
                
                # Apply spoken form transformations
                spoken_utterance = apply_spoken_forms(raw_utterance, domain)
                
                # Calculate ASR difficulty
                difficulty_score = self.difficulty_analyzer.calculate_difficulty_score(spoken_utterance)
                error_targets = self.difficulty_analyzer.identify_error_targets(spoken_utterance)
                
                # Split into sentences
                sentences = self._split_sentences(spoken_utterance)
                
                # Generate unique ID
                utterance_id = f"{domain[:3]}_{int(time.time() * 1000) % 1000000:06d}"
                
                # Create result
                result = {
                    'id': utterance_id,
                    'profile': profile,
                    'utterance': spoken_utterance,
                    'raw_form': raw_utterance,
                    'asr_difficulty': difficulty_score,
                    'error_targets': error_targets,
                    'sentences': sentences,
                    'domain': domain,
                    'generation_timestamp': datetime.utcnow().isoformat() + 'Z',
                    'generation_time': round(time.time() - start_time, 2),
                    'model_version': 'claude-3.7-sonnet',
                    'attempt_number': attempt + 1
                }
                
                # Update statistics
                self._update_stats(domain, difficulty_score, time.time() - start_time)
                
                return result
                
            except Exception as e:
                if attempt < max_retries:
                    print(f"Generation attempt {attempt + 1} failed: {str(e)}")
                    time.sleep(1)  # Brief pause before retry
                    continue
                else:
                    raise Exception(f"Failed to generate utterance after {max_retries + 1} attempts: {str(e)}")
    
    def _validate_utterance(self, utterance: str) -> bool:
        """Validate that utterance meets requirements."""
        # Must have content
        if not utterance or len(utterance.strip()) < 20:
            return False
        
        # Should have approximately 2 sentences
        sentence_count = len([s for s in re.split(r'[.!?]+', utterance) if s.strip()])
        if sentence_count < 2:
            return False
        
        # Should have reasonable word count (10-50 words typical)
        word_count = len(utterance.split())
        if word_count < 10 or word_count > 100:
            return False
        
        return True
    
    def _split_sentences(self, utterance: str) -> List[str]:
        """Split utterance into individual sentences."""
        # Split on sentence endings and clean up
        sentences = re.split(r'[.!?]+', utterance)
        sentences = [s.strip() + '.' for s in sentences if s.strip()]
        
        # Remove extra periods
        sentences = [re.sub(r'\.+$', '.', s) for s in sentences]
        
        return sentences
    
    def _update_stats(self, domain: str, difficulty: float, generation_time: float):
        """Update generation statistics."""
        self.generation_stats['total_generated'] += 1
        
        if domain not in self.generation_stats['by_domain']:
            self.generation_stats['by_domain'][domain] = {
                'count': 0,
                'avg_difficulty': 0.0,
                'avg_time': 0.0
            }
        
        domain_stats = self.generation_stats['by_domain'][domain]
        domain_stats['count'] += 1
        
        # Update rolling averages
        total = self.generation_stats['total_generated']
        self.generation_stats['avg_difficulty'] = (
            (self.generation_stats['avg_difficulty'] * (total - 1) + difficulty) / total
        )
        self.generation_stats['generation_time'] = (
            (self.generation_stats['generation_time'] * (total - 1) + generation_time) / total
        )
        
        domain_count = domain_stats['count']
        domain_stats['avg_difficulty'] = (
            (domain_stats['avg_difficulty'] * (domain_count - 1) + difficulty) / domain_count
        )
        domain_stats['avg_time'] = (
            (domain_stats['avg_time'] * (domain_count - 1) + generation_time) / domain_count
        )
    
    def generate_batch(self, domain: str, count: int, delay_seconds: int = 10) -> List[Dict[str, Any]]:
        """
        Generate multiple utterances for a domain with rate limiting.
        
        Args:
            domain: Domain name
            count: Number of utterances to generate
            delay_seconds: Delay between API calls (default 10 as per requirements)
            
        Returns:
            List of utterance dictionaries
        """
        utterances = []
        
        print(f"Generating {count} utterances for {domain} domain...")
        
        for i in range(count):
            try:
                utterance = self.generate_utterance(domain)
                utterances.append(utterance)
                
                # Progress reporting
                if (i + 1) % 10 == 0:
                    print(f"Generated {i + 1}/{count} utterances for {domain}")
                    print(f"Latest: {utterance['utterance'][:100]}...")
                
                # Rate limiting - wait between API calls
                if i < count - 1:  # Don't wait after the last generation
                    time.sleep(delay_seconds)
                    
            except Exception as e:
                print(f"Error generating utterance {i + 1}: {str(e)}")
                continue
        
        print(f"Completed {len(utterances)}/{count} utterances for {domain}")
        return utterances
    
    def get_stats(self) -> Dict[str, Any]:
        """Get generation statistics."""
        return self.generation_stats.copy()

# Convenience functions
def generate_utterance(domain: str, profile: str = None) -> Dict[str, Any]:
    """Convenience function to generate a single utterance."""
    generator = UtteranceGenerator()
    return generator.generate_utterance(domain, profile)
