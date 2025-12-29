"""
Prompt configuration definitions for prompt-conditioned ASR experiments.
Supports various prompting strategies and templates.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from enum import Enum

class PromptStrategy(Enum):
    """Available prompt conditioning strategies."""
    NONE = "none"                    # No prompting (baseline)
    DATASET_PROMPT = "dataset_prompt"  # Use existing "prompt" field
    DOMAIN_ONLY = "domain_only"       # Use domain field only
    ACCENT_ONLY = "accent_only"       # Use accent info only
    GENDER_ONLY = "gender_only"       # Use gender info only
    DOMAIN_GENDER = "domain_gender"   # Combine domain + gender
    DOMAIN_ACCENT = "domain_accent"   # Combine domain + accent
    ACCENT_GENDER = "accent_gender"   # Combine accent + gender
    FULL_CONTEXT = "full_context"     # Domain + gender + accent
    CUSTOM_TEMPLATE = "custom_template" # User-defined template

@dataclass
class PromptConfig:
    """Configuration for prompt-conditioned ASR."""
    
    # Core settings
    enabled: bool = False
    strategy: PromptStrategy = PromptStrategy.NONE
    custom_template: Optional[str] = None
    max_prompt_length: int = 20  # Reduced to stay within Whisper limits
    
    # Voice profile mappings (corrected per user feedback)
    gender_mapping: Dict[str, str] = field(default_factory=lambda: {
        "michael": "male", 
        "george": "male",
        "emma": "female", 
        "heart": "female"
    })
    
    accent_mapping: Dict[str, str] = field(default_factory=lambda: {
        "michael": "american", 
        "george": "british", 
        "emma": "british",    # Corrected: emma is british
        "heart": "american"
    })
    
    # Domain formatting
    domain_formatting: Dict[str, str] = field(default_factory=lambda: {
        "financial": "financial",
        "medical": "medical", 
        "legal": "legal",
        "technical": "technical"
    })
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "enabled": self.enabled,
            "strategy": self.strategy.value,
            "custom_template": self.custom_template,
            "max_prompt_length": self.max_prompt_length,
            "gender_mapping": self.gender_mapping,
            "accent_mapping": self.accent_mapping,
            "domain_formatting": self.domain_formatting
        }

# Predefined prompt templates
PROMPT_TEMPLATES = {
    PromptStrategy.DATASET_PROMPT: "{prompt}",
    PromptStrategy.DOMAIN_ONLY: "This is {domain} content.",
    PromptStrategy.ACCENT_ONLY: "Speaker has {accent} accent.",
    PromptStrategy.GENDER_ONLY: "Speaker is {gender}.",
    PromptStrategy.DOMAIN_GENDER: "This is {domain} content spoken by a {gender}.",
    PromptStrategy.DOMAIN_ACCENT: "This is {domain} content with {accent} accent.",
    PromptStrategy.ACCENT_GENDER: "This is spoken by a {gender} with {accent} accent.",
    PromptStrategy.FULL_CONTEXT: "This is {domain} content spoken by a {gender} with {accent} accent."
}

# Alternative template variations for experimentation
ALTERNATIVE_TEMPLATES = {
    "formal": {
        PromptStrategy.DOMAIN_ONLY: "The following content pertains to {domain}.",
        PromptStrategy.ACCENT_ONLY: "The speaker exhibits a {accent} accent.",
        PromptStrategy.GENDER_ONLY: "The speaker is {gender}.",
        PromptStrategy.FULL_CONTEXT: "The following {domain} content is spoken by a {gender} speaker with a {accent} accent."
    },
    "casual": {
        PromptStrategy.DOMAIN_ONLY: "{domain} stuff",
        PromptStrategy.ACCENT_ONLY: "{accent} speaker",
        PromptStrategy.GENDER_ONLY: "{gender} voice",
        PromptStrategy.FULL_CONTEXT: "{domain} content, {gender} {accent} speaker"
    },
    "technical": {
        PromptStrategy.DOMAIN_ONLY: "Domain: {domain}",
        PromptStrategy.ACCENT_ONLY: "Accent: {accent}",
        PromptStrategy.GENDER_ONLY: "Gender: {gender}",
        PromptStrategy.FULL_CONTEXT: "Domain: {domain}, Speaker: {gender}, Accent: {accent}"
    },
    "short": {
        PromptStrategy.DOMAIN_ONLY: "{domain}",
        PromptStrategy.ACCENT_ONLY: "{accent}",
        PromptStrategy.GENDER_ONLY: "{gender}",
        PromptStrategy.DOMAIN_GENDER: "{domain} {gender}",
        PromptStrategy.DOMAIN_ACCENT: "{domain} {accent}",
        PromptStrategy.ACCENT_GENDER: "{accent} {gender}",
        PromptStrategy.FULL_CONTEXT: "{domain} {gender} {accent}"
    }
}

def create_prompt_config(
    strategy: str = "none",
    enabled: bool = None,
    custom_template: Optional[str] = None,
    template_style: str = "default"
) -> PromptConfig:
    """Create a prompt configuration with the specified strategy."""
    
    # Auto-enable if strategy is not 'none'
    if enabled is None:
        enabled = strategy != "none"
    
    prompt_strategy = PromptStrategy(strategy)
    
    config = PromptConfig(
        enabled=enabled,
        strategy=prompt_strategy,
        custom_template=custom_template
    )
    
    return config

def generate_prompt_text(
    sample_metadata: Dict[str, Any],
    prompt_config: PromptConfig,
    template_style: str = "default"
) -> str:
    """Generate prompt text based on strategy and sample metadata."""
    
    if not prompt_config.enabled or prompt_config.strategy == PromptStrategy.NONE:
        return ""
    
    # Handle custom template
    if prompt_config.strategy == PromptStrategy.CUSTOM_TEMPLATE:
        if prompt_config.custom_template:
            return _format_template(prompt_config.custom_template, sample_metadata, prompt_config)
        else:
            raise ValueError("Custom template strategy requires custom_template to be set")
    
    # Handle dataset prompt
    if prompt_config.strategy == PromptStrategy.DATASET_PROMPT:
        return sample_metadata.get("prompt", "")
    
    # Select template set
    if template_style == "default":
        templates = PROMPT_TEMPLATES
    else:
        templates = ALTERNATIVE_TEMPLATES.get(template_style, PROMPT_TEMPLATES)
    
    # Get template for strategy
    template = templates.get(prompt_config.strategy, "")
    if not template:
        return ""
    
    return _format_template(template, sample_metadata, prompt_config)

def _format_template(
    template: str,
    sample_metadata: Dict[str, Any],
    prompt_config: PromptConfig
) -> str:
    """Format template with sample metadata."""
    
    format_vars = {}
    
    # Add domain if needed
    if "{domain}" in template:
        domain = sample_metadata.get("domain", "unknown")
        format_vars["domain"] = prompt_config.domain_formatting.get(domain, domain)
    
    # Add gender if needed
    if "{gender}" in template:
        voice = sample_metadata.get("voice", "unknown")
        format_vars["gender"] = prompt_config.gender_mapping.get(voice, "unknown")
    
    # Add accent if needed
    if "{accent}" in template:
        voice = sample_metadata.get("voice", "unknown")
        format_vars["accent"] = prompt_config.accent_mapping.get(voice, "unknown")
    
    # Add raw prompt if needed
    if "{prompt}" in template:
        format_vars["prompt"] = sample_metadata.get("prompt", "")
    
    try:
        formatted = template.format(**format_vars)
        
        # Truncate if too long
        if len(formatted) > prompt_config.max_prompt_length:
            formatted = formatted[:prompt_config.max_prompt_length].rstrip()
        
        return formatted
        
    except KeyError as e:
        raise ValueError(f"Template formatting failed - missing variable {e}")

# Predefined prompt configurations for experiments
PROMPT_CONFIGS = {
    "none": PromptConfig(enabled=False, strategy=PromptStrategy.NONE),
    "dataset_prompt": PromptConfig(enabled=True, strategy=PromptStrategy.DATASET_PROMPT),
    "domain_only": PromptConfig(enabled=True, strategy=PromptStrategy.DOMAIN_ONLY),
    "accent_only": PromptConfig(enabled=True, strategy=PromptStrategy.ACCENT_ONLY),
    "gender_only": PromptConfig(enabled=True, strategy=PromptStrategy.GENDER_ONLY),
    "domain_gender": PromptConfig(enabled=True, strategy=PromptStrategy.DOMAIN_GENDER),
    "domain_accent": PromptConfig(enabled=True, strategy=PromptStrategy.DOMAIN_ACCENT),
    "accent_gender": PromptConfig(enabled=True, strategy=PromptStrategy.ACCENT_GENDER),
    "full_context": PromptConfig(enabled=True, strategy=PromptStrategy.FULL_CONTEXT),
}

def get_prompt_config(name: str) -> PromptConfig:
    """Get prompt configuration by name."""
    if name not in PROMPT_CONFIGS:
        raise ValueError(f"Unknown prompt config: {name}. Available configs: {list(PROMPT_CONFIGS.keys())}")
    return PROMPT_CONFIGS[name]

def list_available_prompt_configs() -> List[str]:
    """List all available prompt configurations."""
    return list(PROMPT_CONFIGS.keys())

# Evaluation sets for systematic comparison
PROMPT_EVALUATION_SETS = {
    "baseline": ["none"],
    "single_factor": ["none", "domain_only", "accent_only", "gender_only"],
    "two_factor": ["none", "domain_gender", "domain_accent", "accent_gender"],
    "comprehensive": ["none", "domain_only", "accent_only", "gender_only", 
                     "domain_gender", "domain_accent", "accent_gender", "full_context"],
    "all_strategies": list(PROMPT_CONFIGS.keys())
}

def get_prompt_evaluation_set(name: str) -> List[str]:
    """Get a predefined set of prompt configurations for evaluation."""
    if name not in PROMPT_EVALUATION_SETS:
        raise ValueError(f"Unknown evaluation set: {name}. Available sets: {list(PROMPT_EVALUATION_SETS.keys())}")
    return PROMPT_EVALUATION_SETS[name]
