"""
Profile generator for creating realistic user profiles per domain.
Generates diverse demographics, professional backgrounds, and speech characteristics.
"""

from typing import Dict, List, Tuple
import random
from .domains import get_domain, Domain

class ProfileGenerator:
    """Generates realistic user profiles for each domain."""
    
    def __init__(self):
        # Define demographics data
        self.regions = {
            "medical": [
                "India", "Texas", "California", "New York", "Florida", "Boston", 
                "Chicago", "Philadelphia", "Atlanta", "Detroit", "Seattle", "Denver"
            ],
            "legal": [
                "California", "New York", "Texas", "Florida", "Illinois",
                "Washington DC", "Boston", "Chicago", "Miami", "Los Angeles"
            ],
            "financial": [
                "London", "New York", "Chicago", "Boston", "San Francisco",
                "Hong Kong", "Singapore", "Toronto", "Frankfurt", "Zurich"
            ],
            "technical": [
                "Seattle", "San Francisco", "Austin", "Boston", "New York",
                "Toronto", "London", "Berlin", "Bangalore", "Tel Aviv"
            ]
        }
        
        self.roles = {
            "medical": [
                "radiologist", "cardiologist", "oncologist", "surgeon", "physician",
                "nurse practitioner", "physician assistant", "medical resident",
                "pharmacist", "medical technologist", "respiratory therapist",
                "physical therapist", "medical administrator", "clinical coordinator"
            ],
            "legal": [
                "immigration lawyer", "corporate attorney", "criminal defense lawyer",
                "personal injury attorney", "family law attorney", "tax attorney",
                "patent attorney", "court reporter", "paralegal", "legal assistant",
                "law clerk", "mediator", "arbitrator", "legal consultant"
            ],
            "financial": [
                "investment banker", "financial advisor", "portfolio manager",
                "tax accountant", "financial analyst", "wealth manager",
                "insurance broker", "loan officer", "credit analyst",
                "investment analyst", "risk manager", "compliance officer",
                "treasury analyst", "financial planner"
            ],
            "technical": [
                "software engineer", "DevOps engineer", "data scientist",
                "system administrator", "security analyst", "cloud architect",
                "product manager", "technical lead", "database administrator",
                "network engineer", "QA engineer", "mobile developer",
                "frontend developer", "backend developer"
            ]
        }
        
        self.experience_levels = [
            "2 years", "5 years", "8 years", "10 years", "12 years", 
            "15 years", "18 years", "20 years", "25 years"
        ]
        
        self.speech_characteristics = [
            "speaks quickly", "speaks slowly and deliberately", "has a slight accent",
            "uses technical jargon frequently", "speaks very clearly", 
            "tends to use abbreviations", "speaks with authority",
            "has a conversational tone", "is very precise with language"
        ]
        
        self.age_ranges = ["mid-twenties", "early thirties", "mid-thirties", 
                          "early forties", "mid-forties", "early fifties"]
    
    def generate_profile(self, domain_name: str) -> str:
        """
        Generate a realistic user profile for the specified domain.
        
        Args:
            domain_name: Name of the domain (medical, legal, financial, technical)
            
        Returns:
            String describing the user profile
        """
        domain = get_domain(domain_name)
        
        # Select random attributes
        role = random.choice(self.roles[domain_name])
        region = random.choice(self.regions[domain_name])
        experience = random.choice(self.experience_levels)
        age = random.choice(self.age_ranges)
        
        # Generate profile variations
        profile_formats = [
            f"{role} from {region} with {experience} experience",
            f"{age} {role} from {region}",
            f"experienced {role} from {region}",
            f"{role} based in {region} with {experience} in the field",
            f"senior {role} from {region}",
            f"{role} working in {region} for {experience}",
        ]
        
        return random.choice(profile_formats)
    
    def generate_detailed_profile(self, domain_name: str) -> Dict[str, str]:
        """
        Generate a detailed user profile with multiple attributes.
        
        Args:
            domain_name: Name of the domain
            
        Returns:
            Dictionary with detailed profile information
        """
        domain = get_domain(domain_name)
        
        profile = {
            "role": random.choice(self.roles[domain_name]),
            "region": random.choice(self.regions[domain_name]),
            "experience": random.choice(self.experience_levels),
            "age_range": random.choice(self.age_ranges),
            "speech_style": random.choice(self.speech_characteristics),
            "domain": domain_name
        }
        
        # Generate summary
        profile["summary"] = f"{profile['age_range']} {profile['role']} from {profile['region']} with {profile['experience']} experience who {profile['speech_style']}"
        
        return profile
    
    def generate_batch_profiles(self, domain_name: str, count: int) -> List[str]:
        """
        Generate multiple profiles for batch processing.
        
        Args:
            domain_name: Name of the domain
            count: Number of profiles to generate
            
        Returns:
            List of profile strings
        """
        profiles = []
        for _ in range(count):
            profiles.append(self.generate_profile(domain_name))
        return profiles
    
    def get_profile_context_hints(self, domain_name: str, profile: str) -> List[str]:
        """
        Get context hints based on profile for utterance generation.
        
        Args:
            domain_name: Name of the domain
            profile: Profile string
            
        Returns:
            List of context hints for utterance generation
        """
        domain = get_domain(domain_name)
        hints = []
        
        # Add regional context
        if "India" in profile:
            hints.append("may use British English terminology")
        if "London" in profile:
            hints.append("uses British terminology and spellings")
        if "Texas" in profile or "Southern" in profile:
            hints.append("may have Southern American speech patterns")
        
        # Add experience-based context
        if "senior" in profile or "20 years" in profile or "25 years" in profile:
            hints.append("uses established professional terminology")
        if "2 years" in profile or "5 years" in profile:
            hints.append("may be more formal or cautious in speech")
        
        # Add role-specific context
        if any(role in profile for role in ["surgeon", "physician", "doctor"]):
            hints.append("speaks with medical authority")
        if any(role in profile for role in ["lawyer", "attorney", "counsel"]):
            hints.append("uses precise legal language")
        if any(role in profile for role in ["engineer", "developer", "architect"]):
            hints.append("uses technical terminology naturally")
        
        return hints

# Singleton instance
profile_generator = ProfileGenerator()

def generate_profile(domain_name: str) -> str:
    """Convenience function to generate a profile."""
    return profile_generator.generate_profile(domain_name)

def generate_detailed_profile(domain_name: str) -> Dict[str, str]:
    """Convenience function to generate detailed profile."""
    return profile_generator.generate_detailed_profile(domain_name)

def generate_batch_profiles(domain_name: str, count: int) -> List[str]:
    """Convenience function to generate batch profiles."""
    return profile_generator.generate_batch_profiles(domain_name, count)
