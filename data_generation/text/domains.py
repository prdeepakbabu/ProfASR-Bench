"""
Domain configuration system for memory bank utterance generation.
Defines 4 research domains with their specific vocabulary, context patterns, and ASR error-prone terms.
"""

from typing import Dict, List, Any
import random

class Domain:
    """Base class for domain configuration."""
    
    def __init__(self, name: str, vocabulary: List[str], contexts: List[str], 
                 error_prone_terms: List[str], entity_types: List[str]):
        self.name = name
        self.vocabulary = vocabulary
        self.contexts = contexts
        self.error_prone_terms = error_prone_terms
        self.entity_types = entity_types
    
    def get_random_vocabulary(self, count: int = 5) -> List[str]:
        """Get random vocabulary terms from this domain."""
        return random.sample(self.vocabulary, min(count, len(self.vocabulary)))
    
    def get_random_context(self) -> str:
        """Get a random context pattern for this domain."""
        return random.choice(self.contexts)
    
    def get_error_prone_terms(self, count: int = 3) -> List[str]:
        """Get random ASR error-prone terms from this domain."""
        return random.sample(self.error_prone_terms, min(count, len(self.error_prone_terms)))

# Medical Domain Configuration
MEDICAL_DOMAIN = Domain(
    name="medical",
    vocabulary=[
        "radiology", "MRI", "CT scan", "ultrasound", "X-ray", "mammogram",
        "prednisone", "ibuprofen", "acetaminophen", "lisinopril", "metformin",
        "hypertension", "diabetes", "pneumonia", "bronchitis", "arrhythmia",
        "physician", "surgeon", "cardiologist", "oncologist", "neurologist",
        "milligrams", "milliliters", "blood pressure", "heart rate", "temperature",
        "diagnosis", "prognosis", "treatment", "medication", "prescription"
    ],
    contexts=[
        "reviewing patient charts in the morning",
        "discussing treatment options with colleagues",
        "updating medical records after consultation",
        "preparing for surgery tomorrow",
        "analyzing diagnostic test results",
        "conducting patient rounds",
        "reviewing medication dosages",
        "discussing discharge planning"
    ],
    error_prone_terms=[
        "fifty milligrams", "fifteen milligrams", "thirty milligrams",
        "M and M allergy", "bee pollen allergy", "penicillin allergy",
        "C T scan", "M R I scan", "P E T scan",
        "Doctor Smith", "Doctor Jones", "Doctor Lee",
        "room two oh one", "room three fifteen", "room four fifty",
        "one twenty over eighty", "ninety over sixty", "one forty over ninety",
        "twice daily", "three times daily", "every six hours",
        "acetaminophen", "ibuprofen", "prednisone"
    ],
    entity_types=[
        "medication_names", "dosages", "room_numbers", "doctor_names",
        "medical_procedures", "vital_signs", "allergies", "frequencies"
    ]
)

# Legal Domain Configuration  
LEGAL_DOMAIN = Domain(
    name="legal",
    vocabulary=[
        "plaintiff", "defendant", "litigation", "deposition", "discovery",
        "contract", "agreement", "statute", "regulation", "ordinance",
        "court", "judge", "jury", "attorney", "counsel",
        "evidence", "testimony", "witness", "exhibit", "motion",
        "appeal", "verdict", "settlement", "damages", "injunction",
        "copyright", "trademark", "patent", "intellectual property",
        "merger", "acquisition", "due diligence", "compliance"
    ],
    contexts=[
        "preparing for court hearing next week",
        "reviewing contract terms with client",
        "drafting legal brief for appeal",
        "conducting deposition this afternoon",
        "analyzing case precedents",
        "meeting with opposing counsel",
        "filing motion with the court",
        "negotiating settlement terms"
    ],
    error_prone_terms=[
        "section five oh one c three", "section four oh one k",
        "case number two thousand twenty four dash one two three",
        "docket number C V dash twenty twenty four dash oh oh one",
        "versus", "plaintiff versus defendant",
        "courtroom two A", "courtroom three B", "courtroom fifteen",
        "judge Johnson", "judge Williams", "judge Garcia",
        "fifteen thousand dollars", "fifty thousand dollars", "two million dollars",
        "L L C", "L L P", "incorporated", "P C",
        "copyright infringement", "trademark violation"
    ],
    entity_types=[
        "case_numbers", "docket_numbers", "section_references", "monetary_amounts",
        "judge_names", "courtroom_numbers", "business_entities", "legal_citations"
    ]
)

# Financial Domain Configuration
FINANCIAL_DOMAIN = Domain(
    name="financial",
    vocabulary=[
        "portfolio", "investment", "equity", "bond", "mutual fund",
        "dividend", "yield", "interest rate", "principal", "capital",
        "assets", "liabilities", "revenue", "profit", "loss",
        "budget", "forecast", "projection", "analysis", "valuation",
        "merger", "acquisition", "IPO", "securities", "derivatives",
        "compliance", "regulation", "audit", "tax", "accounting",
        "client", "investor", "shareholder", "stakeholder"
    ],
    contexts=[
        "reviewing quarterly earnings report",
        "analyzing market trends this morning",
        "preparing investment recommendations",
        "meeting with high net worth client",
        "conducting portfolio rebalancing",
        "discussing tax optimization strategies",
        "evaluating merger opportunity",
        "updating financial projections"
    ],
    error_prone_terms=[
        "fifteen basis points", "fifty basis points", "thirty basis points",
        "two point five percent", "three point seven five percent",
        "one point five million", "twenty five million", "fifty million",
        "Q one", "Q two", "Q three", "Q four",
        "twenty twenty four", "twenty twenty five",
        "S and P five hundred", "nasdaq", "dow jones",
        "Goldman Sachs", "Morgan Stanley", "J P Morgan",
        "triple A rated", "double A rated", "B B B rated",
        "I P O", "E T F", "R E I T"
    ],
    entity_types=[
        "percentages", "basis_points", "monetary_amounts", "quarters",
        "years", "financial_indices", "company_names", "ratings",
        "financial_instruments"
    ]
)

# Technical Domain Configuration
TECHNICAL_DOMAIN = Domain(
    name="technical",
    vocabulary=[
        "database", "server", "application", "software", "hardware",
        "network", "security", "encryption", "authentication", "authorization",
        "API", "REST", "JSON", "XML", "HTTP", "HTTPS",
        "cloud", "AWS", "Azure", "deployment", "container",
        "microservices", "architecture", "scalability", "performance",
        "debugging", "testing", "development", "production", "staging",
        "repository", "version control", "CI/CD", "DevOps"
    ],
    contexts=[
        "deploying new feature to production",
        "debugging performance issue",
        "reviewing code in pull request",
        "setting up CI/CD pipeline",
        "configuring cloud infrastructure",
        "conducting security audit",
        "optimizing database queries",
        "planning system architecture"
    ],
    error_prone_terms=[
        "I P version four", "I P version six",
        "H T T P", "H T T P S", "A P I",
        "C P U", "R A M", "S S D",
        "A W S", "Azure", "G C P",
        "port eighty", "port four forty three", "port twenty two",
        "localhost three thousand", "one nine two dot one six eight",
        "Git Hub", "Git Lab", "Bit Bucket",
        "C I C D", "Dev Ops", "M L Ops",
        "Docker", "Kubernetes", "Jenkins"
    ],
    entity_types=[
        "ip_addresses", "port_numbers", "protocols", "cloud_services",
        "server_names", "urls", "technical_acronyms", "version_numbers"
    ]
)

# Domain registry
DOMAINS = {
    "medical": MEDICAL_DOMAIN,
    "legal": LEGAL_DOMAIN, 
    "financial": FINANCIAL_DOMAIN,
    "technical": TECHNICAL_DOMAIN
}

def get_domain(domain_name: str) -> Domain:
    """Get domain configuration by name."""
    if domain_name.lower() not in DOMAINS:
        raise ValueError(f"Unknown domain: {domain_name}. Available domains: {list(DOMAINS.keys())}")
    return DOMAINS[domain_name.lower()]

def get_all_domains() -> Dict[str, Domain]:
    """Get all available domains."""
    return DOMAINS.copy()

def get_domain_names() -> List[str]:
    """Get list of all domain names."""
    return list(DOMAINS.keys())
