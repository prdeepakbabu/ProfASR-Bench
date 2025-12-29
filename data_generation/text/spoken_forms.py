"""
Spoken form transformation engine for converting written text to natural speech patterns.
Handles numbers, symbols, abbreviations, and domain-specific patterns.
"""

import re
from typing import Dict, List, Tuple, Any
from .domains import get_domain

class SpokenFormTransformer:
    """Transforms written text into spoken form patterns for ASR research."""
    
    def __init__(self):
        self.setup_transformation_rules()
    
    def setup_transformation_rules(self):
        """Initialize all transformation rules."""
        
        # Number words mapping
        self.number_words = {
            '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
            '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine',
            '10': 'ten', '11': 'eleven', '12': 'twelve', '13': 'thirteen', 
            '14': 'fourteen', '15': 'fifteen', '16': 'sixteen', '17': 'seventeen',
            '18': 'eighteen', '19': 'nineteen', '20': 'twenty', '30': 'thirty',
            '40': 'forty', '50': 'fifty', '60': 'sixty', '70': 'seventy',
            '80': 'eighty', '90': 'ninety', '100': 'one hundred'
        }
        
        # Common symbol replacements
        self.symbol_replacements = {
            '&': ' and ',
            '%': ' percent',
            '$': ' dollars',
            '€': ' euros',
            '£': ' pounds',
            '°': ' degrees',
            '#': ' number ',
            '@': ' at ',
            '+': ' plus ',
            '=': ' equals ',
            '/': ' slash ',
            '\\': ' backslash ',
            '*': ' star ',
            '(': ' open paren ',
            ')': ' close paren '
        }
        
        # Abbreviation expansions
        self.abbreviations = {
            'Dr.': 'Doctor',
            'Mr.': 'Mister', 
            'Mrs.': 'Missus',
            'Ms.': 'Miss',
            'St.': 'Street',
            'Ave.': 'Avenue',
            'Blvd.': 'Boulevard',
            'Inc.': 'Incorporated',
            'Corp.': 'Corporation',
            'LLC': 'L L C',
            'LLP': 'L L P',
            'PC': 'P C',
            'vs.': 'versus',
            'etc.': 'etcetera',
            'i.e.': 'that is',
            'e.g.': 'for example'
        }
        
        # Technical acronyms - spell out with spaces
        self.technical_acronyms = {
            'API': 'A P I',
            'HTTP': 'H T T P', 
            'HTTPS': 'H T T P S',
            'URL': 'U R L',
            'HTML': 'H T M L',
            'CSS': 'C S S',
            'JSON': 'J S O N',
            'XML': 'X M L',
            'SQL': 'S Q L',
            'CPU': 'C P U',
            'GPU': 'G P U',
            'RAM': 'R A M',
            'SSD': 'S S D',
            'USB': 'U S B',
            'WiFi': 'Wi Fi',
            'AWS': 'A W S',
            'GCP': 'G C P',
            'CI/CD': 'C I C D',
            'DevOps': 'Dev Ops',
            'MLOps': 'M L Ops',
            'AI': 'A I',
            'ML': 'M L',
            'UI': 'U I',
            'UX': 'U X'
        }
        
        # Medical abbreviations and terms
        self.medical_abbreviations = {
            'CT': 'C T',
            'MRI': 'M R I', 
            'PET': 'P E T',
            'EKG': 'E K G',
            'ECG': 'E C G',
            'BP': 'blood pressure',
            'HR': 'heart rate',
            'mg': 'milligrams',
            'ml': 'milliliters',
            'kg': 'kilograms',
            'cm': 'centimeters',
            'mm': 'millimeters',
            'IV': 'I V',
            'ER': 'E R',
            'ICU': 'I C U',
            'OR': 'O R'
        }
        
        # Legal abbreviations
        self.legal_abbreviations = {
            'v.': 'versus',
            'vs.': 'versus',
            'CV': 'C V',
            'No.': 'number',
            '§': 'section',
            'LLC': 'L L C',
            'LLP': 'L L P', 
            'Inc.': 'Incorporated',
            'Corp.': 'Corporation',
            'Ltd.': 'Limited',
            'Co.': 'Company'
        }
        
        # Financial abbreviations
        self.financial_abbreviations = {
            'IPO': 'I P O',
            'ETF': 'E T F',
            'REIT': 'R E I T',
            'NYSE': 'N Y S E',
            'NASDAQ': 'nasdaq',
            'S&P': 'S and P',
            'Q1': 'Q one',
            'Q2': 'Q two', 
            'Q3': 'Q three',
            'Q4': 'Q four',
            'YTD': 'year to date',
            'ROI': 'R O I',
            'P/E': 'P E ratio',
            'GDP': 'G D P',
            'CPI': 'C P I'
        }
    
    def transform_numbers(self, text: str) -> str:
        """Convert numeric digits to spoken words."""
        
        # Handle years (2020-2030)
        text = re.sub(r'\b(20[2-3][0-9])\b', self._convert_year, text)
        
        # Handle times (12:30, 3:45 PM, etc.)
        text = re.sub(r'\b(\d{1,2}):(\d{2})\s*(AM|PM|am|pm)?\b', self._convert_time, text)
        
        # Handle percentages (3.5%, 15%, etc.)
        text = re.sub(r'\b(\d+(?:\.\d+)?)%\b', self._convert_percentage, text)
        
        # Handle money amounts ($150, $1.5M, etc.)
        text = re.sub(r'\$(\d+(?:\.\d+)?)\s*([KMB]?)\b', self._convert_money, text)
        
        # Handle decimals (3.14, 2.5, etc.)
        text = re.sub(r'\b(\d+)\.(\d+)\b', self._convert_decimal, text)
        
        # Handle ranges (15-20, 50/60, etc.)
        text = re.sub(r'\b(\d+)[-/](\d+)\b', self._convert_range, text)
        
        # Handle simple integers
        text = re.sub(r'\b(\d+)\b', self._convert_integer, text)
        
        return text
    
    def _convert_year(self, match) -> str:
        """Convert year to spoken form (2024 -> twenty twenty four)."""
        year = match.group(1)
        first_two = year[:2]
        last_two = year[2:]
        
        first_part = "twenty" if first_two == "20" else self._convert_integer_value(int(first_two))
        last_part = self._convert_integer_value(int(last_two))
        
        return f"{first_part} {last_part}"
    
    def _convert_time(self, match) -> str:
        """Convert time to spoken form (3:30 PM -> three thirty P M)."""
        hour = int(match.group(1))
        minute = int(match.group(2))
        period = match.group(3)
        
        hour_spoken = self._convert_integer_value(hour)
        
        if minute == 0:
            minute_spoken = "o'clock"
        elif minute < 10:
            minute_spoken = f"oh {self._convert_integer_value(minute)}"
        else:
            minute_spoken = self._convert_integer_value(minute)
        
        result = f"{hour_spoken} {minute_spoken}"
        if period:
            result += f" {period.upper().replace('', ' ').strip()}"
        
        return result
    
    def _convert_percentage(self, match) -> str:
        """Convert percentage to spoken form (3.5% -> three point five percent)."""
        number = match.group(1)
        if '.' in number:
            return f"{self._convert_decimal_value(number)} percent"
        else:
            return f"{self._convert_integer_value(int(number))} percent"
    
    def _convert_money(self, match) -> str:
        """Convert money to spoken form ($1.5M -> one point five million dollars)."""
        amount = match.group(1)
        suffix = match.group(2)
        
        # Convert the base amount
        if '.' in amount:
            amount_spoken = self._convert_decimal_value(amount)
        else:
            amount_spoken = self._convert_integer_value(int(amount))
        
        # Add suffix
        if suffix == 'K':
            amount_spoken += " thousand"
        elif suffix == 'M':
            amount_spoken += " million"
        elif suffix == 'B':
            amount_spoken += " billion"
        
        return f"{amount_spoken} dollars"
    
    def _convert_decimal(self, match) -> str:
        """Convert decimal to spoken form (3.14 -> three point fourteen)."""
        return self._convert_decimal_value(f"{match.group(1)}.{match.group(2)}")
    
    def _convert_range(self, match) -> str:
        """Convert range to spoken form (15-20 -> fifteen to twenty)."""
        first = self._convert_integer_value(int(match.group(1)))
        second = self._convert_integer_value(int(match.group(2)))
        separator = " to " if "-" in match.group(0) else " over "
        return f"{first}{separator}{second}"
    
    def _convert_integer(self, match) -> str:
        """Convert integer to spoken form."""
        return self._convert_integer_value(int(match.group(1)))
    
    def _convert_integer_value(self, num: int) -> str:
        """Convert integer to word form."""
        if num < 0:
            return f"negative {self._convert_integer_value(-num)}"
        
        if num <= 20:
            return self.number_words.get(str(num), str(num))
        elif num < 100:
            tens = (num // 10) * 10
            ones = num % 10
            if ones == 0:
                return self.number_words[str(tens)]
            else:
                return f"{self.number_words[str(tens)]} {self.number_words[str(ones)]}"
        elif num < 1000:
            hundreds = num // 100
            remainder = num % 100
            result = f"{self.number_words[str(hundreds)]} hundred"
            if remainder > 0:
                result += f" {self._convert_integer_value(remainder)}"
            return result
        elif num < 1000000:
            thousands = num // 1000
            remainder = num % 1000
            result = f"{self._convert_integer_value(thousands)} thousand"
            if remainder > 0:
                result += f" {self._convert_integer_value(remainder)}"
            return result
        else:
            return str(num)  # For very large numbers, keep as digits
    
    def _convert_decimal_value(self, decimal_str: str) -> str:
        """Convert decimal string to spoken form."""
        parts = decimal_str.split('.')
        integer_part = self._convert_integer_value(int(parts[0]))
        decimal_part = ' '.join([self.number_words.get(d, d) for d in parts[1]])
        return f"{integer_part} point {decimal_part}"
    
    def transform_symbols(self, text: str) -> str:
        """Convert symbols to spoken form."""
        for symbol, replacement in self.symbol_replacements.items():
            text = text.replace(symbol, replacement)
        return text
    
    def transform_abbreviations(self, text: str) -> str:
        """Convert abbreviations to spoken form."""
        for abbrev, expansion in self.abbreviations.items():
            text = re.sub(rf'\b{re.escape(abbrev)}\b', expansion, text, flags=re.IGNORECASE)
        return text
    
    def transform_domain_specific(self, text: str, domain: str) -> str:
        """Apply domain-specific transformations."""
        domain = domain.lower()
        
        if domain == "medical":
            return self._apply_medical_transforms(text)
        elif domain == "legal":
            return self._apply_legal_transforms(text)
        elif domain == "financial":
            return self._apply_financial_transforms(text)
        elif domain == "technical":
            return self._apply_technical_transforms(text)
        
        return text
    
    def _apply_medical_transforms(self, text: str) -> str:
        """Apply medical domain transformations."""
        for abbrev, expansion in self.medical_abbreviations.items():
            text = re.sub(rf'\b{re.escape(abbrev)}\b', expansion, text, flags=re.IGNORECASE)
        
        # Handle medical dosages (5mg -> five milligrams)
        text = re.sub(r'\b(\d+)\s*mg\b', lambda m: f"{self._convert_integer_value(int(m.group(1)))} milligrams", text)
        text = re.sub(r'\b(\d+)\s*ml\b', lambda m: f"{self._convert_integer_value(int(m.group(1)))} milliliters", text)
        
        # Handle blood pressure (120/80 -> one twenty over eighty)
        text = re.sub(r'\b(\d+)/(\d+)\b', lambda m: f"{self._convert_integer_value(int(m.group(1)))} over {self._convert_integer_value(int(m.group(2)))}", text)
        
        return text
    
    def _apply_legal_transforms(self, text: str) -> str:
        """Apply legal domain transformations."""
        for abbrev, expansion in self.legal_abbreviations.items():
            text = re.sub(rf'\b{re.escape(abbrev)}\b', expansion, text, flags=re.IGNORECASE)
        
        # Handle case numbers (2024-123 -> twenty twenty four dash one two three)
        text = re.sub(r'\b(\d{4})-(\d+)\b', 
                     lambda m: f"{self._convert_year(type('', (), {'group': lambda x: m.group(1)})())} dash {' '.join([self.number_words.get(d, d) for d in m.group(2)])}", 
                     text)
        
        return text
    
    def _apply_financial_transforms(self, text: str) -> str:
        """Apply financial domain transformations."""
        for abbrev, expansion in self.financial_abbreviations.items():
            text = re.sub(rf'\b{re.escape(abbrev)}\b', expansion, text, flags=re.IGNORECASE)
        
        # Handle basis points (15bps -> fifteen basis points)
        text = re.sub(r'\b(\d+)\s*bps?\b', lambda m: f"{self._convert_integer_value(int(m.group(1)))} basis points", text)
        
        return text
    
    def _apply_technical_transforms(self, text: str) -> str:
        """Apply technical domain transformations."""
        for abbrev, expansion in self.technical_acronyms.items():
            text = re.sub(rf'\b{re.escape(abbrev)}\b', expansion, text, flags=re.IGNORECASE)
        
        # Handle IP addresses (192.168.1.1 -> one nine two dot one six eight dot one dot one)
        text = re.sub(r'\b(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})\b', 
                     lambda m: f"{' '.join([self.number_words.get(d, d) for d in m.group(1)])} dot {' '.join([self.number_words.get(d, d) for d in m.group(2)])} dot {' '.join([self.number_words.get(d, d) for d in m.group(3)])} dot {' '.join([self.number_words.get(d, d) for d in m.group(4)])}", 
                     text)
        
        # Handle ports (port 80 -> port eighty)
        text = re.sub(r'\bport\s+(\d+)\b', lambda m: f"port {self._convert_integer_value(int(m.group(1)))}", text, flags=re.IGNORECASE)
        
        return text
    
    def apply_all_transformations(self, text: str, domain: str = None) -> str:
        """Apply all transformations to text."""
        # Apply transformations in order
        text = self.transform_numbers(text)
        text = self.transform_symbols(text)
        text = self.transform_abbreviations(text)
        
        if domain:
            text = self.transform_domain_specific(text, domain)
        
        # Clean up extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

# Singleton instance
spoken_form_transformer = SpokenFormTransformer()

def apply_spoken_forms(text: str, domain: str = None) -> str:
    """Convenience function to apply all spoken form transformations."""
    return spoken_form_transformer.apply_all_transformations(text, domain)

def transform_numbers(text: str) -> str:
    """Convenience function to transform numbers only."""
    return spoken_form_transformer.transform_numbers(text)

def transform_symbols(text: str) -> str:
    """Convenience function to transform symbols only."""
    return spoken_form_transformer.transform_symbols(text)
