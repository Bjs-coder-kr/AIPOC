# module/target_optimizer/guardrail.py
"""
Safety Guardrails for Target Optimization.
Exportable Module - No external dependencies.

Roles:
1. Grounding Check: Prevent number/date hallucination.
2. NER Guard: Prevent arbitrary changes to proper nouns.
"""

import re
import logging

logger = logging.getLogger(__name__)

class TargetGuardrail:
    """
    Triple Guardrail Implementation.
    Currently implements Grounding Check and NER Guard.
    """
    
    def __init__(self):
        # 1. Grounding Patterns (Numbers & Dates)
        self.num_pattern = re.compile(r'\d+(?:,\d{3})*(?:\.\d+)?%?') 
        self.date_pattern = re.compile(r'\d{4}[-./년]\d{2}[-./월]\d{2}?')
        
    def verify_all(self, original: str, generated: str) -> bool:
        """
        Unified verification (Fail-Fast).
        """
        if not self.verify_grounding(original, generated):
            return False
            
        if not self.verify_ner(original, generated):
            return False
            
        return True

    def verify_grounding(self, original: str, generated: str) -> bool:
        """
        Number/Date integrity verification.
        Rule: All numbers/dates in generated text must exist in source.
        """
        src_nums = set(self.num_pattern.findall(original))
        gen_nums = set(self.num_pattern.findall(generated))
        
        src_dates = set(self.date_pattern.findall(original))
        gen_dates = set(self.date_pattern.findall(generated))
        
        src_set = src_nums | src_dates
        gen_set = gen_nums | gen_dates
        
        if not gen_set.issubset(src_set):
            diff = gen_set - src_set
            logger.warning(f"🛡️ [Guardrail] Grounding Fail. Hallucinated: {diff}")
            return False
            
        return True

    def verify_ner(self, original: str, generated: str) -> bool:
        """
        Named Entity Recognition (NER) protection - Heuristic Regex.
        
        Logic:
        - English words starting with capital letters (Proper Nouns)
        - Quoted words ("word") -> Considered important keywords
        """
        eng_ner_pattern = re.compile(r'\b[A-Z][a-zA-Z0-9_]+\b')
        
        src_eng = set(eng_ner_pattern.findall(original))
        gen_eng = set(eng_ner_pattern.findall(generated))
        
        # Check Subset (Soft Guardrail - Warning only)
        if not gen_eng.issubset(src_eng):
            diff = gen_eng - src_eng
            logger.warning(f"🛡️ [Guardrail] NER Fail (Ignored). Unknown Entities: {diff}")
            # Soft guardrail: warn but don't reject
            return True
            
        return True
