# module/utils/json_utils.py
"""
JSON Extraction Utilities.
Exportable Module - No external dependencies.
"""

import logging
import json
import re

logger = logging.getLogger(__name__)

def extract_json(text: str) -> dict:
    """
    Extract and parse JSON object from text.
    Supports Markdown code block removal and regex-based extraction.
    """
    if not text:
        return {}
        
    # 1. Strip Markdown
    cleaned = text.replace("```json", "").replace("```", "").strip()
    
    try:
        # 2. Try simple substring extraction (First { to Last })
        start_idx = cleaned.find("{")
        end_idx = cleaned.rfind("}")
        if start_idx != -1 and end_idx != -1:
            potential_json = cleaned[start_idx : end_idx + 1]
            return json.loads(potential_json)
        else:
            return json.loads(cleaned)
            
    except json.JSONDecodeError:
        pass
        
    # 3. Fallback: Regex-based extraction
    try:
        match = re.search(r'(\{.*\})', cleaned, re.DOTALL)
        if match:
            return json.loads(match.group(1))
    except Exception as e:
        logger.warning(f"Failed to extract JSON via regex: {e}")
        pass
        
    return {}

def extract_specific_key(text: str, key: str) -> str:
    """
    Extract specific key value using regex.
    Useful when full JSON parsing fails.
    """
    try:
        pattern = rf'"{key}"\s*:\s*"(.*?)(?<!\\)"'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1)
    except Exception:
        pass
    return ""
