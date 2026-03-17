"""
JSON parsing utilities with fallback strategies for LLM responses.
"""

import re
import json
import logging
from typing import Optional, List, Dict, Any, Union

logger = logging.getLogger(__name__)


def parse_json(text: str) -> Optional[Dict]:
    """
    Parse JSON object from LLM response with multiple fallback strategies.

    Strategies:
    1. Direct JSON parse
    2. Remove markdown code blocks and parse
    3. Regex extraction of JSON object
    4. Repair common JSON errors and retry
    5. Extract first valid JSON object

    Args:
        text: Raw text response from LLM

    Returns:
        Parsed JSON dict or None if all strategies fail
    """
    if not text or not text.strip():
        return None

    # Strategy 1: Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strategy 2: Remove markdown code blocks
    try:
        cleaned = re.sub(r'```json\s*|\s*```', '', text, flags=re.IGNORECASE)
        cleaned = cleaned.strip()
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Strategy 3: Find JSON
    try:
        repaired = repair_json(text)
        return json.loads(repaired)
    except (json.JSONDecodeError, Exception):
        pass

    # Strategy 5: Extract first valid JSON object
    try:
        # Try to find valid JSON by testing progressively larger substrings
        start_idx = text.find('{')
        if start_idx >= 0:
            depth = 0
            for i in range(start_idx, len(text)):
                if text[i] == '{':
                    depth += 1
                elif text[i] == '}':
                    depth -= 1
                    if depth == 0:
                        # Found matching closing brace
                        candidate = text[start_idx:i+1]
                        try:
                            return json.loads(candidate)
                        except json.JSONDecodeError:
                            continue
    except Exception:
        pass

    # Log full text for debugging truncated/incomplete JSON
    logger.warning(f"Failed to parse JSON from LLM response: {text[:200]}...")
    logger.debug(f"Full LLM response text (length={len(text)}): {repr(text)}")

    # Strategy 6: Handle truncated/incomplete JSON (missing closing brace)
    try:
        # Check if text starts with valid JSON object but is missing closing brace
        start_idx = text.find('{')
        if start_idx >= 0:
            # Try to find the last complete key-value pair and close the JSON
            # Pattern to match: "key": value (where value can be string, number, or nested object)
            candidate = text[start_idx:]

            # Count braces to check if incomplete
            open_count = candidate.count('{')
            close_count = candidate.count('}')

            if open_count > close_count:
                # JSON is incomplete - try to repair by adding missing closing braces
                braces_needed = open_count - close_count
                repaired = candidate + ('}' * braces_needed)
                try:
                    result = json.loads(repaired)
                    logger.debug(f"Successfully repaired incomplete JSON by adding {braces_needed} closing brace(s)")
                    return result
                except json.JSONDecodeError:
                    pass

            # Try another approach: extract valid key-value pairs manually
            # Look for the pattern: "selected": number, "confidence": number, "reason": "string"
            result = {}

            # Extract "selected" field
            selected_match = re.search(r'"selected"\s*:\s*(\d+)', candidate)
            if selected_match:
                result['selected'] = int(selected_match.group(1))

            # Extract "confidence" field
            confidence_match = re.search(r'"confidence"\s*:\s*(0?\.\d+|1\.0|1)', candidate)
            if confidence_match:
                result['confidence'] = float(confidence_match.group(1))

            # Extract "reason" field (handle both complete and truncated strings)
            # First try: capture everything from "reason": " to the closing }
            # This handles cases with unescaped quotes inside the value
            reason_match = re.search(r'"reason"\s*:\s*"(.+?)}', candidate, re.DOTALL)
            if reason_match:
                # Clean up: remove trailing quote if present, replace internal quotes
                reason_text = reason_match.group(1).rstrip().rstrip('"').replace('"', "'")
                result['reason'] = reason_text if reason_text else "local match"
            else:
                # Standard pattern fallback (for well-formed JSON)
                reason_match = re.search(r'"reason"\s*:\s*"([^"]*)"', candidate)
                if reason_match:
                    result['reason'] = reason_match.group(1)
                else:
                    result['reason'] = "local match"

            if result:
                logger.debug(f"Successfully extracted fields from incomplete JSON: {result}")
                return result
    except Exception as e:
        logger.debug(f"Strategy 6 (incomplete JSON repair) failed: {e}")

    return None


def parse_json_array(text: str, expected_count: Optional[int] = None) -> Optional[List[Dict]]:
    """
    Parse JSON array from LLM response with multiple fallback strategies.

    Args:
        text: Raw text response from LLM
        expected_count: Optional expected number of items (for validation)

    Returns:
        Parsed JSON array or None if all strategies fail
    """
    if not text or not text.strip():
        return None

    # Strategy 1: Direct array parse
    try:
        result = json.loads(text)
        if isinstance(result, list):
            if expected_count is None or len(result) == expected_count:
                return result
    except json.JSONDecodeError:
        pass

    # Strategy 2: Remove markdown and parse
    try:
        cleaned = re.sub(r'```json\s*|\s*```', '', text, flags=re.IGNORECASE)
        cleaned = cleaned.strip()
        result = json.loads(cleaned)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    # Strategy 3: Find array with regex
    try:
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            result = json.loads(match.group())
            if isinstance(result, list):
                return result
    except json.JSONDecodeError:
        pass

    # Strategy 4: Repair and parse
    try:
        repaired = repair_json(text)
        result = json.loads(repaired)
        if isinstance(result, list):
            return result
    except (json.JSONDecodeError, Exception):
        pass

    # Strategy 5: Extract individual objects and build array
    try:
        objects = []
        # Find all {...} blocks
        depth = 0
        start = None
        for i, char in enumerate(text):
            if char == '{':
                if depth == 0:
                    start = i
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0 and start is not None:
                    # Try to parse this object
                    candidate = text[start:i+1]
                    try:
                        obj = json.loads(candidate)
                        if isinstance(obj, dict):
                            objects.append(obj)
                    except json.JSONDecodeError:
                        pass
                    start = None

        if objects:
            if expected_count is None or len(objects) == expected_count:
                return objects
    except Exception:
        pass

    # Strategy 6: Regex pattern extraction (last resort)
    try:
        # Look for patterns like: {"key": "value", ...}
        pattern = r'\{[^{}]*\}'
        matches = re.findall(pattern, text)
        objects = []
        for match in matches:
            try:
                obj = json.loads(match)
                if isinstance(obj, dict):
                    objects.append(obj)
            except json.JSONDecodeError:
                continue

        if objects:
            return objects
    except Exception:
        pass

    logger.warning(f"Failed to parse JSON array from LLM response: {text[:200]}...")
    return None


def repair_json(text: str) -> str:
    """
    Attempt to repair common JSON errors in LLM responses.

    Common issues fixed:
    - Markdown code blocks
    - Trailing commas
    - Missing commas between objects
    - Extra text before/after JSON
    - Unescaped quotes in strings

    Args:
        text: Raw text that may contain malformed JSON

    Returns:
        Repaired text (may still be invalid JSON)
    """
    # Remove markdown code blocks
    text = re.sub(r'```json\s*|\s*```', '', text, flags=re.IGNORECASE)
    text = text.strip()

    # Find JSON boundaries (first [ or { to last ] or })
    start_match = re.search(r'[\[\{]', text)
    if start_match:
        text = text[start_match.start():]

    # Find last closing bracket/brace
    for i in range(len(text) - 1, -1, -1):
        if text[i] in ']}':
            text = text[:i + 1]
            break

    # Fix unescaped quotes inside string values
    # Pattern: "key": "value with "unescaped" quotes"}
    # Strategy: Find string values and escape inner quotes
    text = _repair_unescaped_quotes(text)

    # Fix trailing commas before closing brackets/braces
    text = re.sub(r',\s*([}\]])', r'\1', text)

    # Fix missing commas between objects (risky, may introduce errors)
    # Pattern: }{ or ][ without comma
    text = re.sub(r'\}\s*\{', '},{', text)
    text = re.sub(r'\]\s*\[', '],[', text)

    # Fix missing commas between object and array
    text = re.sub(r'\}\s*\[', '},[', text)
    text = re.sub(r'\]\s*\{', '],{', text)

    return text


def _repair_unescaped_quotes(text: str) -> str:
    """
    Repair unescaped quotes inside JSON string values.

    Handles cases like:
    {"reason": "match in 'hope' and "match in 'hope'" keywords"}

    The LLM sometimes generates quotes inside string values without escaping.
    """
    # Find all string value patterns: "key": "value"
    # We need to handle the case where value contains unescaped quotes

    result = []
    i = 0

    while i < len(text):
        # Look for the start of a string value (after a colon)
        if text[i] == ':':
            result.append(text[i])
            i += 1

            # Skip whitespace after colon
            while i < len(text) and text[i] in ' \t\n':
                result.append(text[i])
                i += 1

            # Check if this is a string value (starts with quote)
            if i < len(text) and text[i] == '"':
                result.append(text[i])  # Opening quote
                i += 1

                # Now find the actual end of the string
                # The real end is a quote followed by , } ] or end of text
                string_content = []
                while i < len(text):
                    if text[i] == '"':
                        # Check if this is the real end of the string
                        # Look ahead for , } ] or whitespace followed by these
                        j = i + 1
                        while j < len(text) and text[j] in ' \t\n':
                            j += 1

                        if j >= len(text) or text[j] in ',}]':
                            # This is the real closing quote
                            # Escape any quotes we collected in string_content
                            escaped_content = ''.join(string_content).replace('"', '\\"')
                            result.append(escaped_content)
                            result.append('"')  # Closing quote
                            i = j
                            break
                        else:
                            # This is an unescaped quote inside the string
                            string_content.append(text[i])
                            i += 1
                    elif text[i] == '\\' and i + 1 < len(text):
                        # Already escaped character - keep as is
                        string_content.append(text[i])
                        string_content.append(text[i + 1])
                        i += 2
                    else:
                        string_content.append(text[i])
                        i += 1
                else:
                    # Reached end without finding closing quote
                    escaped_content = ''.join(string_content).replace('"', '\\"')
                    result.append(escaped_content)
            else:
                continue  # Not a string value, continue
        else:
            result.append(text[i])
            i += 1

    return ''.join(result)


def extract_json_by_keys(text: str, expected_keys: List[str]) -> Optional[Dict]:
    """
    Extract JSON object by looking for expected keys using regex.

    This is a last-resort strategy when standard parsing fails.

    Args:
        text: Raw text response
        expected_keys: List of keys that should be in the JSON object

    Returns:
        Extracted dict or None
    """
    result = {}

    for key in expected_keys:
        # Look for "key": value patterns
        # This is very basic and may not work for complex nested structures
        pattern = rf'"{key}"\s*:\s*([^,}}\]]+)'
        match = re.search(pattern, text)
        if match:
            value_str = match.group(1).strip()

            # Try to parse the value
            # Remove quotes if present
            if value_str.startswith('"') and value_str.endswith('"'):
                result[key] = value_str[1:-1]
            # Try to parse as number
            elif value_str.replace('.', '').replace('-', '').isdigit():
                try:
                    result[key] = float(value_str) if '.' in value_str else int(value_str)
                except ValueError:
                    result[key] = value_str
            # Try to parse as boolean
            elif value_str.lower() in ('true', 'false'):
                result[key] = value_str.lower() == 'true'
            # Keep as string
            else:
                result[key] = value_str

    return result if result else None
