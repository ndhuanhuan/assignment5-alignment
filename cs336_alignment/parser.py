import re
from typing import Any

def parse_mmlu_response(
    mmlu_example: dict[str, Any],
    model_output: str,
) -> str | None:
    """
    Given an MMLU example and a model output, parse the model output into a
    predicted option letter (i.e., 'A', 'B', 'C', or 'D'). If the model output
    cannot be parsed into a prediction option letter, return None.

    mmlu_example: dict[str, Any]
        Dictionary with an MMLU example. Contains the following keys:
        - "subject": str with the subject of the question.
        - "question": str with the text of the question.
        - "options": list[str] with the four answer options (in order).
                     The first option refers to letter "A", the second to "B", etc.
        - "answer": str with the option of the correct answer (e.g., "A")
    model_output: str
        str with the model's output to the MMLU example.

    Returns:
        str (one of "A", "B", "C", or "D") if the model output can be parsed into a prediction,
        else None.
    """
    
    # Clean the model output by stripping whitespace
    output = model_output.strip()
    
    # Strategy 1: Look for explicit answer patterns like "The answer is A" or "Answer: B"
    answer_patterns = [
        r'(?:the\s+)?answer\s+is\s+([A-D])',
        r'answer\s*:\s*([A-D])',
        r'correct\s+answer\s+is\s+([A-D])',
        r'correct\s+option\s+is\s+([A-D])',
        r'the\s+correct\s+answer\s+is\s+([A-D])',
        r'therefore\s*,?\s*([A-D])',
        r'so\s+the\s+answer\s+is\s+([A-D])',
    ]
    
    for pattern in answer_patterns:
        match = re.search(pattern, output, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    
    # Strategy 2: Look for patterns like "(A)", "[A]", or "A)"
    bracket_patterns = [
        r'\(([A-D])\)',
        r'\[([A-D])\]',
        r'([A-D])\)',
    ]
    
    for pattern in bracket_patterns:
        match = re.search(pattern, output, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    
    # Strategy 3: Look for standalone letters A, B, C, D at the end of the output
    # This handles cases where the model just outputs the letter
    end_letter_match = re.search(r'\b([A-D])\b\s*$', output, re.IGNORECASE)
    if end_letter_match:
        return end_letter_match.group(1).upper()
    
    # Strategy 4: Look for the first occurrence of A, B, C, or D in the output
    # This is a fallback for cases where the model mentions the letter somewhere
    first_letter_match = re.search(r'\b([A-D])\b', output, re.IGNORECASE)
    if first_letter_match:
        return first_letter_match.group(1).upper()
    
    # If none of the strategies work, return None
    return None

# uv run pytest -k test_parse_gsm8k_response
def parse_gsm8k_response(
    model_output: str,
) -> str | None:
    """
    Given a GSM8K model output, parse the model output into a predicted numeric answer by
    taking the last number that occurs in the output.

    model_output: str
        str with the model's output to a GSM8K example.

    Returns:
        str with the predicted numeric answer if the model output can be parsed into a prediction,
        else None.
    """
    # Clean the model output by stripping whitespace
    output = model_output.strip()
    
    # Find all numbers (integers and decimals) in the output
    # This pattern matches:
    # - Optional negative sign
    # - One or more digits
    # - Optional decimal point followed by one or more digits
    # - Handles comma-separated numbers like 1,000
    number_pattern = r'-?\d+(?:,\d+)*(?:\.\d+)?'
    
    # Find all matches
    matches = re.findall(number_pattern, output)
    
    if not matches:
        return None
    
    # Take the last number found
    last_number = matches[-1]
    
    # Remove commas from the number (e.g., "1,000" -> "1000")
    last_number = last_number.replace(',', '')
    
    return last_number
    