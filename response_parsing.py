"""Response parsers retained from the original 2025 benchmark."""
from typing import Any, Optional, List, Dict
import json
import re
from loguru import logger

def _remove_thinking_tags(text: str) -> str:
    """Remove <think>...</think> tags from text (reasoning model output).

    Args:
        text: Raw text that might contain thinking tags.

    Returns:
        Text with thinking tags removed.
    """
    import re

    if not text:
        return text

    # Remove <think>...</think> blocks (case insensitive, multi-line)
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.IGNORECASE | re.DOTALL)

    # Clean up any extra whitespace left behind
    cleaned = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned)

    return cleaned.strip()


def _parse_json_from_text(text: str) -> Any:
    """Attempt to parse and return a JSON object from arbitrary model text.

    The model may wrap JSON in code fences or include explanatory prose. This helper
    attempts a few strategies to extract a valid JSON object or array. If parsing fails,
    it returns None.

    Args:
        text: The raw text produced by the model.

    Returns:
        The parsed JSON (dict or list) if found, otherwise None.
    """
    import re

    if not text:
        return None

    # 1) Direct parse
    try:
        return json.loads(text)
    except Exception:
        pass

    # 2) Fenced code block ```json ... ``` or ``` ... ```
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    if fence_match:
        candidate = fence_match.group(1).strip()
        try:
            return json.loads(candidate)
        except Exception:
            pass

    # 3) Heuristic: first '{' to last '}'
    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1 and last > first:
        candidate = text[first : last + 1]
        try:
            return json.loads(candidate)
        except Exception:
            pass

    # 4) Heuristic: first '[' to last ']'
    first = text.find("[")
    last = text.rfind("]")
    if first != -1 and last != -1 and last > first:
        candidate = text[first : last + 1]
        try:
            return json.loads(candidate)
        except Exception:
            pass

    return None


def _parse_markdown_table_from_text(text: str) -> Optional[List[Dict[str, str]]]:
    """Extract and parse a Markdown table from a string.

    Args:
        text: The string containing the Markdown table.

    Returns:
        A list of dictionaries representing the table rows, or None if no table is found.
    """
    lines = text.strip().split('\n')

    # Find header and separator
    header_index = -1
    for i, line in enumerate(lines):
        if '|' in line and '---' in lines[i+1]:
            header_index = i
            break

    if header_index == -1:
        return None

    headers = [h.strip().lower().replace(' ', '_') for h in lines[header_index].split('|') if h.strip()]

    table_rows = []
    for i in range(header_index + 2, len(lines)):
        line = lines[i].strip()
        if not line.startswith('|') or not line.endswith('|'):
            break # End of table

        values = [v.strip() for v in line.split('|')][1:-1] # Remove empty start/end strings
        if len(values) == len(headers):
            # Map values to a more generic schema if possible
            row_data = dict(zip(headers, values))
            mapped_row = {
                "item": row_data.get("item", ""),
                "price_huf": row_data.get("price_(huf)", row_data.get("price_huf", "")),
                "link": row_data.get("link", ""),
                "source": row_data.get("source", "")
            }
            table_rows.append(mapped_row)

    return table_rows if table_rows else None
