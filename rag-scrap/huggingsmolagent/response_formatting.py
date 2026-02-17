import json
import re
from typing import Any


def extract_final_answer(llm_response: Any, safe_utf8_str) -> str:
    """Extracts the final user response from structured LLM output (Thought/Code/Observation/final_answer)."""
    if llm_response is None:
        return ""

    if not isinstance(llm_response, str):
        llm_response = str(llm_response)

    llm_response = safe_utf8_str(llm_response)

    match = re.search(r'final_answer\("([^\"]*)"\)', llm_response, re.DOTALL)
    if not match:
        match = re.search(r"final_answer\('([^']*)'\)", llm_response, re.DOTALL)

    if match:
        answer = match.group(1).strip()
        return format_json_response(answer)

    match = re.search(r"Out - Final answer: (.*?)$", llm_response, re.MULTILINE)
    if match:
        answer = match.group(1).strip()
        return format_json_response(answer)

    match = re.search(r"Final answer:\s*(.*?)$", llm_response, re.MULTILINE | re.IGNORECASE)
    if match:
        answer = match.group(1).strip()
        return format_json_response(answer)

    lines = [line for line in llm_response.strip().split("\n") if line.strip()]
    if lines:
        clean_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith(("ActionStep(", "MessageRole.", "<", "Calling tools:", "Execution logs:")):
                continue
            if stripped.startswith(("Out:", "In:", ">>>")):
                continue
            clean_lines.append(stripped)

        if clean_lines:
            result = "\n".join(clean_lines)
            return format_json_response(result)

    return "Unable to extract final answer"


def format_json_response(response: str) -> str:
    """Formats JSON responses into human-readable text using a universal approach."""
    if not response:
        return response

    response = response.strip()
    if response.startswith("{") and response.endswith("}"):
        try:
            data = json.loads(response)
            return format_any_json(data, "Information")
        except (json.JSONDecodeError, ValueError):
            pass
    elif response.startswith("[") and response.endswith("]"):
        try:
            data = json.loads(response)
            return format_any_json(data, "Results")
        except (json.JSONDecodeError, ValueError):
            pass

    return response


def format_any_json(data: Any, title: str = "Data", level: int = 0) -> str:
    """Universal JSON formatter that works with any structure."""
    if data is None:
        return "No data available."

    indent = "  " * level

    if isinstance(data, dict):
        if not data:
            return f"**{title}:** Empty"

        formatted = f"**{title}:**\n\n" if level == 0 else f"{indent}**{title}:**\n"

        for key, value in data.items():
            clean_key = key.replace("_", " ").replace("-", " ").title()

            if isinstance(value, dict):
                if value:
                    formatted += f"{indent}🔹 **{clean_key}:**\n"
                    formatted += format_any_json(value, "", level + 1)
                else:
                    formatted += f"{indent}🔹 **{clean_key}:** Empty\n"

            elif isinstance(value, list):
                if value:
                    formatted += f"{indent}🔹 **{clean_key}:** {len(value)} item(s)\n"
                    formatted += format_any_json(value, "", level + 1)
                else:
                    formatted += f"{indent}🔹 **{clean_key}:** Empty list\n"

            else:
                value_str = str(value)
                if len(value_str) > 100:
                    value_str = value_str[:100] + "..."
                formatted += f"{indent}🔹 **{clean_key}:** {value_str}\n"

        return formatted

    if isinstance(data, list):
        if not data:
            return f"{indent}No items found.\n"

        formatted = ""

        for i, item in enumerate(data):
            if i >= 15:
                remaining = len(data) - i
                formatted += f"{indent}... and {remaining} more item(s)\n"
                break

            if isinstance(item, dict):
                if item:
                    formatted += f"{indent}**{i+1}.** "

                    important_keys = []
                    for key in item.keys():
                        if any(word in key.lower() for word in ["name", "title", "team", "home", "away"]):
                            important_keys.append(key)

                    other_keys = [k for k in item.keys() if k not in important_keys]
                    all_keys = important_keys + other_keys

                    key_values = []
                    for key in all_keys[:3]:
                        clean_key = key.replace("_", " ").replace("-", " ").title()
                        value = item[key]
                        value_str = str(value)
                        if len(value_str) > 50:
                            value_str = value_str[:50] + "..."
                        key_values.append(f"{clean_key}: {value_str}")

                    formatted += " | ".join(key_values)

                    if len(item) > 3:
                        formatted += f" | (+{len(item)-3} more fields)"

                    formatted += "\n"
                else:
                    formatted += f"{indent}**{i+1}.** Empty item\n"

            elif isinstance(item, list):
                formatted += f"{indent}**{i+1}.** List with {len(item)} items\n"

            else:
                item_str = str(item)
                if len(item_str) > 100:
                    item_str = item_str[:100] + "..."
                formatted += f"{indent}**{i+1}.** {item_str}\n"

        return formatted

    return f"{indent}{str(data)}\n"
