import math
import re
from pathlib import Path
from typing import Optional, Tuple, Union


def parse_pbtxt(*, file_path: Union[str, Path] | None = None, content: str | None = None) -> dict:
    """Parses a triton pbtxt config file and returns the result as a dictionary

    Args:
        file_path: the path to the config file
        content: the content of the config file as a string

    Returns:
        configuration as a dictionary
    """
    assert file_path is not None or content is not None, "Either file_path or content must be provided"
    assert file_path is None or content is None, "Only one of file_path or content can be provided"

    if file_path:
        with open(file_path, "r") as f:
            content = f.readlines()
    else:
        content = content.splitlines()

    for i, line in enumerate(content):
        if "#" in line:
            new_line = line.split("#")[0]
            content[i] = new_line + "\n"

    content = "\n".join(content)

    sections = re.split(r"\n(?=\w)", content)
    parsed_data = {}

    for section in sections:
        data = parse_section(section.strip())
        if data is not None:
            key, value = data
            if key in parsed_data:
                if not isinstance(parsed_data[key], list):
                    parsed_data[key] = [parsed_data[key]]
                if isinstance(value, list):
                    parsed_data[key].extend(value)
                else:
                    parsed_data[key].append(value)
            else:
                parsed_data[key] = value

    return parsed_data


def parse_section(section: str) -> Optional[Tuple[str, Union[str, list, dict]]]:
    """Helper function for parsing section text"""
    if section.startswith("#") or section == "":
        return None

    s_idx, _ = first_sep_token(section)

    section_name, body = section[:s_idx], section[s_idx:].strip(":")
    return section_name.strip(), parse_section_body(body)


def parse_section_body(body: str):
    """Helper function for parsing section body data"""
    text = body.strip()

    if text[0] == "{" and text[-1] == "}":
        text = text[1:-1]
        result = {}
        while len(text) != 0:
            s_idx, token = first_sep_token(text)
            key, value = text[:s_idx].strip(",").strip(), text[s_idx:].strip(":").strip()
            if value[0] == "{":
                matching_idx = find_matching_brace(value, s_index=0)
                value, text = value[0 : matching_idx + 1], value[matching_idx + 1 :]
            elif value[0] == "[":
                matching_idx = find_matching_brace(value, s_index=0, l_char="[", r_char="]")
                value, text = value[0 : matching_idx + 1], value[matching_idx + 1 :]
            elif ":" in value:
                value, text = value.split(None, maxsplit=1)
            else:
                text = ""
            result[key.strip()] = parse_section_body(value)
        return result
    elif text[0] == "[" and text[-1] == "]" and "{" not in text:
        text = text[1:-1].strip()
        result = []
        if text:
            for t in text.split(","):
                result.append(parse_section_body(t))
        return result
    elif text[0] == "[" and text[-1] == "]" and "{" in text:
        text = text[1:-1].strip()
        result = []
        while len(text) != 0:
            matching_idx = find_matching_brace(text, s_index=0)
            result.append(parse_section_body(text[0 : matching_idx + 1]))
            text = text[matching_idx + 1 :].strip("\n\t ,")
        return result
    else:
        # parse literal
        try:
            return int(text.strip())
        except ValueError:
            return text.replace('"', "")


def first_sep_token(text: str, sep_tokens: tuple[str, ...] = (":", "{", "[")) -> tuple[int, str]:
    """Finds the first occurrence of any of the separator tokens in the text and returns its index and the token"""
    indices = [text.find(token) for token in sep_tokens if text.find(token) != -1]
    if not indices:
        raise ValueError(f'No separator tokens {sep_tokens} found in the text: "{text}"')
    min_index = min(indices)
    return min_index, text[min_index]


def find_matching_brace(text: str, s_index: int = 0, l_char: str = "{", r_char: str = "}"):
    """Finds the properly matching curl brace '}' that matches the brace specified by s_index"""
    if text[s_index] != l_char:
        raise ValueError("There is no curly brace at s_index in the provided text")
    counter = 0
    i = s_index + 1
    while i < len(text):
        if text[i] == r_char and counter == 0:
            return i
        elif text[i] == r_char:
            counter -= 1
        elif text[i] == l_char:
            counter += 1
        i += 1
    raise ValueError("There is no matching brace")
