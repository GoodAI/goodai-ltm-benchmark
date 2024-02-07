import json
import re
from json import JSONDecodeError
from typing import Any

_javascript_re = re.compile(r"^.*```javascript(.*)```.*$", re.MULTILINE | re.DOTALL)
_json_re = re.compile(r"^.*```(?:json)?(.*)```.*$", re.MULTILINE | re.DOTALL)


def sanitize_and_parse_json(content: str) -> Any:
    try:
        return json.loads(content)
    except JSONDecodeError:
        return _step1_sanitize_and_parse_json(content)


def _step1_sanitize_and_parse_json(content: str) -> Any:
    match_json = _json_re.search(content)
    match_js = _javascript_re.search(content)
    if match_js:
        # Check Javascript first
        stripped_json = match_js.group(1)
    elif match_json:
        stripped_json = match_json.group(1)
    else:
        stripped_json = content
    countdown_start = max(5, len(stripped_json) // 50)
    return _step2_sanitize_and_parse_json(stripped_json, countdown=countdown_start)


def _step2_sanitize_and_parse_json(content: str, countdown: int) -> Any:
    if countdown <= 0:
        raise ValueError(f"Unable to sanitize JSON after a number of iterations: {content}")
    content = content.strip()
    try:
        return json.loads(content)
    except JSONDecodeError as error:
        msg = error.msg
        if msg.startswith("Extra data"):
            return _step2_sanitize_and_parse_json(content[:error.pos], countdown=countdown - 1)
        elif msg.startswith("Expecting value"):
            if error.pos == 0:
                return _handle_extra_beginning_text(content, countdown=countdown)
            else:
                remain = content[error.pos:].lstrip()
                if remain.startswith("//"):
                    return _handle_extra_line_comment(content, error.pos, countdown=countdown)
                elif remain.startswith("..."):
                    return _handle_ellipsis(content, error.pos, countdown=countdown)
                elif remain.startswith("]"):
                    return _handle_comma_before_brace(content, error.pos, countdown=countdown)
                else:
                    return _handle_missing_seq(content, error.pos, '"', countdown=countdown)
        elif msg.startswith("Expecting property name enclosed in double quotes"):
            remain = content[error.pos:].lstrip()
            if remain.startswith("..."):
                return _handle_ellipsis(content, error.pos, countdown=countdown)
            elif remain.startswith("}"):
                return _handle_comma_before_brace(content, error.pos, countdown=countdown)
            elif remain.startswith("//"):
                return _handle_extra_line_comment(content, error.pos, countdown=countdown)
            else:
                raise
        elif msg.startswith("Expecting ',' delimiter"):
            return _handle_missing_seq(content, error.pos, ",", countdown=countdown)
        elif msg.startswith("Expecting ':' delimiter"):
            remain = content[error.pos:].lstrip()
            if remain.startswith("}") or remain.startswith(","):
                return _handle_missing_seq(content, error.pos, ": null", countdown=countdown)
            else:
                raise
        elif msg.startswith("Invalid control character at"):
            cc = content[error.pos]
            if ord(cc) in [10]:
                return _handle_extra_char(content, error.pos, countdown=countdown)
            else:
                raise
        elif msg.startswith("Expecting property name enclosed in double quotes"):
            remain = content[error.pos:]
            if remain.startswith("//"):
                return _handle_remove_to_line_break(content, error.pos, countdown=countdown)
            elif remain.startswith("}"):
                return _handle_comma_before_brace(content, error.pos, countdown=countdown)
            else:
                raise
        else:
            raise


def _handle_ellipsis(content: str, error_pos: int, countdown: int) -> Any:
    expected_text = "..."
    idx_ellipsis = content.find(expected_text, error_pos)
    if idx_ellipsis == -1:
        raise ValueError(f"Expected ellipsis at position {error_pos}: {content}")
    new_content = content[:idx_ellipsis] + content[idx_ellipsis + len(expected_text):]
    return _step2_sanitize_and_parse_json(new_content, countdown=countdown - 1)


def _handle_extra_line_comment(content: str, error_pos: int, countdown: int) -> Any:
    idx_comment = content.find("//", error_pos)
    if idx_comment == -1:
        raise ValueError(f"Expected line comment marker at position {error_pos}: {content}")
    idx_line_break = content.find('\n', idx_comment)
    if idx_line_break == -1:
        raise ValueError(f"Expected line break after position {idx_comment}: {content}")
    new_content = content[:idx_comment] + content[idx_line_break+1:]
    return _step2_sanitize_and_parse_json(new_content, countdown=countdown - 1)


def _handle_comma_before_brace(content: str, error_pos: int, countdown: int) -> Any:
    comma_idx = content.rfind(",", 0, error_pos + 1)
    if comma_idx == -1:
        raise ValueError(f"Unable to resolve JSON error at position {error_pos}: {content}")
    else:
        new_content = content[:comma_idx] + content[comma_idx + 1:]
        return _step2_sanitize_and_parse_json(new_content, countdown=countdown - 1)


def _handle_remove_to_line_break(content: str, error_pos: int, countdown: int) -> Any:
    lb_pos = content.find('\n', error_pos)
    if lb_pos == -1:
        raise ValueError(f"Unable to resolve JSON error at position {error_pos}: {content}")
    else:
        new_content = content[:error_pos] + content[lb_pos+1:]
        return _step2_sanitize_and_parse_json(new_content, countdown=countdown - 1)


def _handle_missing_seq(content: str, error_pos: int, sub: str, countdown: int) -> Any:
    new_content = content[:error_pos] + sub + content[error_pos:]
    return _step2_sanitize_and_parse_json(new_content, countdown=countdown - 1)


def _handle_extra_char(content: str, error_pos: int, countdown: int) -> Any:
    new_content = content[:error_pos] + content[(error_pos + 1):]
    return _step2_sanitize_and_parse_json(new_content, countdown=countdown - 1)


def _handle_extra_beginning_text(content: str, countdown: int) -> Any:
    idx_brace = content.find('{')
    idx_bracket = content.find('[')
    if idx_brace != -1 and idx_bracket != -1:
        start_at = idx_brace if idx_brace < idx_bracket else idx_bracket
    elif idx_brace != -1:
        start_at = idx_brace
    elif idx_bracket != -1:
        start_at = idx_bracket
    else:
        raise ValueError("Content provided is not JSON!")
    return _step2_sanitize_and_parse_json(content[start_at:], countdown=countdown - 1)
