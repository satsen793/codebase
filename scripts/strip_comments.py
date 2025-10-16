import os
import io
import re
import sys
import tokenize
from typing import List


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def strip_python_comments(code: str) -> str:
    """
    Remove only comment tokens from Python source while preserving strings and formatting.
    Does NOT remove docstrings or string literals.
    """
    out_tokens: List[tokenize.TokenInfo] = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(code).readline):
            if tok.type == tokenize.COMMENT:
                               
                continue
            out_tokens.append(tok)
        return tokenize.untokenize(out_tokens)
    except tokenize.TokenError:
                                                                              
        return code


def process_python_file(path: str) -> bool:
    try:
        with open(path, "r", encoding="utf-8") as f:
            original = f.read()
        stripped = strip_python_comments(original)
        if stripped != original:
            with open(path, "w", encoding="utf-8", newline="") as f:
                f.write(stripped)
            return True
        return False
    except Exception:
        return False


def strip_markdown_html_comments(text: str) -> str:
                                                      
    return re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)


def process_markdown_file(path: str) -> bool:
    try:
        with open(path, "r", encoding="utf-8") as f:
            original = f.read()
        stripped = strip_markdown_html_comments(original)
        if stripped != original:
            with open(path, "w", encoding="utf-8", newline="") as f:
                f.write(stripped)
            return True
        return False
    except Exception:
        return False


def process_requirements_file(path: str) -> bool:
    """
    Remove full-line comments (lines starting with optional whitespace then '#').
    Avoid stripping inline fragments that may be part of URLs (e.g., VCS #egg=...).
    """
    try:
        changed = False
        out_lines: List[str] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if re.match(r"^\s*#", line):
                    changed = True
                    continue
                out_lines.append(line)
        if changed:
            with open(path, "w", encoding="utf-8", newline="") as f:
                f.writelines(out_lines)
        return changed
    except Exception:
        return False


def main() -> int:
    modified_files: List[str] = []
    script_path = os.path.abspath(__file__)
    for dirpath, dirnames, filenames in os.walk(ROOT):
        parts = {p.lower() for p in dirpath.split(os.sep)}
        if any(p in parts for p in {".git", "__pycache__", ".venv", "venv"}):
            continue
        for name in filenames:
            path = os.path.join(dirpath, name)
            lower = name.lower()
            try:
                # Avoid processing this script itself
                if os.path.abspath(path) == script_path:
                    continue
                if lower.endswith(".py"):
                    if process_python_file(path):
                        modified_files.append(path)
                elif lower == "readme.md":
                    if process_markdown_file(path):
                        modified_files.append(path)
                elif lower == "requirements.txt":
                    if process_requirements_file(path):
                        modified_files.append(path)
            except Exception:
                pass

    print(f"Modified {len(modified_files)} file(s).")
    for f in modified_files:
        print(f" - {os.path.relpath(f, ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
