"""config_loader.py — load the global YAML configuration.

Reconstructed minimal version: loads config.yaml into a plain dict.
Used by run_exp101_reproducible.py (reads processed_dir / splits_dir / logs_dir).
"""
import os


def load_config(path):
    """Load a YAML config file and return it as a dict."""
    # Prefer PyYAML if available (standard in the project venv).
    try:
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except ImportError:
        # Fallback: very small YAML reader sufficient for this flat+nested config.
        return _mini_yaml(path)


def _mini_yaml(path):
    """Extremely small YAML parser for the specific config.yaml structure.

    Supports: nested section headers (2-space indent), key: value (scalars,
    ints, floats, bool, null), and inline flow lists [a, b, c].
    Does NOT aim to be a general YAML implementation.
    """
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    root = {}
    stack = [(-1, root)]  # (indent, dict)

    def coerce(v):
        v = v.strip()
        if v == "" or v == "null":
            return None
        if v.lower() == "true":
            return True
        if v.lower() == "false":
            return False
        if v.startswith("[") and v.endswith("]"):
            inner = v[1:-1].strip()
            if not inner:
                return []
            return [coerce(x) for x in inner.split(",")]
        try:
            return int(v)
        except ValueError:
            pass
        try:
            return float(v)
        except ValueError:
            pass
        return v.strip().strip('"').strip("'")

    for raw in lines:
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        content = raw.strip()
        if content.endswith(":"):
            key = content[:-1].strip()
            new_dict = {}
            while stack and stack[-1][0] >= indent:
                stack.pop()
            stack[-1][1][key] = new_dict
            stack.append((indent, new_dict))
        elif ":" in content:
            key, _, val = content.partition(":")
            key = key.strip()
            stack[-1][1][key] = coerce(val)
    return root
