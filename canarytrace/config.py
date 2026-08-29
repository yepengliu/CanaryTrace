"""Central configuration and path resolution for CanaryTrace.

Everything is resolved relative to the repository root so the code contains no
machine-specific absolute paths. Override the root by setting the environment
variable ``CANARYTRACE_ROOT`` if you run the scripts from an unusual location.

Experiments are described by YAML files under ``configs/``. A config may inherit
from another via the ``_base_`` key; child values are deep-merged over the base.
Command-line ``--set section.key=value`` flags override any config value.
"""

import os
import sys
import json
from pathlib import Path

try:
    import yaml
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "PyYAML is required to load experiment configs. Install with `pip install pyyaml`."
    ) from e


# --------------------------------------------------------------------------- #
# Repository layout
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(os.environ.get("CANARYTRACE_ROOT", Path(__file__).resolve().parent.parent))

DATASETS_DIR = PROJECT_ROOT / "datasets"
RESULTS_DIR = PROJECT_ROOT / "results"
CONFIGS_DIR = PROJECT_ROOT / "configs"
THIRD_PARTY_DIR = PROJECT_ROOT / "third_party"
CONTRIEVER_DIR = THIRD_PARTY_DIR / "contriever"


def setup_import_paths():
    """Put the Contriever checkout on ``sys.path`` so its modules import cleanly.

    ``third_party/contriever`` is not vendored -- clone it as described in the
    README. Only the retrieval stage needs it; every other stage imports nothing
    from outside this package.
    """
    for p in (
        CONTRIEVER_DIR,
        CONTRIEVER_DIR / "src",
    ):
        sp = str(p)
        if sp not in sys.path:
            sys.path.insert(0, sp)


def resolve_path(path):
    """Resolve a possibly-relative config path against the repository root."""
    if path is None:
        return None
    p = Path(path)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def _load_dotenv():
    """Load simple KEY=VALUE lines from a repo-root .env into os.environ (no overwrite)."""
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def get_openai_api_key(cfg=None):
    """OpenAI key from the environment / repo .env (preferred) or an explicit config value."""
    _load_dotenv()
    key = os.environ.get("OPENAI_API_KEY")
    if not key and cfg is not None:
        key = cfg.get("models", {}).get("openai_api_key") or None
    if not key:
        raise RuntimeError(
            "No OpenAI API key found. Set the OPENAI_API_KEY environment variable "
            "(preferred) or models.openai_api_key in your config."
        )
    return key


# --------------------------------------------------------------------------- #
# Config loading
# --------------------------------------------------------------------------- #
def _deep_merge(base, override):
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _coerce(value):
    """Best-effort scalar coercion for --set overrides (int/float/bool/json)."""
    try:
        return json.loads(value)
    except (ValueError, TypeError):
        return value


def load_config(name_or_path, overrides=None):
    """Load an experiment config, resolving ``_base_`` inheritance and overrides.

    ``name_or_path`` may be a bare name (``main_nfcorpus``), a name with extension,
    or a path to a YAML file. ``overrides`` is a list of ``section.key=value`` strings.
    """
    path = Path(name_or_path)
    if not path.suffix:
        path = CONFIGS_DIR / f"{name_or_path}.yaml"
    elif not path.is_absolute() and not path.exists():
        path = CONFIGS_DIR / path.name

    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    base_name = cfg.pop("_base_", None)
    if base_name:
        base_cfg = load_config(base_name)
        cfg = _deep_merge(base_cfg, cfg)

    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Bad --set override (expected section.key=value): {item!r}")
        dotted, value = item.split("=", 1)
        keys = dotted.split(".")
        node = cfg
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = _coerce(value)

    return cfg
