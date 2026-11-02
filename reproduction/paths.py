"""Portable archive/artifact roots and relocation of text-only manifest placeholders."""
import json
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASSET_ROOT = Path(os.environ.get('N2N_ASSET_ROOT', PROJECT_ROOT / 'artifacts')).expanduser().resolve()
PM_ROOT = Path(os.environ.get('N2N_PM_ROOT', PROJECT_ROOT / 'data' / 'PM')).expanduser().resolve()
CHECKPOINT_ROOT = Path(os.environ.get('N2N_CHECKPOINT_ROOT', ASSET_ROOT / 'checkpoints')).expanduser().resolve()
AT_MODEL_ROOT = Path(os.environ.get('N2N_AT_MODEL_ROOT', ASSET_ROOT / 'at_models')).expanduser().resolve()


def resolve_paths(value):
    """Expand declared placeholders only; never evaluate strings as code."""
    if isinstance(value, str):
        replacements = {'N2N_PROJECT_ROOT': PROJECT_ROOT, 'N2N_ASSET_ROOT': ASSET_ROOT,
                        'N2N_PM_ROOT': PM_ROOT, 'N2N_CHECKPOINT_ROOT': CHECKPOINT_ROOT,
                        'N2N_AT_MODEL_ROOT': AT_MODEL_ROOT}
        for name, path in replacements.items():
            value = value.replace('${' + name + '}', str(path))
        return value
    if isinstance(value, list):
        return [resolve_paths(item) for item in value]
    if isinstance(value, dict):
        return {resolve_paths(key): resolve_paths(item) for key, item in value.items()}
    return value


def loads_json(text):
    return resolve_paths(json.loads(text))
