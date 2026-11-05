from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_package() -> None:
    """Import the plugin directory as ``hermes_colab`` without running register()."""
    if "hermes_colab" in sys.modules:
        return
    spec = importlib.util.spec_from_file_location(
        "hermes_colab", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
    module = importlib.util.module_from_spec(spec)
    sys.modules["hermes_colab"] = module
    spec.loader.exec_module(module)


_load_package()
