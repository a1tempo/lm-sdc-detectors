"""Smoke test: every module in the skeleton imports."""

import importlib
import pkgutil

import sdc

# Optional extras (pyproject [lm], [tflite]): a module may need one that isn't installed here.
OPTIONAL = {"llama_cpp", "ai_edge_litert", "litert_torch"}


def test_all_modules_import():
    for mod in pkgutil.walk_packages(sdc.__path__, prefix="sdc."):
        try:
            importlib.import_module(mod.name)
        except ModuleNotFoundError as error:
            if error.name not in OPTIONAL:
                raise
