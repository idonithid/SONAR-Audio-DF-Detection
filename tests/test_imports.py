"""Smoke test: every public sonar module imports without side effects.

We import every module *except* the ones that load the XLSR-300M checkpoint
at import time (model.py, guided_model.py, small_guided_model.py); those
need fairseq + the checkpoint and are covered by integration tests.
"""
import importlib

import pytest


SAFE_MODULES = [
    "sonar",
    "sonar.srm_filters",
    "sonar.HFFM",
    "sonar.utils",
    "sonar.eval_metric_LA",
    "sonar.eval_metric_DF",
    "sonar.data",
    "sonar.data.rawboost",
]


@pytest.mark.parametrize("name", SAFE_MODULES)
def test_safe_import(name):
    importlib.import_module(name)
