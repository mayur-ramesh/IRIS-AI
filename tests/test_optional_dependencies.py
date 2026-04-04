"""Tests for graceful startup when optional ML dependencies are unavailable."""

import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import iris_mvp


class TestOptionalTransformersDependency(unittest.TestCase):
    def test_finbert_can_be_disabled_explicitly(self):
        """IRIS_System should honor the explicit FinBERT feature flag."""
        with patch.object(iris_mvp, "FINBERT_ENABLED", False), \
             patch.object(iris_mvp, "NEWS_API_KEY", None), \
             patch.object(iris_mvp, "WEBZ_API_KEY", None), \
             patch.object(iris_mvp.IRIS_System, "merge_alias_reports", return_value=None):
            system = iris_mvp.IRIS_System()

        self.assertIsNone(system.sentiment_analyzer)
        self.assertFalse(system.finbert_status["enabled"])
        self.assertEqual(system.finbert_status["reason"], "disabled via IRIS_ENABLE_FINBERT")

    def test_iris_system_initializes_without_transformers_stack(self):
        """IRIS_System should still initialize when transformers/PyTorch are unavailable."""
        with patch.object(iris_mvp, "_TRANSFORMERS_AVAILABLE", False), \
             patch.object(iris_mvp, "_TRANSFORMERS_IMPORT_ERROR", ImportError("PyTorch not found")), \
             patch.object(iris_mvp, "NEWS_API_KEY", None), \
             patch.object(iris_mvp, "WEBZ_API_KEY", None), \
             patch.object(iris_mvp.IRIS_System, "merge_alias_reports", return_value=None):
            system = iris_mvp.IRIS_System()

        self.assertIsNone(system.sentiment_analyzer)
        self.assertFalse(system.finbert_status["ready"])
        self.assertIn("PyTorch not found", system.finbert_status["reason"])

    def test_app_imports_when_transformers_stack_is_missing(self):
        """The Flask app module should still import when FinBERT dependencies are unavailable."""
        project_root = Path(__file__).resolve().parents[1]
        script = """
import builtins

real_import = builtins.__import__

def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "transformers":
        raise ImportError("PyTorch not found")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = fake_import
try:
    import app
    print("APP_IMPORTED", app.iris_app is not None)
finally:
    builtins.__import__ = real_import
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(
            result.returncode,
            0,
            msg=f"App import failed.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}",
        )
        self.assertIn("APP_IMPORTED True", result.stdout)


if __name__ == "__main__":
    unittest.main()
