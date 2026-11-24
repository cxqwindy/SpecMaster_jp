"""The v6.4 registry merge is idempotent and does not drop unrelated keys."""
from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "register_llamafactory_nmr_v6_4.py"
REGISTRY = REPO / "configs" / "llamafactory" / "dataset_info_nmr_v6_4.json"


def _load():
    spec = importlib.util.spec_from_file_location("register_llamafactory_nmr_v6_4", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class RegisterDatasetTests(unittest.TestCase):
    def test_add_then_leave_bytes_unchanged(self) -> None:
        module = _load()
        registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dataset_info.json"
            original = {
                "other_task": {"file_name": "other.json"},
                "smiles_repair_train": {"file_name": "smiles_repair_train.json"},
            }
            path.write_text(json.dumps(original, indent=2) + "\n", encoding="utf-8")
            first = module.apply_registry(path, REGISTRY, write=True, force=False)
            self.assertEqual(first["status"], "ok")
            self.assertTrue(first["wrote"])
            saved = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(saved["other_task"], original["other_task"])
            self.assertEqual(saved["smiles_repair_train"], original["smiles_repair_train"])
            for key, value in registry.items():
                self.assertEqual(saved[key], value)
            snapshot = path.read_bytes()
            second = module.apply_registry(path, REGISTRY, write=True, force=False)
            self.assertFalse(second["changed"])
            self.assertFalse(second["wrote"])
            self.assertEqual(path.read_bytes(), snapshot)

    def test_conflict_is_not_written(self) -> None:
        module = _load()
        registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
        key = next(iter(registry))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dataset_info.json"
            current = {key: {"file_name": "different.json"}, "keep": {"file_name": "keep.json"}}
            path.write_text(json.dumps(current) + "\n", encoding="utf-8")
            before = path.read_bytes()
            result = module.apply_registry(path, REGISTRY, write=True, force=False)
            self.assertEqual(result["status"], "conflict")
            self.assertFalse(result["wrote"])
            self.assertEqual(path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
