"""Offline checks for gateway credential loading. No network and no real key."""
from __future__ import annotations

import importlib.util
import os
import socket
import sys
import types
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ENV_NAME = "NMR_GATEWAY_API_KEY"
FAKE_KEY = "unit-test-not-a-real-key"


def _block_network() -> None:
    def blocked(*_args, **_kwargs):
        raise AssertionError("network access is blocked in this test")

    socket.socket = blocked  # type: ignore[assignment]
    socket.create_connection = blocked  # type: ignore[assignment]


def _install_stubs() -> None:
    litellm = types.ModuleType("litellm")
    litellm.drop_params = True
    litellm.completion = lambda **_kwargs: None
    sys.modules["litellm"] = litellm

    rdkit = types.ModuleType("rdkit")
    chem = types.ModuleType("rdkit.Chem")
    rdlog = types.ModuleType("rdkit.RDLogger")
    rdlog.DisableLog = lambda *_a, **_k: None
    rdkit.Chem = chem
    rdkit.RDLogger = rdlog
    sys.modules["rdkit"] = rdkit
    sys.modules["rdkit.Chem"] = chem
    sys.modules["rdkit.RDLogger"] = rdlog


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_completion(calls: list):
    def fake(**kwargs):
        calls.append(kwargs)
        message = types.SimpleNamespace(
            content='{"final_answer":"Yes","confidence":0.5,"key_evidence":"synthetic"}',
            reasoning_content="",
        )
        choice = types.SimpleNamespace(message=message)
        usage = types.SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2)
        return types.SimpleNamespace(choices=[choice], usage=usage)

    return fake


class GatewayCredentialTests(unittest.TestCase):
    def setUp(self) -> None:
        _block_network()
        self._previous = os.environ.pop(ENV_NAME, None)

    def tearDown(self) -> None:
        if self._previous is None:
            os.environ.pop(ENV_NAME, None)
        else:
            os.environ[ENV_NAME] = self._previous

    def test_helper_missing_blank_and_present(self) -> None:
        from gateway_credentials import require_gateway_api_key

        os.environ.pop(ENV_NAME, None)
        with self.assertRaises(RuntimeError) as missing:
            require_gateway_api_key()
        self.assertIn(ENV_NAME, str(missing.exception))

        os.environ[ENV_NAME] = "   "
        with self.assertRaises(RuntimeError) as blank:
            require_gateway_api_key()
        self.assertIn(ENV_NAME, str(blank.exception))

        os.environ[ENV_NAME] = f"  {FAKE_KEY}  "
        self.assertEqual(require_gateway_api_key(), FAKE_KEY)

    def test_tracked_sources_have_no_embedded_sk_token(self) -> None:
        import re

        roots = [REPO / "scripts", REPO / "perturbation_generator", REPO]
        pattern = re.compile(r"sk-[A-Za-z0-9_\-]{8,}")
        offenders = []
        for root in roots:
            paths = [root] if root.is_file() else list(root.rglob("*.py"))
            if root == REPO:
                paths = [REPO / "gateway_credentials.py"]
            for path in paths:
                if "__pycache__" in path.parts or "rsi_runs" in path.parts:
                    continue
                text = path.read_text(encoding="utf-8")
                if pattern.search(text):
                    offenders.append(str(path.relative_to(REPO)))
        self.assertEqual(offenders, [])

    def test_api_callers_require_env_and_pass_it(self) -> None:
        _install_stubs()
        os.environ[ENV_NAME] = FAKE_KEY
        cases = [
            (
                REPO / "scripts" / "evaluate_testset_api.py",
                "evaluate_testset_api",
                "score_one",
                ("unit-model", "synthetic user"),
            ),
            (
                REPO / "scripts" / "evaluate_rerank_fairnmr500_api.py",
                "evaluate_rerank_fairnmr500_api",
                "score_one",
                ("unit-model", "synthetic user"),
            ),
            (
                REPO / "scripts" / "evaluate_baselines.py",
                "evaluate_baselines",
                "call_model",
                ("unit-model", "instruction", "input"),
            ),
            (
                REPO / "scripts" / "distill_teacher_cot.py",
                "distill_teacher_cot",
                "call_teacher",
                ("unit-model", "synthetic user"),
            ),
            (
                REPO / "perturbation_generator" / "generate_cot.py",
                "generate_cot",
                "call_gemini",
                ({"candidate_smiles": "CCO", "c_nmr_peaks": [], "h_nmr_peaks": []},),
            ),
        ]
        for path, name, fn_name, args in cases:
            with self.subTest(module=name):
                module = _load(path, name)
                calls: list = []
                module.completion = _fake_completion(calls)
                os.environ.pop(ENV_NAME, None)
                with self.assertRaises(RuntimeError) as ctx:
                    getattr(module, fn_name)(*args)
                self.assertIn(ENV_NAME, str(ctx.exception))
                self.assertEqual(calls, [])

                os.environ[ENV_NAME] = FAKE_KEY
                getattr(module, fn_name)(*args)
                self.assertTrue(calls)
                self.assertEqual(calls[-1]["api_key"], FAKE_KEY)
                self.assertEqual(calls[-1]["api_base"], "https://ai-gateway-global.dp.tech")

    def test_real_task_api_path(self) -> None:
        _install_stubs()
        module = _load(REPO / "scripts" / "evaluate_real_task.py", "evaluate_real_task")
        calls: list = []
        sys.modules["litellm"].completion = _fake_completion(calls)
        args = types.SimpleNamespace(model="unit-model", workers=1)
        record = {
            "id": "synthetic-ethanol-001",
            "candidate_smiles": "CCO",
            "c_nmr_peaks": [{"delta (ppm)": 57.8}],
            "h_nmr_peaks": [{"delta": 1.18, "nH": 3, "category": "t"}],
        }
        os.environ.pop(ENV_NAME, None)
        with self.assertRaises(RuntimeError) as ctx:
            module.run_api(args, [record])
        self.assertIn(ENV_NAME, str(ctx.exception))
        self.assertEqual(calls, [])

        os.environ[ENV_NAME] = FAKE_KEY
        results = module.run_api(args, [record])
        self.assertEqual(results[0]["ans"], "Yes")
        self.assertEqual(calls[-1]["api_key"], FAKE_KEY)
        self.assertEqual(calls[-1]["api_base"], "https://ai-gateway-global.dp.tech")


if __name__ == "__main__":
    unittest.main()
