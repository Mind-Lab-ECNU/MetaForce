#!/usr/bin/env python3
"""Tests for v2/v4 overlay merge behavior."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path


TEST_DIR = Path(__file__).resolve().parent
ROOT_DIR = TEST_DIR.parent


def _install_dependency_stubs() -> None:
    if "pandas" not in sys.modules:
        pandas_stub = types.ModuleType("pandas")

        class _FakeDataFrame:
            def __init__(self, items):
                self._items = items

            def to_parquet(self, path, index=False):
                Path(path).write_text(json.dumps(self._items, ensure_ascii=False), encoding="utf-8")

        pandas_stub.DataFrame = _FakeDataFrame
        sys.modules["pandas"] = pandas_stub

    if "PIL" not in sys.modules or "PIL.Image" not in sys.modules:
        pil_stub = types.ModuleType("PIL")
        image_stub = types.ModuleType("PIL.Image")

        class _FakeImageHandle:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def verify(self):
                return None

        def _open(path):
            if not Path(path).exists():
                raise FileNotFoundError(path)
            return _FakeImageHandle()

        image_stub.open = _open
        image_stub.Image = object
        pil_stub.Image = image_stub
        sys.modules["PIL"] = pil_stub
        sys.modules["PIL.Image"] = image_stub


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_install_dependency_stubs()
overlay_module = _load_module(TEST_DIR / "merge_v2_v4_test_overlay.py", "merge_v2_v4_test_overlay")


def _write_sample(path: Path, data_source: str, image_path: str, question: str, answer: str) -> None:
    sample = [
        {
            "data_source": data_source,
            "prompt": [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": f"<image>{question}\n\nGuidelines: Use tools/skills only when helpful.",
                },
            ],
            "images": [{"image": image_path}],
            "reward_model": {"style": "rule", "ground_truth": answer},
            "extra_info": {
                "split": "test",
                "index": 0,
                "qid": f"{data_source}_0",
                "images": [image_path],
                "question": question,
            },
        }
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sample, f, ensure_ascii=False, indent=2)


class OverlayMergeV2V4Test(unittest.TestCase):
    def test_v4_dataset_path_overrides_v2_same_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            v2_ood = root / "test_merge_v2" / "OOD"
            v4_ood = root / "test_merge_v4" / "OOD"

            for path in [
                v2_ood / "CLEVR_MATH_test_OOD" / "test.json",
                v2_ood / "ScienceQA_test_OOD" / "test.json",
                v4_ood / "CLEVR_MATH_test_OOD" / "test.json",
            ]:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("[]", encoding="utf-8")

            merged, overridden = overlay_module._overlay_dataset_paths(v2_ood, v4_ood)

            self.assertEqual(overridden, ["CLEVR_MATH_test_OOD"])
            self.assertEqual(set(merged), {"CLEVR_MATH_test_OOD", "ScienceQA_test_OOD"})
            self.assertTrue(str(merged["CLEVR_MATH_test_OOD"]).startswith(str(v4_ood)))
            self.assertTrue(str(merged["ScienceQA_test_OOD"]).startswith(str(v2_ood)))

    def test_overlay_script_keeps_v2_and_replaces_matching_clevr_math(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "data_root"
            data_root.mkdir(parents=True)
            image_path = data_root / "img.png"
            image_path.write_bytes(b"not-a-real-png-but-good-enough-for-stub")

            v2_root = root / "out" / "test_merge_v2"
            v4_root = root / "out" / "test_merge_v4"

            _write_sample(
                v2_root / "ID" / "DocVQA_test_ID" / "test.json",
                "DocVQA_test_ID",
                str(image_path),
                "v2-id-question",
                "v2-id-answer",
            )
            _write_sample(
                v2_root / "OOD" / "CLEVR_MATH_test_OOD" / "test.json",
                "CLEVR_MATH_test_OOD",
                str(image_path),
                "v2-clevr-question",
                "v2-clevr-answer",
            )
            _write_sample(
                v2_root / "OOD" / "ScienceQA_test_OOD" / "test.json",
                "ScienceQA_test_OOD",
                str(image_path),
                "v2-science-question",
                "v2-science-answer",
            )
            _write_sample(
                v4_root / "OOD" / "CLEVR_MATH_test_OOD" / "test.json",
                "CLEVR_MATH_test_OOD",
                str(image_path),
                "v4-clevr-question",
                "v4-clevr-answer",
            )

            out_root = root / "out" / "test_merge_v2_overlay_v4"
            old_argv = sys.argv
            try:
                sys.argv = [
                    "merge_v2_v4_test_overlay.py",
                    "--data-root",
                    str(data_root),
                    "--output-base-dir",
                    str(root / "out"),
                    "--v2-test-merge-dir-name",
                    "test_merge_v2",
                    "--v4-test-merge-dir-name",
                    "test_merge_v4",
                    "--out-test-merge-dir-name",
                    "test_merge_v2_overlay_v4",
                    "--strict",
                ]
                overlay_module.main()
            finally:
                sys.argv = old_argv

            with open(out_root / "test_ood_merge_v2_overlay_v4.json", "r", encoding="utf-8") as f:
                ood = json.load(f)
            with open(out_root / "test_id_merge_v2_overlay_v4.json", "r", encoding="utf-8") as f:
                iid = json.load(f)

            self.assertEqual(len(iid), 1)
            self.assertEqual(iid[0]["data_source"], "DocVQA_test_ID")

            by_source = {item["data_source"]: item for item in ood}
            self.assertEqual(set(by_source), {"CLEVR_MATH_test_OOD", "ScienceQA_test_OOD"})
            self.assertEqual(
                by_source["CLEVR_MATH_test_OOD"]["reward_model"]["ground_truth"],
                "v4-clevr-answer",
            )
            self.assertEqual(
                by_source["ScienceQA_test_OOD"]["reward_model"]["ground_truth"],
                "v2-science-answer",
            )


if __name__ == "__main__":
    unittest.main()
