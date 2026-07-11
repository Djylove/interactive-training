"""Minimal dependency-free test runner (the container has no pytest/pip)."""
from __future__ import annotations

import importlib
import inspect
import tempfile
import traceback
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

MODULES = ["tests.test_core", "tests.test_transport", "tests.test_agents", "tests.test_recipes"]


def main() -> int:
    passed, failed = 0, []
    for mod_name in MODULES:
        mod = importlib.import_module(mod_name)
        for name, fn in sorted(vars(mod).items()):
            if not (name.startswith("test_") and callable(fn)):
                continue
            kwargs = {}
            tmp = None
            if "tmp_path" in inspect.signature(fn).parameters:
                tmp = tempfile.TemporaryDirectory()
                kwargs["tmp_path"] = Path(tmp.name)
            try:
                fn(**kwargs)
                passed += 1
                print(f"PASS {mod_name}.{name}")
            except Exception:
                failed.append(f"{mod_name}.{name}")
                print(f"FAIL {mod_name}.{name}")
                traceback.print_exc()
            finally:
                if tmp is not None:
                    tmp.cleanup()
    print(f"\n{passed} passed, {len(failed)} failed")
    for f in failed:
        print("  FAILED:", f)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
