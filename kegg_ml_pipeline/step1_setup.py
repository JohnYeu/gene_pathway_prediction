from __future__ import annotations

import importlib
import os
import sys


PACKAGES = [
    "xgboost",
    "shap",
    "sklearn",
    "pandas",
    "numpy",
    "requests",
    "matplotlib",
    "seaborn",
    "scipy",
    "tqdm",
]


def run_setup() -> bool:
    if sys.version_info < (3, 10):
        print("[ERROR] Python 3.10+ required")
        return False

    cache_root = os.path.join(os.getcwd(), ".cache")
    os.makedirs(cache_root, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", cache_root)
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(cache_root, "matplotlib"))
    os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

    imported_modules = {}
    missing_packages: list[str] = []

    for package_name in PACKAGES:
        try:
            imported_modules[package_name] = importlib.import_module(package_name)
        except ImportError:
            missing_packages.append(package_name)

    for directory in ("data", "results", "results/shap"):
        os.makedirs(directory, exist_ok=True)

    print(f"Python: {sys.version.split()[0]}")
    print(f"numpy: {getattr(imported_modules.get('numpy'), '__version__', 'missing')}")
    print(
        f"xgboost: {getattr(imported_modules.get('xgboost'), '__version__', 'missing')}"
    )

    if missing_packages:
        print(f"[ERROR] Missing: {', '.join(missing_packages)}")
        return False

    print("[OK] Setup complete")
    return True


if __name__ == "__main__":
    sys.exit(0 if run_setup() else 1)
