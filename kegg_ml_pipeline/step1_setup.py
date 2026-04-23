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
    """Validate the local Python environment for the KEGG ML pipeline.

    This step is intentionally lightweight: it does not install anything,
    fetch remote data, or mutate caches beyond creating the directories
    that later pipeline stages expect to exist.
    """
    if sys.version_info < (3, 10):
        print("[ERROR] Python 3.10+ required")
        return False

    # Some plotting libraries attempt to create caches under the user home
    # directory during import. In sandboxed or shared environments those
    # locations may be read-only, so we redirect the caches into the project.
    cache_root = os.path.join(os.getcwd(), ".cache")
    os.makedirs(cache_root, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", cache_root)
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(cache_root, "matplotlib"))
    os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

    imported_modules = {}
    missing_packages: list[str] = []

    # Import checks are done one package at a time so the script can report
    # the full list of missing dependencies instead of failing on the first one.
    for package_name in PACKAGES:
        try:
            imported_modules[package_name] = importlib.import_module(package_name)
        except ImportError:
            missing_packages.append(package_name)

    # Later stages assume these output directories already exist.
    for directory in ("data", "results", "results/shap"):
        os.makedirs(directory, exist_ok=True)

    # Version reporting is useful when reproducing model behavior or debugging
    # library mismatches on another machine.
    print(f"Python: {sys.version.split()[0]}")
    print(f"numpy: {getattr(imported_modules.get('numpy'), '__version__', 'missing')}")
    print(
        f"xgboost: {getattr(imported_modules.get('xgboost'), '__version__', 'missing')}"
    )

    if missing_packages:
        # Keep the output machine-readable and compact so CI or wrapper scripts
        # can detect a setup failure from stdout/stderr easily.
        print(f"[ERROR] Missing: {', '.join(missing_packages)}")
        return False

    print("[OK] Setup complete")
    return True


if __name__ == "__main__":
    # Exit code 0/1 lets this script serve as a simple health check in shell
    # scripts or future pipeline orchestration.
    sys.exit(0 if run_setup() else 1)
