import importlib
import pkgutil
import sys
from pathlib import Path

import pytest

# Racine du projet sur sys.path pour les imports du type `from prompt...`.
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def _try_import(module_name):
    try:
        importlib.import_module(module_name)
        return True, None
    except Exception as e:  # noqa: BLE001 - on veut le message d'erreur brut
        return False, str(e)


PACKAGES = [
    "app",
    "prompt",
    "chroma",
    "data_handler",
    "function_chunk",
    "pipelines",
]


@pytest.mark.parametrize("package", PACKAGES)
def test_import_all_modules(package):
    """Chaque sous-module de chaque package doit s'importer sans effet de bord."""
    package_path = ROOT_DIR / package
    assert package_path.exists(), f"Le dossier {package} n'existe pas."

    for module in pkgutil.walk_packages([str(package_path)], prefix=f"{package}."):
        ok, err = _try_import(module.name)
        assert ok, f"Echec d'import de {module.name} : {err}"
