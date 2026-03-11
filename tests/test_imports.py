import importlib
import pkgutil
from pathlib import Path


def iter_python_modules(package_root: Path):
    for module_info in pkgutil.walk_packages([str(package_root)]):
        if not module_info.ispkg:
            yield module_info.name


def test_all_src_modules_importable():
    """
    Basic smoke test that all modules under src/ can be imported.
    This helps catch syntax errors or missing dependencies early in CI.
    """
    src_root = Path(__file__).resolve().parents[1] / "src"
    assert src_root.is_dir(), f"Expected src directory at {src_root}"

    failed = []
    for module_name in iter_python_modules(src_root):
        dotted = f"src.{module_name}"
        try:
            importlib.import_module(dotted)
        except Exception as exc:  # noqa: BLE001
            failed.append((dotted, repr(exc)))

    assert not failed, "Failed to import modules:\n" + "\n".join(
        f"{name}: {err}" for name, err in failed
    )

