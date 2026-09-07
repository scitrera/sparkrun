"""Host-side contracts for the commit-pinned SparkRoute vendor snapshot."""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path
import tomllib

import pytest


ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "vendor" / "sparkroute.lock"
PROVENANCE = ROOT / "src" / "sparkrun" / "plugins" / "sparkroute" / "VENDORED.toml"


@pytest.fixture
def vendor_module():
    path = ROOT / "scripts" / "vendor-sparkroute.py"
    spec = importlib.util.spec_from_file_location("_sparkrun_vendor_sparkroute_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(spec.name, None)


def _manifest_text(*, duplicate_version: bool = False) -> str:
    version = 'version = "0.1.0"\n' if duplicate_version else ""
    return (
        "schema = 1\n"
        'name = "sparkroute"\n' + version + 'module = "sparkrun.plugins.sparkroute"\n'
        'feature = "gateway.sparkroute"\n'
        'repository = "git@github.com:sparksq/sparkrun-sparkroute-plugin.git"\n'
        'sparkrun = ">=0.3.8,<0.4"\n'
        'source = "src/sparkrun/plugins/sparkroute"\n'
        'tests = "tests/test_sparkroute_*.py"\n'
    )


def test_importer_uses_the_generated_project_version(vendor_module, monkeypatch):
    def show(_repository, *arguments, **_kwargs):
        return '[project]\nversion = "0.1.0"\n' if arguments[-1].endswith(":pyproject.toml") else _manifest_text()

    monkeypatch.setattr(vendor_module, "_run_git", show)
    manifest = vendor_module._manifest(Path("unused"), "a" * 40)

    assert manifest.version == "0.1.0"


def test_importer_rejects_a_duplicate_manifest_version(vendor_module, monkeypatch):
    def show(_repository, *arguments, **_kwargs):
        return '[project]\nversion = "0.1.0"\n' if arguments[-1].endswith(":pyproject.toml") else _manifest_text(duplicate_version=True)

    monkeypatch.setattr(vendor_module, "_run_git", show)

    with pytest.raises(vendor_module.VendorError, match="must not duplicate"):
        vendor_module._manifest(Path("unused"), "a" * 40)


def test_sparkroute_vendor_snapshot_verifies_offline():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "vendor-sparkroute.py"), "verify"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "SparkRoute vendor snapshot verified" in result.stdout


def test_packaged_provenance_matches_the_repository_lock():
    lock = tomllib.loads(LOCK.read_text(encoding="utf-8"))
    provenance = tomllib.loads(PROVENANCE.read_text(encoding="utf-8"))

    assert lock["repository"] == "git@github.com:sparksq/sparkrun-sparkroute-plugin.git"
    assert len(lock["commit"]) == 40
    assert lock["module"] == "sparkrun.plugins.sparkroute"
    assert lock["feature"] == "gateway.sparkroute"
    for field in ("repository", "commit", "tree", "version", "content_sha256"):
        assert provenance[field] == lock[field]


def test_verifier_rejects_a_locally_modified_vendor_file(tmp_path: Path):
    checkout = tmp_path / "checkout"
    (checkout / "scripts").mkdir(parents=True)
    (checkout / "vendor").mkdir()
    shutil.copy2(ROOT / "scripts" / "vendor-sparkroute.py", checkout / "scripts" / "vendor-sparkroute.py")
    shutil.copy2(LOCK, checkout / "vendor" / "sparkroute.lock")
    shutil.copytree(
        ROOT / "src" / "sparkrun" / "plugins" / "sparkroute",
        checkout / "src" / "sparkrun" / "plugins" / "sparkroute",
    )
    shutil.copytree(ROOT / "tests" / "vendor" / "sparkroute", checkout / "tests" / "vendor" / "sparkroute")

    changed = checkout / "src" / "sparkrun" / "plugins" / "sparkroute" / "engine.py"
    changed.write_bytes(changed.read_bytes() + b"\n# local edit\n")
    result = subprocess.run(
        [sys.executable, str(checkout / "scripts" / "vendor-sparkroute.py"), "verify"],
        cwd=checkout,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "vendored file differs from its lock" in result.stderr
