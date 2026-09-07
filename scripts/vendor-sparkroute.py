#!/usr/bin/env python3
"""Import and verify the commit-pinned first-party SparkRoute plugin snapshot."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
import tomllib
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "vendor" / "sparkroute.lock"
SOURCE_DESTINATION = ROOT / "src" / "sparkrun" / "plugins" / "sparkroute"
TEST_DESTINATION = ROOT / "tests" / "vendor" / "sparkroute"
PROVENANCE_PATH = SOURCE_DESTINATION / "VENDORED.toml"
_IGNORED_PARTS = {"__pycache__", ".pytest_cache", ".ruff_cache"}
_UPSTREAM_SOURCE = "src/sparkrun/plugins/sparkroute"
_UPSTREAM_TESTS = "tests/test_sparkroute_*.py"


class VendorError(RuntimeError):
    """A vendor import or verification failed."""


@dataclass(frozen=True)
class PluginManifest:
    name: str
    version: str
    module: str
    feature: str
    repository: str
    sparkrun: str
    source: str
    tests: str


@dataclass(frozen=True)
class ImportedFile:
    path: str
    sha256: str


def _run_git(repository: Path, *args: str, text: bool = True) -> str | bytes:
    try:
        result = subprocess.run(
            ["git", "-C", str(repository), *args],
            check=True,
            capture_output=True,
            text=text,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        detail = getattr(error, "stderr", None)
        if isinstance(detail, bytes):
            detail = detail.decode("utf-8", errors="replace")
        raise VendorError("git %s failed%s" % (" ".join(args), ": %s" % detail.strip() if detail else "")) from error
    return result.stdout


@contextmanager
def _repository(source: str) -> Iterator[Path]:
    local = Path(source).expanduser()
    if local.exists():
        repository = local.resolve()
        _run_git(repository, "rev-parse", "--git-dir")
        yield repository
        return

    with tempfile.TemporaryDirectory(prefix="sparkrun-sparkroute-source-") as temporary:
        repository = Path(temporary) / "repository"
        try:
            subprocess.run(
                ["git", "clone", "--quiet", "--filter=blob:none", "--no-checkout", source, str(repository)],
                check=True,
            )
        except (OSError, subprocess.CalledProcessError) as error:
            raise VendorError("could not clone SparkRoute plugin source %s" % source) from error
        yield repository


def _manifest(repository: Path, commit: str) -> PluginManifest:
    raw = _run_git(repository, "show", "%s:plugin.toml" % commit)
    assert isinstance(raw, str)
    project_raw = _run_git(repository, "show", "%s:pyproject.toml" % commit)
    assert isinstance(project_raw, str)
    try:
        values = tomllib.loads(raw)
        project_values = tomllib.loads(project_raw)
    except tomllib.TOMLDecodeError as error:
        raise VendorError("upstream plugin metadata is invalid: %s" % error) from error

    if values.get("schema") != 1:
        raise VendorError("upstream plugin.toml must declare schema = 1")
    required = ("name", "module", "feature", "repository", "sparkrun", "source", "tests")
    missing = [name for name in required if not isinstance(values.get(name), str) or not values[name]]
    if missing:
        raise VendorError("upstream plugin.toml has missing or invalid fields: %s" % ", ".join(missing))
    project = project_values.get("project")
    version = project.get("version") if isinstance(project, dict) else None
    if not isinstance(version, str) or not version:
        raise VendorError("upstream pyproject.toml must declare project.version")
    if "version" in values:
        raise VendorError("upstream plugin.toml must not duplicate the version managed by versions.yaml")
    if values["name"] != "sparkroute" or values["module"] != "sparkrun.plugins.sparkroute":
        raise VendorError("upstream manifest does not describe the in-tree SparkRoute module")
    if values["feature"] != "gateway.sparkroute":
        raise VendorError("upstream manifest does not use the gateway.sparkroute feature gate")
    if values["source"] != _UPSTREAM_SOURCE or values["tests"] != _UPSTREAM_TESTS:
        raise VendorError("upstream manifest requests an unsupported source or test export")

    for field in ("source", "tests"):
        candidate = Path(values[field])
        if candidate.is_absolute() or ".." in candidate.parts:
            raise VendorError("upstream plugin.toml %s must stay within the repository" % field)

    return PluginManifest(version=version, **{name: values[name] for name in required})


def _archive(repository: Path, commit: str, manifest: PluginManifest, destination: Path) -> None:
    test_root = str(Path(manifest.tests).parent)
    raw = _run_git(repository, "archive", "--format=tar", commit, manifest.source, test_root, text=False)
    assert isinstance(raw, bytes)
    try:
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as archive:
            archive.extractall(destination, filter="data")
    except (tarfile.TarError, OSError) as error:
        raise VendorError("could not extract the upstream source archive: %s" % error) from error


def _regular_files(root: Path) -> list[Path]:
    files: list[Path] = []
    if not root.exists():
        return files
    for path in root.rglob("*"):
        if any(part in _IGNORED_PARTS for part in path.parts) or path.suffix == ".pyc":
            continue
        if path.is_symlink():
            raise VendorError("vendored content may not contain symlinks: %s" % path)
        if path.is_file():
            files.append(path)
    return sorted(files)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _content_digest(files: Sequence[ImportedFile]) -> str:
    digest = hashlib.sha256()
    for item in sorted(files, key=lambda value: value.path):
        digest.update(item.path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(item.sha256.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _replace_directory(destination: Path, files: Mapping[Path, Path]) -> None:
    expected = {SOURCE_DESTINATION.resolve(), TEST_DESTINATION.resolve()}
    if destination.resolve() not in expected:
        raise VendorError("refusing to replace unexpected path %s" % destination)
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    for relative, source in sorted(files.items()):
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _toml_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _render_lock(
    *,
    manifest: PluginManifest,
    commit: str,
    tree: str,
    digest: str,
    files: Sequence[ImportedFile],
) -> str:
    lines = [
        "# Generated by scripts/vendor-sparkroute.py; do not edit by hand.",
        "schema = 1",
        "name = %s" % _toml_string(manifest.name),
        "version = %s" % _toml_string(manifest.version),
        "module = %s" % _toml_string(manifest.module),
        "feature = %s" % _toml_string(manifest.feature),
        "repository = %s" % _toml_string(manifest.repository),
        "commit = %s" % _toml_string(commit),
        "tree = %s" % _toml_string(tree),
        "sparkrun = %s" % _toml_string(manifest.sparkrun),
        "content_sha256 = %s" % _toml_string(digest),
    ]
    for item in sorted(files, key=lambda value: value.path):
        lines.extend(
            [
                "",
                "[[files]]",
                "path = %s" % _toml_string(item.path),
                "sha256 = %s" % _toml_string(item.sha256),
            ]
        )
    return "\n".join(lines) + "\n"


def _render_provenance(*, manifest: PluginManifest, commit: str, tree: str, digest: str) -> str:
    return "\n".join(
        [
            "# Generated by scripts/vendor-sparkroute.py; do not edit by hand.",
            "schema = 1",
            "repository = %s" % _toml_string(manifest.repository),
            "commit = %s" % _toml_string(commit),
            "tree = %s" % _toml_string(tree),
            "version = %s" % _toml_string(manifest.version),
            "content_sha256 = %s" % _toml_string(digest),
            "",
        ]
    )


def _load_lock() -> dict:
    if not LOCK_PATH.is_file():
        raise VendorError("%s does not exist; run the update command" % LOCK_PATH.relative_to(ROOT))
    try:
        values = tomllib.loads(LOCK_PATH.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise VendorError("could not read %s: %s" % (LOCK_PATH.relative_to(ROOT), error)) from error
    if values.get("schema") != 1 or not isinstance(values.get("files"), list):
        raise VendorError("%s has an unsupported schema" % LOCK_PATH.relative_to(ROOT))
    return values


def _is_hex_digest(value: object, length: int) -> bool:
    return isinstance(value, str) and len(value) == length and all(character in "0123456789abcdef" for character in value)


def verify() -> None:
    lock = _load_lock()
    for name in ("repository", "commit", "tree", "version", "content_sha256"):
        if not isinstance(lock.get(name), str) or not lock[name]:
            raise VendorError("vendor lock has missing or invalid field %s" % name)
    if not _is_hex_digest(lock["commit"], 40) or not _is_hex_digest(lock["tree"], 40):
        raise VendorError("vendor lock commit and tree must be full lowercase Git object IDs")

    expected: dict[str, str] = {}
    for value in lock["files"]:
        if not isinstance(value, dict) or not isinstance(value.get("path"), str) or not isinstance(value.get("sha256"), str):
            raise VendorError("vendor lock contains an invalid file record")
        relative = Path(value["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise VendorError("vendor lock contains an unsafe file path: %s" % value["path"])
        if value["path"] in expected:
            raise VendorError("vendor lock contains a duplicate file path: %s" % value["path"])
        if not _is_hex_digest(value["sha256"], 64):
            raise VendorError("vendor lock contains an invalid SHA-256 for %s" % value["path"])
        expected[value["path"]] = value["sha256"]

    actual_paths: set[str] = set()
    for destination in (SOURCE_DESTINATION, TEST_DESTINATION):
        for path in _regular_files(destination):
            if path == PROVENANCE_PATH:
                continue
            relative = path.relative_to(ROOT).as_posix()
            actual_paths.add(relative)
            wanted = expected.get(relative)
            if wanted is None:
                raise VendorError("unrecorded file in vendored snapshot: %s" % relative)
            actual = _sha256(path)
            if actual != wanted:
                raise VendorError("vendored file differs from its lock: %s" % relative)

    missing = sorted(set(expected) - actual_paths)
    if missing:
        raise VendorError("vendored snapshot is missing: %s" % ", ".join(missing))

    imported = [ImportedFile(path=path, sha256=digest) for path, digest in expected.items()]
    digest = _content_digest(imported)
    if digest != lock["content_sha256"]:
        raise VendorError("vendor lock content digest does not match its file records")

    try:
        provenance = tomllib.loads(PROVENANCE_PATH.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise VendorError("could not read packaged SparkRoute provenance: %s" % error) from error
    for name in ("repository", "commit", "tree", "version", "content_sha256"):
        if provenance.get(name) != lock[name]:
            raise VendorError("packaged SparkRoute provenance disagrees with the lock for %s" % name)

    print("SparkRoute vendor snapshot verified: %s at %s" % (lock["version"], lock["commit"]))


def update(*, source: str, revision: str, initial: bool, force: bool) -> None:
    if LOCK_PATH.exists():
        if initial:
            raise VendorError("--initial cannot be used when the vendor lock already exists")
        if not force:
            verify()
    elif not initial:
        raise VendorError("the first import requires --initial")

    with _repository(source) as repository:
        commit_raw = _run_git(repository, "rev-parse", "--verify", "%s^{commit}" % revision)
        tree_raw = _run_git(repository, "rev-parse", "%s^{tree}" % revision)
        assert isinstance(commit_raw, str) and isinstance(tree_raw, str)
        commit = commit_raw.strip()
        tree = tree_raw.strip()
        manifest = _manifest(repository, commit)

        with tempfile.TemporaryDirectory(prefix="sparkrun-sparkroute-import-") as temporary:
            extracted = Path(temporary)
            _archive(repository, commit, manifest, extracted)

            upstream_source = extracted / manifest.source
            source_files = _regular_files(upstream_source)
            if not source_files:
                raise VendorError("upstream source export is empty")
            source_mapping = {path.relative_to(upstream_source): path for path in source_files}

            test_files = sorted(extracted.glob(manifest.tests))
            if not test_files or any(not path.is_file() or path.is_symlink() for path in test_files):
                raise VendorError("upstream test export is empty or contains unsupported entries")
            test_mapping = {path.name: path for path in test_files}

            imported: list[ImportedFile] = []
            for relative, path in source_mapping.items():
                target = (SOURCE_DESTINATION / relative).relative_to(ROOT).as_posix()
                imported.append(ImportedFile(path=target, sha256=_sha256(path)))
            for relative, path in test_mapping.items():
                target = (TEST_DESTINATION / relative).relative_to(ROOT).as_posix()
                imported.append(ImportedFile(path=target, sha256=_sha256(path)))
            digest = _content_digest(imported)

            _replace_directory(SOURCE_DESTINATION, source_mapping)
            _replace_directory(TEST_DESTINATION, {Path(name): path for name, path in test_mapping.items()})

    PROVENANCE_PATH.write_text(
        _render_provenance(manifest=manifest, commit=commit, tree=tree, digest=digest),
        encoding="utf-8",
    )
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOCK_PATH.write_text(
        _render_lock(manifest=manifest, commit=commit, tree=tree, digest=digest, files=imported),
        encoding="utf-8",
    )
    verify()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("verify", help="verify the checked-in snapshot without network access")
    update_parser = commands.add_parser("update", help="import a particular upstream commit")
    update_parser.add_argument("--source", required=True, help="local checkout path or Git repository URL")
    update_parser.add_argument("--rev", required=True, help="commit, tag, or ref to resolve and pin")
    update_parser.add_argument("--initial", action="store_true", help="permit the first import before a lock exists")
    update_parser.add_argument("--force", action="store_true", help="replace a locally modified existing snapshot")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "verify":
            verify()
        else:
            update(source=args.source, revision=args.rev, initial=args.initial, force=args.force)
    except VendorError as error:
        print("error: %s" % error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
