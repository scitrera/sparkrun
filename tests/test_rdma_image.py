"""Drift guard between the RDMA test image's pin and the code that pulls it.

``docker/rdma-test/image.env`` is the one place the NCCL version — and so this
image's release cadence — is recorded. A Python default naming a tag nobody
built is worse than no default at all: the failure lands minutes into a run,
on the cluster, as an unexplained pull error.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from sparkrun.api.setup import DEFAULT_RDMA_TEST_IMAGE

REPO_ROOT = Path(__file__).resolve().parent.parent
IMAGE_ENV = REPO_ROOT / "docker" / "rdma-test" / "image.env"
DOCKERFILE = REPO_ROOT / "docker" / "rdma-test" / "Dockerfile"

needs_repo = pytest.mark.skipif(not IMAGE_ENV.is_file(), reason="requires the repo checkout")


def _load_env() -> dict[str, str]:
    values = {}
    for line in IMAGE_ENV.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key.strip()] = value
    return values


def _source_env() -> dict[str, str]:
    """Read image.env the way build.sh does — through real bash."""
    import json
    import shutil
    import subprocess

    if shutil.which("bash") is None:
        pytest.skip("requires bash")
    script = (
        "set -a; . %s; set +a; "
        "python3 -c \"import json,os;print(json.dumps({k:os.environ.get(k,'') "
        "for k in ['IMAGE_REPO','IMAGE_TAG','NCCL_VERSION','CUDA_VERSION','UBUNTU_VERSION','NVCC_GENCODE']}))\""
    ) % IMAGE_ENV
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=30)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


@needs_repo
def test_default_image_matches_the_pin():
    env = _load_env()
    expected = "%s:%s" % (env["IMAGE_REPO"], env["IMAGE_TAG"])
    assert DEFAULT_RDMA_TEST_IMAGE == expected, (
        "DEFAULT_RDMA_TEST_IMAGE disagrees with docker/rdma-test/image.env; bump both together or the default names an image nobody built"
    )


@needs_repo
def test_default_image_is_pinned_not_latest():
    """A moving tag would make two runs a month apart incomparable."""
    assert not DEFAULT_RDMA_TEST_IMAGE.endswith(":latest")


@needs_repo
def test_image_tag_tracks_the_nccl_version():
    """The tag IS the cadence signal, so it must name the NCCL release."""
    env = _load_env()
    assert env["IMAGE_TAG"] == "nccl-%s" % env["NCCL_VERSION"].lstrip("v")


@needs_repo
def test_gencode_covers_gb10():
    """sm_121 is DGX Spark; without it nccl-tests will not run there."""
    assert "sm_121" in _load_env()["NVCC_GENCODE"]


@needs_repo
def test_every_value_survives_being_sourced_by_bash():
    """build.sh ``source``s this file, so multi-word values must be quoted.

    Unquoted, bash reads ``NVCC_GENCODE=-gencode=A -gencode=B`` as an
    assignment prefixing the command ``-gencode=B``: the variable is never set
    globally, the build gets an empty gencode list and silently drops sm_121.
    Nothing downstream errors — you get an image that cannot run on a DGX
    Spark.
    """
    sourced = _source_env()
    for key, value in _load_env().items():
        if key in sourced:
            assert sourced[key] == value, "%s does not survive `source`; quote it in image.env" % key
    assert "sm_121" in sourced["NVCC_GENCODE"]


@needs_repo
def test_gencode_lists_every_architecture_after_sourcing():
    """All four, not just the first — the failure mode is a truncated list."""
    gencode = _source_env()["NVCC_GENCODE"]
    for arch in ("sm_90", "sm_100", "sm_120", "sm_121"):
        assert arch in gencode, "%s missing from the sourced NVCC_GENCODE" % arch


@needs_repo
def test_dockerfile_installs_perftest_and_builds_nccl_tests():
    """The image's whole reason to exist: both suites' tooling in one place."""
    text = DOCKERFILE.read_text()
    assert re.search(r"^\s+perftest", text, re.MULTILINE), "perftest missing from the runtime stage"
    assert "nccl-tests" in text
    assert "MPI=1" in text, "nccl-tests without MPI cannot span nodes"


# Environment variables the NGC CUDA images define. An ARG sharing one of
# these names is shadowed by the inherited ENV under the legacy builder — but
# not under BuildKit — so the same Dockerfile builds different content
# depending on who ran it. Observed live: `ARG NCCL_VERSION` built NVIDIA's
# bundled 2.30.7-1 under `docker build` and our pinned tag under buildx.
NGC_RESERVED_ENV = {
    "NCCL_VERSION",
    "CUDA_VERSION",
    "CUDA_HOME",
    "CUDA_PKG_VERSION",
    "NVIDIA_VISIBLE_DEVICES",
    "NVIDIA_DRIVER_CAPABILITIES",
    "NVIDIA_REQUIRE_CUDA",
    "NV_LIBNCCL_PACKAGE",
    "NV_LIBNCCL_PACKAGE_NAME",
    "NV_LIBNCCL_PACKAGE_VERSION",
    "LD_LIBRARY_PATH",
    "PATH",
}


def _instructions() -> str:
    """The Dockerfile with comment lines removed.

    Assertions about what the build *does* must not match the prose
    explaining why it does it — these comments name the very strings the
    checks forbid.
    """
    return "\n".join(line for line in DOCKERFILE.read_text().splitlines() if not line.lstrip().startswith("#"))


def _stage_args() -> list[str]:
    """ARG names declared *inside* a stage (i.e. after the first FROM).

    Global ARGs before the first FROM are only expanded in ``FROM`` lines,
    where no base-image environment exists yet, so they cannot be shadowed.
    """
    names, seen_from = [], False
    for raw in DOCKERFILE.read_text().splitlines():
        line = raw.strip()
        if line.upper().startswith("FROM "):
            seen_from = True
        elif seen_from and line.upper().startswith("ARG "):
            names.append(line[4:].split("=", 1)[0].strip())
    return names


@needs_repo
def test_no_stage_arg_collides_with_an_ngc_environment_variable():
    """A shadowed ARG silently changes what gets built, per builder."""
    for name in _stage_args():
        assert name not in NGC_RESERVED_ENV, (
            "ARG %s collides with an env the NGC base image sets; the legacy builder "
            "will use NVIDIA's value instead of ours. Rename it (e.g. NCCL_GIT_REF)." % name
        )


@needs_repo
def test_nccl_ref_is_wired_through_every_caller():
    """image.env's pin must actually reach the Dockerfile's arg."""
    assert "NCCL_GIT_REF" in DOCKERFILE.read_text()

    build_sh = (REPO_ROOT / "docker" / "rdma-test" / "build.sh").read_text()
    assert "NCCL_GIT_REF=${NCCL_VERSION}" in build_sh

    workflow = (REPO_ROOT / ".github" / "workflows" / "rdma-test-image.yml").read_text()
    assert "NCCL_GIT_REF=${{ env.NCCL_VERSION }}" in workflow


@needs_repo
def test_nccl_tests_is_pinned_too():
    """An unpinned nccl-tests is how upstream broke this build once already.

    ``git clone --depth 1`` of the default branch picked up the new
    ``device_api/gin`` tests, whose ``utils/common.cc`` instantiates the
    deprecated MPI C++ bindings — headers Ubuntu ships without the matching
    ``libmpi_cxx`` — and the link failed. No change on our side.
    """
    env = _load_env()
    assert env.get("NCCL_TESTS_VERSION", "").startswith("v")

    text = DOCKERFILE.read_text()
    assert "NCCL_TESTS_GIT_REF" in text
    assert 'git clone --depth 1 --branch "${NCCL_TESTS_GIT_REF}"' in text
    # No unpinned clone of either repository may survive.
    assert "git clone --depth 1 https://github.com/NVIDIA/nccl-tests.git" not in text
    assert "git clone --depth 1 https://github.com/NVIDIA/nccl.git" not in text

    build_sh = (REPO_ROOT / "docker" / "rdma-test" / "build.sh").read_text()
    assert "NCCL_TESTS_GIT_REF=${NCCL_TESTS_VERSION}" in build_sh

    workflow = (REPO_ROOT / ".github" / "workflows" / "rdma-test-image.yml").read_text()
    assert "NCCL_TESTS_GIT_REF=${{ env.NCCL_TESTS_VERSION }}" in workflow


@needs_repo
def test_make_cxxflags_is_not_overridden():
    """Setting CXXFLAGS on make's command line REPLACES the Makefile's own.

    Tried as a fix for the MPI C++ bindings and it broke the build earlier and
    more confusingly, by dropping nccl-tests' include paths: ``gethostname was
    not declared in this scope``.
    """
    assert "CXXFLAGS=" not in _instructions()


@needs_repo
def test_dockerfile_asserts_the_binary_sparkrun_actually_runs():
    """A build that produced no all_gather_perf must fail at build time."""
    assert "test -x /opt/nccl-tests/build/all_gather_perf" in DOCKERFILE.read_text()


@needs_repo
def test_dockerfile_verifies_the_ref_it_cloned():
    """A mis-resolved pin must fail the build, not produce a quietly wrong image."""
    text = DOCKERFILE.read_text()
    assert 'test -n "${NCCL_GIT_REF}"' in text
    assert "describe --tags --exact-match" in text


@needs_repo
def test_runtime_stage_does_not_inherit_the_cuda_base():
    """The runtime stage is plain Ubuntu; only the builder needs CUDA.

    Measured on arm64: 4.3 GB inheriting ``nvcr.io/nvidia/cuda:*-runtime``
    against ~739 MB this way, for identical contents. The base carries ~1.9 GB
    of cuBLAS / cuFFT / cuSPARSE / cuSolver / cuRAND and the only CUDA library
    anything here links is libcudart, at 768 KB.
    """
    froms = [line.split()[1] for line in DOCKERFILE.read_text().splitlines() if line.startswith("FROM ")]
    assert len(froms) == 2, "expected a builder and a runtime stage"
    assert froms[0].startswith("nvcr.io/nvidia/cuda"), "the builder needs the CUDA toolchain"
    assert froms[1].startswith("ubuntu:"), "the runtime stage must not inherit the CUDA base"


@needs_repo
def test_runtime_stage_declares_driver_capabilities():
    """Off the CUDA base, these are ours to set.

    The NVIDIA container runtime defaults ``NVIDIA_DRIVER_CAPABILITIES`` to
    ``utility``, which excludes compute — CUDA would simply be absent and the
    collective would fail with nothing explaining why.
    """
    text = DOCKERFILE.read_text()
    assert "NVIDIA_DRIVER_CAPABILITIES=compute,utility" in text
    assert "NVIDIA_VISIBLE_DEVICES=all" in text


@needs_repo
def test_cuda_library_is_staged_architecture_neutrally():
    """CUDA's real lib dir is targets/<arch>/lib — sbsa-linux vs x86_64-linux.

    A COPY naming either one builds on one architecture and fails on the
    other, so the builder stages the file to a fixed path where a shell can
    expand the glob.
    """
    text = DOCKERFILE.read_text()
    assert "/staging/cuda" in text
    instructions = _instructions()
    for arch_path in ("sbsa-linux", "x86_64-linux"):
        assert arch_path not in instructions, "%s hardcoded; the other architecture would fail" % arch_path


@needs_repo
def test_ubuntu_tag_is_separate_from_ubuntu_version():
    """NGC wants `ubuntu24.04`, the Ubuntu image wants `24.04`.

    A Dockerfile FROM has no shell, so ``${UBUNTU_VERSION#ubuntu}`` is not
    expansion Docker performs — the two spellings must both be pinned.
    """
    env = _load_env()
    assert env["UBUNTU_VERSION"] == "ubuntu" + env["UBUNTU_TAG"]
    assert "FROM ubuntu:${UBUNTU_TAG}" in DOCKERFILE.read_text()


@needs_repo
def test_build_verifies_its_own_libraries_resolve():
    """An image that cannot resolve its libraries must fail at build time.

    The version probe greps output rather than testing the exit status:
    ``ib_write_bw --version`` prints its version and exits 1.
    """
    text = DOCKERFILE.read_text()
    assert "ld.so.conf.d" in text and "ldconfig" in text
    assert 'grep -q "not found"' in text
    assert "grep -q '^Version:'" in text


@needs_repo
def test_bulky_build_artifacts_are_dropped_in_the_builder():
    """252 MB static archive and 41 MB of .o files, neither used at runtime.

    Dropped in the builder specifically: deleting them in the runtime stage
    would leave them in the earlier layer, which still counts toward the
    image size.
    """
    text = DOCKERFILE.read_text()
    assert "-name '*.o' -delete" in text
    assert "libnccl_static.a" not in text.split("FROM ubuntu")[1], "static archive must not reach the runtime stage"


@needs_repo
def test_dockerfile_declares_no_entrypoint():
    """sparkrun drives this with `docker exec`; a consuming ENTRYPOINT breaks that."""
    assert not re.search(r"^ENTRYPOINT", DOCKERFILE.read_text(), re.MULTILINE)
