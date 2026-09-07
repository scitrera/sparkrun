#!/usr/bin/env bash
# Build (and optionally publish) the sparkrun RDMA test image.
#
# All pinned inputs live in image.env — that file is the single place the NCCL
# version, and therefore this image's release cadence, is recorded.
#
# Each architecture is built NATIVELY. This image compiles NCCL and nccl-tests
# with nvcc across four GPU architectures, and doing that under QEMU turns a
# ~20 minute build into hours. So there is no cross-build mode here: run
# --push once on an arm64 machine and once on an amd64 machine, then --merge
# from either.
#
#   ./build.sh                 build locally for this machine's architecture
#   ./build.sh --push          build natively and push <tag>-<arch>
#   ./build.sh --merge         combine both <tag>-<arch> into the real tags
#
# Full publish, by hand:
#   arm64-box $ ./build.sh --push
#   amd64-box $ ./build.sh --push
#   either    $ ./build.sh --merge
#
# CI does the same thing on two native runners (see
# .github/workflows/rdma-test-image.yml), but pushes by digest instead of by
# per-arch tag — it has artifacts to carry digests between jobs, whereas two
# separate machines only share the registry, so a tag is the channel.
set -euo pipefail

# Resolve self BEFORE cd: $0 is relative for `docker/rdma-test/build.sh`, and
# --help reads this file back, which the cd would otherwise break.
SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$SELF")"
# shellcheck disable=SC1091
set -a; source ./image.env; set +a

MODE="local"
while [ $# -gt 0 ]; do
    case "$1" in
        --push)  MODE="push"; shift ;;
        --merge) MODE="merge"; shift ;;
        # Prints the header comment block, stopping at the first line that
        # isn't one — a fixed line range silently drifts as the header grows.
        -h|--help) awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "$SELF"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

# Docker's platform vocabulary, not uname's.
case "$(uname -m)" in
    aarch64|arm64) ARCH="arm64" ;;
    x86_64|amd64)  ARCH="amd64" ;;
    *) echo "unsupported build architecture: $(uname -m)" >&2; exit 1 ;;
esac

IMAGE="${IMAGE_REPO}:${IMAGE_TAG}"
ARCH_TAGS=("${IMAGE_REPO}:${IMAGE_TAG}-arm64" "${IMAGE_REPO}:${IMAGE_TAG}-amd64")

# NCCL_VERSION is the pin's name in image.env, but it reaches the Dockerfile as
# NCCL_GIT_REF: the NGC base images define ENV NCCL_VERSION, which outranks a
# same-named ARG under the legacy builder (but not BuildKit), so the arg has to
# use a name NGC does not.
build_args=(
    --build-arg "CUDA_VERSION=${CUDA_VERSION}"
    --build-arg "UBUNTU_VERSION=${UBUNTU_VERSION}"
    --build-arg "UBUNTU_TAG=${UBUNTU_TAG}"
    --build-arg "NCCL_GIT_REF=${NCCL_VERSION}"
    --build-arg "NCCL_TESTS_GIT_REF=${NCCL_TESTS_VERSION}"
    --build-arg "NVCC_GENCODE=${NVCC_GENCODE}"
)

case "$MODE" in
    local)
        echo "Building ${IMAGE} natively for ${ARCH}"
        docker build "${build_args[@]}" -t "${IMAGE}" -t "${IMAGE_REPO}:latest" .
        echo
        echo "Built ${IMAGE} (${ARCH} only). Publish with: $0 --push"
        ;;

    push)
        # Only the per-arch tag is pushed. Writing ${IMAGE} here would have
        # whichever machine finished last overwrite the other's work with a
        # single-architecture image — the exact failure the merge step exists
        # to avoid.
        echo "Building and pushing ${IMAGE}-${ARCH} natively for ${ARCH}"
        docker build "${build_args[@]}" -t "${IMAGE}-${ARCH}" .
        docker push "${IMAGE}-${ARCH}"
        echo
        echo "Pushed ${IMAGE}-${ARCH}."
        echo "Run this on the other architecture too, then: $0 --merge"
        ;;

    merge)
        echo "Checking both architecture tags exist..."
        for tag in "${ARCH_TAGS[@]}"; do
            if ! docker buildx imagetools inspect "$tag" >/dev/null 2>&1; then
                echo "missing: $tag" >&2
                echo "Build and push it from a native ${tag##*-} machine first: $0 --push" >&2
                exit 1
            fi
            echo "  found $tag"
        done

        echo "Creating multi-arch manifest ${IMAGE}"
        docker buildx imagetools create \
            -t "${IMAGE}" \
            -t "${IMAGE_REPO}:latest" \
            "${ARCH_TAGS[@]}"

        echo
        docker buildx imagetools inspect "${IMAGE}"
        echo
        # The per-arch tags are left in place deliberately: they cost nothing,
        # make a re-merge possible without rebuilding, and are what you inspect
        # when one architecture misbehaves.
        echo "Published ${IMAGE} and ${IMAGE_REPO}:latest."
        ;;
esac
