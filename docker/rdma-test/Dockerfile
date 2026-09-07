# sparkrun RDMA test image — perftest + NCCL + nccl-tests.
#
# Used by `sparkrun setup rdma-test`. Two suites, two very different needs:
#
#   perftest  Pure fabric test, no CUDA. Ships preinstalled on DGX OS, so the
#             command runs it host-native there and never pulls this image.
#             It is here for hosts that lack it.
#   nccl      The collective. This is why the image exists: building NCCL and
#             nccl-tests takes ~10 minutes plus nvcc, OpenMPI and a compiler,
#             which is not something a diagnostic may do to a cluster.
#
# NCCL is built from a pinned source tag rather than taken from apt because
# the tag is this image's release cadence (see image.env), and because GB10
# support lives on NVIDIA's update branch.
#
# Multi-stage on purpose: the arm64 CUDA -devel image is ~6 GB, and this gets
# distributed to every host in the cluster.

ARG CUDA_VERSION=13.3.1
ARG UBUNTU_VERSION=ubuntu24.04
ARG UBUNTU_TAG=24.04

# ---------------------------------------------------------------------------
# Builder — compiles NCCL and nccl-tests
# ---------------------------------------------------------------------------
FROM nvcr.io/nvidia/cuda:${CUDA_VERSION}-devel-${UBUNTU_VERSION} AS builder

# DELIBERATELY NOT `NCCL_VERSION`. The NGC CUDA base images set
# `ENV NCCL_VERSION=<the libnccl2 they bundle>` (2.30.7-1 in cuda:13.3.1), and
# an ENV inherited from the base image outranks an ARG of the same name — but
# only under the LEGACY builder. BuildKit lets the ARG win. So an ARG called
# NCCL_VERSION builds a DIFFERENT NCCL depending on which builder ran it:
# ours under buildx, NVIDIA's bundled one under `DOCKER_BUILDKIT=0`. Observed
# live, and it is silent whenever the base image's value happens to be a valid
# git ref. A name NGC does not define is the only fix that holds either way.
ARG NCCL_GIT_REF=v2.31.2-1
ARG NVCC_GENCODE="-gencode=arch=compute_90,code=sm_90 -gencode=arch=compute_100,code=sm_100 -gencode=arch=compute_120,code=sm_120 -gencode=arch=compute_121,code=sm_121"

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        devscripts \
        git \
        libopenmpi-dev \
        openmpi-bin \
    && rm -rf /var/lib/apt/lists/*

# NCCL from the pinned tag. The guards make a mis-resolved pin loud: an empty
# ref would otherwise clone the default branch, and a shadowed one would build
# a version nobody asked for while every log line still looks healthy.
RUN test -n "${NCCL_GIT_REF}" || { echo "NCCL_GIT_REF is empty" >&2; exit 1; } \
    && test -n "${NVCC_GENCODE}" || { echo "NVCC_GENCODE is empty" >&2; exit 1; } \
    && echo "Building NCCL ${NCCL_GIT_REF} for: ${NVCC_GENCODE}" \
    && git clone --depth 1 --branch "${NCCL_GIT_REF}" https://github.com/NVIDIA/nccl.git /opt/nccl \
    && git -C /opt/nccl describe --tags --exact-match 2>/dev/null | grep -qx "${NCCL_GIT_REF}" \
        || { echo "cloned ref is not ${NCCL_GIT_REF}" >&2; exit 1; } \
    && make -C /opt/nccl -j"$(nproc)" src.build NVCC_GENCODE="${NVCC_GENCODE}" \
    && rm -rf /opt/nccl/.git

# nccl-tests against the NCCL we just built, with MPI so mpirun can span nodes.
#
# Pinned, and it has to be: cloned unpinned this broke the moment upstream
# added the `device_api/gin` tests, whose utils/common.cc instantiates the
# deprecated MPI C++ bindings. Ubuntu ships those headers but not the matching
# libmpi_cxx, so the link died on `undefined reference to MPI::Comm::Comm()`.
# v2.19.7 predates that directory; the pin is the whole fix.
#
# Note what is NOT done here: passing CXXFLAGS=-DOMPI_SKIP_MPICXX to suppress
# those bindings. A variable set on make's command line REPLACES the
# Makefile's own assignment rather than adding to it, which drops nccl-tests'
# include paths and fails earlier and more confusingly (`gethostname` not
# declared). If a future bump reintroduces the C++ bindings, patch the source
# or choose a different tag — do not reach for CXXFLAGS.
ARG NCCL_TESTS_GIT_REF=v2.19.7
RUN test -n "${NCCL_TESTS_GIT_REF}" || { echo "NCCL_TESTS_GIT_REF is empty" >&2; exit 1; } \
    && git clone --depth 1 --branch "${NCCL_TESTS_GIT_REF}" https://github.com/NVIDIA/nccl-tests.git /opt/nccl-tests \
    && git -C /opt/nccl-tests describe --tags --exact-match 2>/dev/null | grep -qx "${NCCL_TESTS_GIT_REF}" \
        || { echo "cloned ref is not ${NCCL_TESTS_GIT_REF}" >&2; exit 1; } \
    && make -C /opt/nccl-tests -j"$(nproc)" \
        MPI=1 \
        MPI_HOME=/usr/lib/$(uname -m)-linux-gnu/openmpi \
        NCCL_HOME=/opt/nccl/build \
        NVCC_GENCODE="${NVCC_GENCODE}" \
    && test -x /opt/nccl-tests/build/all_gather_perf \
        || { echo "all_gather_perf was not built" >&2; exit 1; } \
    && rm -rf /opt/nccl-tests/.git \
    && find /opt/nccl-tests/build -name '*.o' -delete

# Stage the few runtime pieces the final image needs at ARCHITECTURE-NEUTRAL
# paths. CUDA's real library directory is /usr/local/cuda/targets/<arch>/lib —
# `sbsa-linux` on arm64, `x86_64-linux` on amd64 — so a COPY naming it directly
# builds on one architecture and fails on the other. Resolving it here, where
# a shell can expand the glob, keeps the runtime stage arch-agnostic.
#
# Only libcudart is taken: it is the single CUDA library nccl-tests links
# (768 KB), against ~1.9 GB of cuBLAS/cuFFT/cuSPARSE/cuSolver in the runtime
# base that nothing here ever touches. libnccl_static.a (252 MB) is likewise
# skipped — we link the shared object.
RUN mkdir -p /staging/cuda /staging/nccl \
    && cp -a /usr/local/cuda/lib64/libcudart.so* /staging/cuda/ \
    && cp -a /opt/nccl/build/lib/libnccl.so.* /staging/nccl/ \
    && test -s /staging/cuda/libcudart.so.13 -o -s /staging/cuda/libcudart.so \
        || { echo "libcudart was not staged" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------
#
# Plain Ubuntu, NOT the NGC CUDA runtime image. That base is ~2.2 GB of
# cuBLAS / cuFFT / cuSPARSE / cuSolver / cuRAND, and the only CUDA library
# anything here links is libcudart (768 KB), staged above. Measured on arm64:
# 4.3 GB inheriting the CUDA base, ~730 MB this way, with identical contents.
#
# The whole nccl-tests binary set is kept rather than a curated subset, so
# `api.setup.rdma_test(nccl_binary=...)` accepts any collective it names
# instead of failing at run time on a binary nobody shipped. Only the .o
# objects were dropped, in the builder, where they never reach a layer here.
# UBUNTU_TAG, not UBUNTU_VERSION: the NGC bases want `ubuntu24.04` while the
# Ubuntu image wants `24.04`, and a Dockerfile FROM has no shell to strip the
# prefix with — ${VAR#ubuntu} is not expansion Docker performs.
FROM ubuntu:${UBUNTU_TAG}

# Same ARG/ENV collision applies here, so this is NCCL_GIT_REF too: labelling
# from NCCL_VERSION would record whatever the base image bundles.
ARG NCCL_GIT_REF=v2.31.2-1
LABEL org.opencontainers.image.title="sparkrun-rdma-test" \
      org.opencontainers.image.description="perftest + NCCL + nccl-tests for sparkrun setup rdma-test" \
      org.opencontainers.image.source="https://github.com/spark-arena/sparkrun" \
      dev.sparkrun.nccl.version="${NCCL_GIT_REF}"

ENV DEBIAN_FRONTEND=noninteractive

# perftest brings ib_write_bw / ib_write_lat; rdma-core + ibverbs-providers
# bring the userspace verbs stack (libmlx5) the tools and NCCL talk through —
# loaded by dlopen, so it is invisible to ldd and easy to leave out by
# accident. openssh-client is needed by mpirun's rsh agent, which SSHes to the
# peer host and docker-execs into its container. infiniband-diags /
# ibverbs-utils / pciutils / iproute2 are not used by sparkrun itself; they are
# kept because this is a diagnostic image and the moment it matters is the
# moment someone is inside it with --keep-containers asking why the fabric is
# unhappy.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        ibverbs-providers \
        ibverbs-utils \
        infiniband-diags \
        iproute2 \
        libibumad3 \
        libibverbs1 \
        libopenmpi3t64 \
        librdmacm1 \
        openmpi-bin \
        openssh-client \
        pciutils \
        perftest \
        rdma-core \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /staging/nccl/ /opt/nccl/lib/
COPY --from=builder /staging/cuda/ /usr/local/cuda/lib64/
COPY --from=builder /opt/nccl-tests/build/ /opt/nccl-tests/build/

# Note the version check greps the OUTPUT rather than testing the exit code:
# `ib_write_bw --version` prints its version and exits 1, so an exit-code test
# rejects a perfectly good image.
#
# Registered with ld.so rather than left to LD_LIBRARY_PATH alone: the
# workload arrives via `docker exec`, whose environment is not guaranteed to
# carry it, and mpirun re-execs ranks through an rsh agent. The link check runs
# here so an image that cannot resolve its own libraries fails at build time
# rather than mid-launch on the cluster.
RUN printf '/opt/nccl/lib\n/usr/local/cuda/lib64\n' > /etc/ld.so.conf.d/sparkrun-rdma.conf \
    && ldconfig \
    && if ldd /opt/nccl-tests/build/all_gather_perf | grep -q "not found"; then \
           echo "unresolved libraries in all_gather_perf:" >&2; \
           ldd /opt/nccl-tests/build/all_gather_perf >&2; \
           exit 1; \
       fi \
    && if ! ib_write_bw --version 2>&1 | grep -q '^Version:'; then \
           echo "ib_write_bw did not report a version" >&2; \
           ib_write_bw --version >&2 2>&1 || true; \
           exit 1; \
       fi

# Off the CUDA base these are ours to declare: the NVIDIA container runtime
# defaults DRIVER_CAPABILITIES to "utility", which excludes compute — CUDA
# would simply be absent and the collective would fail with nothing to explain
# why.
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    NCCL_HOME=/opt/nccl \
    LD_LIBRARY_PATH=/opt/nccl/lib:/usr/local/cuda/lib64 \
    PATH=/opt/nccl-tests/build:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

# No ENTRYPOINT: sparkrun starts this with `sleep infinity` and drives it with
# `docker exec`, and an entrypoint that consumed the command would break that
# the way `Executor.verify_command_passthrough` exists to catch. Ubuntu
# declares none either, so unlike the NGC base there is nothing to inherit.
CMD ["sleep", "infinity"]
