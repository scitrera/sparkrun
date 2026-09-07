"""RDMA fabric verification — the mechanism behind ``sparkrun setup rdma-test``.

``infiniband.py`` *detects* the fabric; this module *exercises* it.  The
distinction matters: :func:`~sparkrun.orchestration.infiniband.validate_ib_connectivity`
proves an IB IP answers SSH, which a link running at 1 Gb/s over the wrong NIC
would also do.  Only a real RDMA transfer distinguishes a working 200 Gb/s
fabric from a silently degraded one.

Pure mechanism, no policy: command builders, output parsers and link
derivation.  Verdicts, thresholds and rendering belong to
:mod:`sparkrun.api.setup` and the CLI respectively — the split
``generate_ib_detect_script`` / ``parse_ib_detect_output`` already follows.

Two facts drive the shape of everything here:

- **A link is a subnet, not a host pair.**  ``sparkrun setup cx7`` gives every
  point-to-point cable its own /24, so grouping configured interfaces by
  subnet recovers the physical topology exactly, with no probing.  That is
  deliberately *not*
  :func:`~sparkrun.orchestration.networking.detect_topology`, which discovers
  links by arping before addresses exist — the right tool for planning, and
  needless work once ``setup cx7`` has run.
- **A DGX Spark QSFP112 cable presents as two RDMA devices**
  (``rocep1s0f1`` and ``roceP2p1s0f1``).  Driven one at a time each reaches
  ~100 Gb/s; the cable's real ~196 Gb/s only appears when both run at once.
  A suite that tests devices only in isolation under-reports by half, which is
  why :class:`RdmaPair` groups every link between two hosts and the runner has
  a concurrent pass.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from sparkrun.orchestration.networking import CX7HostDetection, CX7Interface

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_PERFTEST_PORT",
    "RdmaLink",
    "RdmaPair",
    "BwSample",
    "LatSample",
    "NcclSample",
    "derive_link_pairs",
    "build_perftest_cmd",
    "build_nccl_test_cmd",
    "parse_perftest_bw",
    "parse_perftest_lat",
    "parse_nccl_busbw",
    "parse_device_facts",
    "parse_framed_runs",
    "rate_to_gbps",
]

#: perftest's default out-of-band port. Each concurrent test gets its own.
DEFAULT_PERFTEST_PORT = 18515


# ---------------------------------------------------------------------------
# Topology
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RdmaLink:
    """One point-to-point RDMA link: two interfaces sharing a subnet.

    Carries the RDMA device name (``hca``) alongside the net interface and IP
    for both ends, because a perftest invocation needs the device on the local
    side and the address on the remote side, and mixing them up produces a
    test that passes while measuring the wrong wire.
    """

    subnet: str
    host_a: str
    iface_a: str
    hca_a: str
    ip_a: str
    host_b: str
    iface_b: str
    hca_b: str
    ip_b: str

    @property
    def usable(self) -> bool:
        """Both ends name an RDMA device and an address."""
        return bool(self.hca_a and self.hca_b and self.ip_a and self.ip_b)

    def describe(self) -> str:
        return "%s:%s <-> %s:%s (%s)" % (self.host_a, self.hca_a, self.host_b, self.hca_b, self.subnet)


@dataclass(frozen=True)
class RdmaPair:
    """Every link between one ordered pair of hosts.

    Grouped because the aggregate-bandwidth figure is a property of the
    *cable*, not of either device on it.
    """

    host_a: str
    host_b: str
    links: tuple[RdmaLink, ...] = ()

    @property
    def key(self) -> tuple[str, str]:
        return (self.host_a, self.host_b)

    def describe(self) -> str:
        return "%s <-> %s (%d link%s)" % (
            self.host_a,
            self.host_b,
            len(self.links),
            "" if len(self.links) == 1 else "s",
        )


def _configured_interfaces(det: CX7HostDetection) -> list[CX7Interface]:
    """Interfaces with both an address and an RDMA device."""
    return [i for i in det.interfaces if i.ip and i.subnet and i.hca]


def derive_link_pairs(
    detections: dict[str, CX7HostDetection],
    hosts: list[str],
) -> list[RdmaPair]:
    """Group configured CX7 interfaces into host pairs by shared subnet.

    Only hosts present in *hosts* are considered, and only subnets shared by
    at least two of them. A subnet carrying **more than two** hosts is a
    switched segment: it is chained (``h1<->h2``, ``h2<->h3``, …) rather than
    fully meshed, so coverage stays O(N) and every host is exercised at least
    once. Full-mesh testing of a switch is a different, much longer command.

    Args:
        detections: Per-host CX7 detection results.
        hosts: Hosts to consider, in the caller's order. Pair ordering
            follows this list so output is stable across runs.

    Returns:
        One :class:`RdmaPair` per host pair, each carrying every link between
        them. Empty when nothing is configured.
    """
    order = {h: i for i, h in enumerate(hosts)}

    # subnet -> [(host, iface)], in caller host order
    by_subnet: dict[str, list[tuple[str, CX7Interface]]] = {}
    for host in hosts:
        det = detections.get(host)
        if not det or not det.detected:
            continue
        for iface in _configured_interfaces(det):
            by_subnet.setdefault(iface.subnet, []).append((host, iface))

    pairs: dict[tuple[str, str], list[RdmaLink]] = {}

    for subnet, members in sorted(by_subnet.items()):
        # One interface per host per subnet: a second would be a
        # misconfiguration, and picking arbitrarily between them would make
        # the result depend on detection order.
        seen: dict[str, CX7Interface] = {}
        for host, iface in members:
            if host in seen:
                logger.warning(
                    "Host %s has multiple interfaces on %s (%s, %s); using %s",
                    host,
                    subnet,
                    seen[host].name,
                    iface.name,
                    seen[host].name,
                )
                continue
            seen[host] = iface

        endpoints = sorted(seen.items(), key=lambda kv: order.get(kv[0], len(order)))
        if len(endpoints) < 2:
            continue

        for (host_a, if_a), (host_b, if_b) in zip(endpoints, endpoints[1:], strict=False):
            link = RdmaLink(
                subnet=subnet,
                host_a=host_a,
                iface_a=if_a.name,
                hca_a=if_a.hca,
                ip_a=if_a.ip,
                host_b=host_b,
                iface_b=if_b.name,
                hca_b=if_b.hca,
                ip_b=if_b.ip,
            )
            pairs.setdefault((host_a, host_b), []).append(link)

    return [
        RdmaPair(host_a=a, host_b=b, links=tuple(links))
        for (a, b), links in sorted(pairs.items(), key=lambda kv: (order.get(kv[0][0], 0), order.get(kv[0][1], 0)))
    ]


# ---------------------------------------------------------------------------
# Command construction
# ---------------------------------------------------------------------------


def build_perftest_cmd(
    tool: str,
    device: str,
    *,
    peer_ip: str = "",
    port: int = DEFAULT_PERFTEST_PORT,
    queue_pairs: int | None = None,
    duration: int | None = None,
    gid_index: int | None = None,
    link_type: str | None = "IB",
) -> str:
    """Build one ``ib_write_bw`` / ``ib_write_lat`` invocation.

    The server form omits *peer_ip*; the client form supplies it. Everything
    else is identical on both sides, and perftest requires that — a flag
    mismatch between the two ends is reported as a confusing negotiation
    failure rather than as the misconfiguration it is.

    Args:
        tool: ``ib_write_bw`` or ``ib_write_lat``.
        device: Local RDMA device (e.g. ``rocep1s0f1``). This is the *local*
            side's HCA on both ends, never the peer's.
        peer_ip: Server's RoCE address; empty for the server side.
        port: Out-of-band port. Concurrent tests must not share one.
        queue_pairs: ``-q``. Bandwidth only; latency is single-QP.
        duration: ``-D`` seconds. Bounds the run in wall-clock rather than
            iterations, so a slow link cannot stretch the test indefinitely.
        gid_index: ``-x``. Normally unnecessary with ``-R`` (the connection
            manager picks the GID) — an explicit override for stacks where
            it guesses wrong.
        link_type: ``--force-link``. ``"IB"`` on DGX Spark, whose ConnectX-7
            presents as IB over the QSFP cable. ``None`` omits the flag and
            lets perftest decide.

    Returns:
        The command string.
    """
    parts = [tool]
    if peer_ip:
        parts.append(peer_ip)
    parts.extend(["-d", device, "--report_gbits", "-R"])
    if link_type:
        parts.extend(["--force-link", link_type])
    if queue_pairs:
        parts.extend(["-q", str(queue_pairs)])
    if duration:
        parts.extend(["-D", str(duration)])
    if gid_index is not None:
        parts.extend(["-x", str(gid_index)])
    parts.extend(["-p", str(port)])
    return " ".join(parts)


def build_nccl_test_cmd(
    addresses: list[str],
    *,
    rsh_agent: str,
    binary: str = "all_gather_perf",
    msg_size: str = "16G",
    env: dict[str, str] | None = None,
    ranks_per_node: int = 1,
    nccl_tests_dir: str = "/opt/nccl-tests/build",
) -> str:
    """Build the ``mpirun`` command for an nccl-tests collective.

    Mirrors :meth:`~sparkrun.runtimes.trtllm.TrtllmRuntime._build_mpirun_command`
    and uses the same rsh agent
    (:func:`sparkrun.orchestration.mpi.build_rsh_wrapper`), so the collective
    is launched exactly the way a real multi-node TRT-LLM workload is.

    Args:
        addresses: Host addresses in rank order. Must match the rsh agent's
            keys exactly — mpirun passes them to the agent verbatim.
        rsh_agent: Path to the agent *inside the container*.
        binary: nccl-tests binary name.
        msg_size: Both ``-b`` and ``-e``; a single size keeps the run short
            and the number comparable between invocations.
        env: Variables to forward with ``-x``.
        ranks_per_node: One per accelerator. 1 on DGX Spark.
        nccl_tests_dir: Directory holding the built binaries in the image.

    Returns:
        The command string.
    """
    parts = [
        "mpirun",
        "--allow-run-as-root",
        "--mca",
        "plm_rsh_agent",
        rsh_agent,
        "--mca",
        "rmaps_ppr_n_pernode",
        str(ranks_per_node),
        "-H",
        ",".join("%s:%d" % (a, ranks_per_node) for a in addresses),
    ]
    for key in sorted(env or {}):
        parts.extend(["-x", key])
    parts.extend(
        [
            "%s/%s" % (nccl_tests_dir.rstrip("/"), binary),
            "-b",
            msg_size,
            "-e",
            msg_size,
            "-f",
            "2",
            "-g",
            "1",
        ]
    )
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BwSample:
    """One ``ib_write_bw`` result row."""

    bytes_: int
    iterations: int
    peak_gbps: float
    avg_gbps: float
    msg_rate_mpps: float = 0.0


@dataclass(frozen=True)
class LatSample:
    """One ``ib_write_lat`` result row. Microseconds throughout."""

    bytes_: int
    iterations: int
    t_min: float
    t_max: float
    t_typical: float
    t_avg: float = 0.0
    p99: float | None = None


@dataclass(frozen=True)
class NcclSample:
    """Summary figures from an nccl-tests run (GB/s)."""

    avg_bus_bw: float
    algo_bw: float | None = None


_NUM = r"[0-9]+(?:\.[0-9]+)?"


def _numeric_rows(stdout: str, min_fields: int) -> list[list[str]]:
    """Data rows: whitespace-split lines whose fields are all numeric.

    perftest frames its table with ``---`` rules and prefixes commentary with
    ``#``; the payload is the only line that is numbers all the way across.
    Matching on that rather than on a line offset survives the banner varying
    between versions and between ``-R`` / non-``-R`` runs.
    """
    rows = []
    for raw in stdout.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        fields = line.split()
        if len(fields) < min_fields:
            continue
        if all(re.fullmatch(_NUM, f) for f in fields):
            rows.append(fields)
    return rows


def parse_perftest_bw(stdout: str) -> BwSample | None:
    """Parse ``ib_write_bw`` output.

    Returns ``None`` when no result row is present — a timed-out or refused
    run must not be reported as a measurement. The caller keeps the raw text.
    """
    rows = _numeric_rows(stdout, min_fields=4)
    if not rows:
        return None
    f = rows[-1]
    try:
        return BwSample(
            bytes_=int(float(f[0])),
            iterations=int(float(f[1])),
            peak_gbps=float(f[2]),
            avg_gbps=float(f[3]),
            msg_rate_mpps=float(f[4]) if len(f) > 4 else 0.0,
        )
    except (ValueError, IndexError):
        logger.debug("Unparseable ib_write_bw row: %r", f)
        return None


def parse_perftest_lat(stdout: str) -> LatSample | None:
    """Parse ``ib_write_lat`` output.

    ``t_typical`` is the headline figure, not ``t_avg``: the average is
    skewed by scheduler outliers, which is why perftest reports both.
    """
    rows = _numeric_rows(stdout, min_fields=5)
    if not rows:
        return None
    f = rows[-1]
    try:
        return LatSample(
            bytes_=int(float(f[0])),
            iterations=int(float(f[1])),
            t_min=float(f[2]),
            t_max=float(f[3]),
            t_typical=float(f[4]),
            t_avg=float(f[5]) if len(f) > 5 else 0.0,
            # Older perftest builds stop at t_stdev.
            p99=float(f[7]) if len(f) > 7 else None,
        )
    except (ValueError, IndexError):
        logger.debug("Unparseable ib_write_lat row: %r", f)
        return None


_NCCL_AVG_BUS = re.compile(r"Avg bus bandwidth\s*:\s*(" + _NUM + r")", re.IGNORECASE)


def parse_nccl_busbw(stdout: str) -> NcclSample | None:
    """Parse the ``# Avg bus bandwidth`` summary from nccl-tests.

    The summary line is used rather than the per-size table because it is
    stable across nccl-tests versions, while the table's column count is not.
    """
    m = _NCCL_AVG_BUS.search(stdout)
    if not m:
        return None
    try:
        return NcclSample(avg_bus_bw=float(m.group(1)))
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Host-probe parsing
# ---------------------------------------------------------------------------


@dataclass
class RdmaDevice:
    """One RDMA device as the host reports it."""

    name: str
    state: str = ""
    rate: str = ""
    link_layer: str = ""
    netdev: str = ""
    sys_image_guid: str = ""
    phys_port_name: str = ""

    @property
    def active(self) -> bool:
        return "ACTIVE" in self.state.upper()

    @property
    def rate_gbps(self) -> float | None:
        return rate_to_gbps(self.rate)

    @property
    def port_key(self) -> tuple[str, str]:
        """Identity of the physical port this device drives.

        Devices sharing a key share a wire, so **their bandwidths are not
        additive** — the pair's ceiling is the port rate, not the sum.

        ``sys_image_guid`` names the physical adapter (every function of one
        card reports the same value); ``phys_port_name`` names the port on it.
        Both are needed: a dual-port ConnectX with two cables shares the guid
        but not the port name and *is* additive, while DGX Spark exposes
        several PCIe functions of one adapter on a single port and is not.

        Falls back to the device name when the fields are unavailable — an
        unknown device is then its own port, i.e. the additive assumption,
        which is what this code did before the fields existed.
        """
        if self.sys_image_guid:
            return (self.sys_image_guid, self.phys_port_name)
        return (self.name, self.phys_port_name)


@dataclass
class RdmaHostFacts:
    """What one host reports about its RDMA tooling and devices."""

    host: str
    has_perftest: bool = False
    has_docker: bool = False
    gid_index: int | None = None
    devices: list[RdmaDevice] = field(default_factory=list)
    complete: bool = False
    error: str = ""

    def device(self, name: str) -> RdmaDevice | None:
        for d in self.devices:
            if d.name == name:
                return d
        return None


_RATE = re.compile(r"(" + _NUM + r")\s*Gb/sec", re.IGNORECASE)


def rate_to_gbps(rate: str) -> float | None:
    """Extract Gb/s from a sysfs rate string like ``100 Gb/sec (4X EDR)``.

    This is where the bandwidth expectation comes from. Deriving it from the
    link's own reported rate is what keeps the check honest on hardware that
    is not a DGX Spark — a hardcoded "expect 100 Gb/s" would misreport every
    other NVIDIA platform.
    """
    if not rate:
        return None
    m = _RATE.search(rate)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def parse_device_facts(host: str, kv: dict[str, str]) -> RdmaHostFacts:
    """Build :class:`RdmaHostFacts` from parsed ``rdma_probe.sh`` output."""
    facts = RdmaHostFacts(host=host)
    facts.complete = kv.get("RDMA_COMPLETE") == "1"
    facts.has_perftest = kv.get("RDMA_PERFTEST") == "1"
    facts.has_docker = kv.get("RDMA_DOCKER") == "1"

    gid = kv.get("RDMA_GID_INDEX", "")
    if gid:
        try:
            facts.gid_index = int(gid)
        except ValueError:
            pass

    try:
        count = int(kv.get("RDMA_DEV_COUNT", "0"))
    except ValueError:
        count = 0
    for i in range(count):
        name = kv.get("RDMA_DEV_%d_NAME" % i, "")
        if not name:
            continue
        facts.devices.append(
            RdmaDevice(
                name=name,
                state=kv.get("RDMA_DEV_%d_STATE" % i, ""),
                rate=kv.get("RDMA_DEV_%d_RATE" % i, ""),
                link_layer=kv.get("RDMA_DEV_%d_LINK_LAYER" % i, ""),
                netdev=kv.get("RDMA_DEV_%d_NETDEV" % i, ""),
                sys_image_guid=kv.get("RDMA_DEV_%d_SYS_IMAGE_GUID" % i, ""),
                phys_port_name=kv.get("RDMA_DEV_%d_PHYS_PORT_NAME" % i, ""),
            )
        )
    return facts


_RUN_BEGIN = re.compile(r"^SPARKRUN_RDMA_RUN_(\d+)_BEGIN\s*$")
_RUN_END = re.compile(r"^SPARKRUN_RDMA_RUN_(\d+)_END\s*$")
_RUN_RC = re.compile(r"^SPARKRUN_RDMA_RUN_(\d+)_RC=(-?\d+)\s*$")


def parse_framed_runs(stdout: str) -> dict[int, tuple[str, int]]:
    """Split ``rdma_perftest.sh`` output into ``{index: (raw_output, rc)}``.

    The runner frames each invocation because several run concurrently and
    their output would otherwise interleave into an unparseable stream.
    """
    runs: dict[int, list[str]] = {}
    codes: dict[int, int] = {}
    current: int | None = None

    for line in stdout.splitlines():
        if current is None:
            m = _RUN_BEGIN.match(line)
            if m:
                current = int(m.group(1))
                runs[current] = []
                continue
            m = _RUN_RC.match(line)
            if m:
                codes[int(m.group(1))] = int(m.group(2))
            continue

        m = _RUN_END.match(line)
        if m and int(m.group(1)) == current:
            current = None
            continue
        runs[current].append(line)

    return {i: ("\n".join(lines), codes.get(i, -1)) for i, lines in sorted(runs.items())}
