# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only
# Additional permission under AGPLv3 section 7: see src/sparkrun/plugins/sparkroute/LICENSE_EXCEPTION.

"""Opt-in integration with the real Go executable; no GPUs or model downloads.

SPARKROUTE_TEST_BINARY=/absolute/path/sparkroute pytest tests/test_sparkroute_live.py
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

from sparkrun.plugins.sparkroute import register
from sparkrun.plugins.sparkroute.engine import SparkrouteEngine
from sparkrun.plugins.sparkroute.release import resolve_sparkrun_executable

pytestmark = pytest.mark.skipif(
    not os.environ.get("SPARKROUTE_TEST_BINARY"), reason="set SPARKROUTE_TEST_BINARY for the real Go integration"
)


def _port():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def _json(url, body=None, token=None):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = "Bearer " + token
    request = urllib.request.Request(url, data=json.dumps(body).encode() if body else None, headers=headers)
    with urllib.request.urlopen(request, timeout=15) as response:
        return json.load(response)


def test_real_gateway_is_supervised_authenticated_and_serves_warm_aliases(tmp_path, monkeypatch):
    calls = []

    class Upstream(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append((self.path, body))
            response = json.dumps({"id": "chat-test", "object": "chat.completion", "model": body["model"], "choices": []}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

        def log_message(self, *_args):
            pass

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    serving = threading.Thread(target=upstream.serve_forever, daemon=True)
    serving.start()
    seed = tmp_path / "operator.json"
    seed.write_text(
        json.dumps(
            {
                "providers": [
                    {"name": "fixture", "type": "openai_compatible", "base_url": "http://127.0.0.1:%d/v1" % upstream.server_port}
                ],
                "deployments": [{"name": "warm", "provider": "fixture", "model": "upstream-model"}],
                "virtual_models": [
                    {
                        "name": "assistant",
                        "aliases": ["coding"],
                        "response_model": "virtual",
                        "pools": [{"targets": [{"deployment": "warm", "weight": 1}]}],
                    }
                ],
            }
        )
    )
    monkeypatch.setenv("SPARKRUN_FEATURE_GATEWAY_SPARKROUTE", "1")
    monkeypatch.setenv("SPARKRUN_SPARKROUTE_BINARY", str(Path(os.environ["SPARKROUTE_TEST_BINARY"]).resolve()))
    monkeypatch.setenv("SPARKROUTE_OPERATIONS_ADDRESS", "127.0.0.1:0")
    register(None)
    config = SimpleNamespace(
        bindings=[],
        aliases={},
        discovered_models=[],
        capability_policy="permissive",
        gateway_config=str(seed),
        gateway_admin_port=_port(),
        gateway_admin_host="127.0.0.1",
        gateway_admin_configured=True,
        gateway_allow_insecure_admin_nonloopback=False,
    )
    engine = SparkrouteEngine(
        host="127.0.0.1", port=_port(), master_key="test-only-token", state_dir=tmp_path / "supervisor", proxy_config=config
    )
    try:
        assert engine.start() == 0
        assert engine.is_running()
        base = "http://" + engine.data_address
        names = {entry["id"] for entry in _json(base + "/v1/models", token="test-only-token")["data"]}
        assert {"assistant", "coding"} <= names
        reply = _json(
            base + "/v1/chat/completions", {"model": "coding", "messages": [{"role": "user", "content": "hi"}]}, "test-only-token"
        )
        assert reply["model"] == "assistant"  # response_model=virtual uses the canonical name
        assert calls[0][0] == "/v1/chat/completions"
        assert calls[0][1]["model"] == "upstream-model"
        assert engine.reconcile(reason="idempotent integration check") == (0, 0)
        with pytest.raises(urllib.error.HTTPError) as rejected:
            _json(engine.admin_url + "/v1/ui/bootstrap")
        assert rejected.value.code == 401
        bootstrap = _json(engine.admin_url + "/v1/ui/bootstrap", token=engine.credential.read_admin_token())
        assert bootstrap["edition"] == "standalone"
        assert bootstrap["build"]["license"] == "AGPL-3.0-only"
        # The actual host console script must accept the hidden JSON bridge.
        bridge = subprocess.run(
            [resolve_sparkrun_executable(), "gateway-bridge"],
            input=json.dumps({"schema_version": 3, "request_id": "live-capabilities", "operation": "capabilities"}),
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
        payload = json.loads(bridge.stdout)
        assert payload["request_id"] == "live-capabilities"
        assert payload["schema_version"] == 3
        assert payload["ok"] is True
    finally:
        engine.stop()
        upstream.shutdown()
        upstream.server_close()
        serving.join(timeout=5)
    assert not engine.is_running()
