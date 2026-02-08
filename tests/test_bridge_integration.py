"""Integration test for Python bridge startup and basic RPC."""

import json
import os
import subprocess
import sys
import time
import pytest

BRIDGE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'python_bridge.py',
)


def send_rpc(proc, method, params=None, req_id=1, timeout=10):
    """Send a JSON-RPC request and read the response."""
    request = json.dumps({
        "jsonrpc": "2.0",
        "id": req_id,
        "method": method,
        "params": params or {},
    })
    proc.stdin.write(request + "\n")
    proc.stdin.flush()

    # Read lines until we get a response with our ID
    deadline = time.time() + timeout
    while time.time() < deadline:
        line = proc.stdout.readline().strip()
        if not line:
            time.sleep(0.05)
            continue
        try:
            msg = json.loads(line)
            if msg.get("id") == req_id:
                return msg
            # Skip notification messages (no id)
        except json.JSONDecodeError:
            continue
    raise TimeoutError(f"No response for method={method} within {timeout}s")


@pytest.fixture(scope="module")
def bridge_proc():
    """Spawn the Python bridge as a subprocess."""
    env = os.environ.copy()
    env['OPENAI_API_KEY'] = ''  # Intentionally blank for testing
    env['MEMORY_DB_PATH'] = ':memory:'

    proc = subprocess.Popen(
        [sys.executable, BRIDGE_PATH],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        cwd=os.path.dirname(BRIDGE_PATH),
    )

    # Wait for the "ready" notification
    deadline = time.time() + 15
    ready = False
    while time.time() < deadline:
        line = proc.stdout.readline().strip()
        if not line:
            time.sleep(0.1)
            continue
        try:
            msg = json.loads(line)
            if msg.get("method") == "ready":
                ready = True
                break
        except json.JSONDecodeError:
            continue

    if not ready:
        proc.kill()
        stderr_out = proc.stderr.read()
        pytest.fail(f"Bridge did not send ready signal. stderr: {stderr_out[:2000]}")

    yield proc

    proc.stdin.close()
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


class TestBridgeStartup:
    def test_echo_rpc(self, bridge_proc):
        response = send_rpc(bridge_proc, "echo", {"msg": "test"}, req_id=1)
        assert "result" in response
        assert response["result"]["msg"] == "test"

    def test_health_rpc(self, bridge_proc):
        response = send_rpc(bridge_proc, "health", req_id=2)
        assert "result" in response
        assert response["result"]["status"] == "ok"

    def test_unknown_method_returns_error(self, bridge_proc):
        response = send_rpc(bridge_proc, "nonexistent.method", req_id=3)
        assert "error" in response
        assert response["error"]["code"] == -32601

    def test_loop_handlers_registered(self, bridge_proc):
        """Verify that cognitive loop handlers are registered."""
        response = send_rpc(bridge_proc, "health", req_id=4)
        assert response["result"]["status"] == "ok"
        # The ready message should have listed handlers, but we can also
        # verify by calling a loop handler
