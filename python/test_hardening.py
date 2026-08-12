"""Verifies the hard wall-clock deadline actually cuts off a hung connection.

Round-5 hardening (2026-08-12): a plain `requests.post(..., timeout=N)` hung
for 10+ minutes three separate times this session despite timeout=90,
because `requests`' timeout is a per-read inactivity timer, not a total
wall-clock deadline -- a connection that keeps trickling bytes (e.g. a
provider keep-alive during a long generation) never lets any single read
exceed the timeout, so the clock never actually expires.

This test does NOT trust the fix by reading the code (the same discipline
this project applied to external review claims all session): it spins up a
real local TCP server that trickles single bytes slowly enough that each
individual read stays under a generous per-read timeout, so a naive
`requests.post(timeout=per_read_timeout)` would hang for the FULL simulated
duration -- and confirms `_post_with_hard_deadline` cuts it off at the hard
deadline instead, well before the per-read timeout would ever fire.
"""

from __future__ import annotations

import socket
import threading
import time

import pytest
import requests

from experiment_llm_judge_saturation import _post_with_hard_deadline


def _start_trickling_server(
    byte_interval: float, n_bytes: int
) -> tuple[str, int, threading.Thread]:
    """A TCP server that accepts one connection and sends `n_bytes` single
    bytes of a (never-completing) HTTP status line, `byte_interval` seconds
    apart. Never sends a complete response, so a client blocks reading it
    for `byte_interval * n_bytes` seconds unless something cuts it off first.
    """
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]

    status_line = b"HTTP/1.1 200 OK\r\n"  # deliberately never followed by headers/body

    def _serve():
        srv.settimeout(5)
        try:
            conn, _ = srv.accept()
        except OSError:
            return
        with conn:
            conn.settimeout(byte_interval + 2)
            try:
                conn.recv(65536)  # consume the client's request
                for i in range(n_bytes):
                    conn.send(status_line[i % len(status_line) : i % len(status_line) + 1])
                    time.sleep(byte_interval)
            except OSError:
                pass  # client gave up (the hard deadline fired) -- expected
        srv.close()

    t = threading.Thread(target=_serve, daemon=True)
    t.start()
    return "127.0.0.1", port, t


def test_hard_deadline_actually_cuts_off_a_hung_request():
    byte_interval = 0.15
    n_bytes = 30  # full hang would take ~4.5s if nothing intervenes
    host, port, server_thread = _start_trickling_server(byte_interval, n_bytes)
    url = f"http://{host}:{port}/"

    hard_deadline = 1.0  # must fire well before the ~4.5s full trickle
    per_read_timeout = 30  # deliberately generous -- each byte arrives well inside this,
    # so a naive requests.post(timeout=per_read_timeout) would NOT time out on its own
    # until the full trickle finished (~4.5s) -- reproducing the real bug this guards.

    t0 = time.time()
    with pytest.raises(requests.RequestException):
        _post_with_hard_deadline(
            url,
            headers={},
            payload={"x": 1},
            per_read_timeout=per_read_timeout,
            hard_deadline=hard_deadline,
        )
    elapsed = time.time() - t0

    # Cut off at (approximately) the hard deadline, not the full trickle duration
    # and nowhere near the generous per-read timeout.
    assert elapsed < byte_interval * n_bytes, (
        f"took {elapsed:.2f}s -- did not cut off before the full "
        f"{byte_interval * n_bytes:.2f}s trickle finished"
    )
    assert elapsed < hard_deadline + 1.0, (
        f"took {elapsed:.2f}s -- too slow for a {hard_deadline}s deadline"
    )

    server_thread.join(timeout=byte_interval * n_bytes + 3)
