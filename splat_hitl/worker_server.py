"""Serve the render worker to a flight that runs inside the Docker container.

The flight runs in the container, and the container has no GPU. The render
worker needs one, so it runs on the lab PC itself, and this joins the two over
the PC's own network:

    on the lab PC, outside Docker, from the folder that holds scenes/:
        python3 -m splat_hitl.worker_server --bundle scenes/a3_train \\
            --python <a python with torch and gsplat>

    in the container:
        python3 -m splat_hitl.ros_node ... --worker-port 7790

At start-up it launches the worker once and waits for it to say it is ready,
so a missing GPU or a broken environment shows up here rather than at the
start of a flight. Then each flight that connects gets a fresh worker on the
scene, started with `worker_command(bundle)`, and bytes pass both ways until
either side closes. The worker's own messages appear in this terminal.

It listens on 127.0.0.1 only, so nothing outside the PC can reach it. The
container shares the PC's network, so 127.0.0.1 is the same address from inside
it.
"""
from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
import threading
from typing import Callable, Optional, Sequence

__all__ = ["DEFAULT_PORT", "listen", "check_worker", "serve"]

DEFAULT_PORT = 7790


def listen(port: int = DEFAULT_PORT, host: str = "127.0.0.1") -> socket.socket:
    """A listening socket. Port 0 picks a free one (for tests)."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, int(port)))
    srv.listen(1)
    return srv


def _start(cmd: Sequence[str]) -> subprocess.Popen:
    return subprocess.Popen(list(cmd), stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE)


def _stop(proc: subprocess.Popen, wait_s: float = 5.0) -> None:
    try:
        proc.stdin.close()
    except (OSError, ValueError):
        pass
    try:
        proc.wait(timeout=wait_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def check_worker(cmd: Sequence[str]) -> dict:
    """Start the worker, read its handshake, and close it again.

    Raises RuntimeError if it never says it is ready.
    """
    proc = _start(cmd)
    try:
        line = proc.stdout.readline()
        if not line:
            raise RuntimeError("the worker exited before it was ready (exit code "
                               "%r). Its error is printed above." % proc.wait())
        hello = json.loads(line)
        if not hello.get("ready"):
            raise RuntimeError("the worker did not say it was ready: %r" % (hello,))
        proc.stdin.write(b'{"cmd": "close"}\n')
        proc.stdin.flush()
        return hello
    finally:
        _stop(proc)


def _pump_in(conn: socket.socket, proc: subprocess.Popen) -> None:
    """The flight's requests, into the worker."""
    try:
        while True:
            chunk = conn.recv(1 << 16)
            if not chunk:
                break
            proc.stdin.write(chunk)
            proc.stdin.flush()
    except (OSError, ValueError):
        pass
    finally:
        try:
            proc.stdin.close()            # the worker reads EOF and exits
        except (OSError, ValueError):
            pass


def _bridge(conn: socket.socket, cmd: Sequence[str], scene: Optional[str],
            log: Callable[[str], None]) -> None:
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    proc = _start(cmd)
    try:
        first = proc.stdout.readline()
        if not first:
            log("the worker exited before it was ready (exit code %r); its error "
                "is printed above" % proc.wait())
            return
        if scene is not None:
            # Name the scene in the handshake, so the flight can log which one
            # it was rendered from.
            try:
                hello = json.loads(first)
                hello["scene"] = scene
                first = (json.dumps(hello) + "\n").encode()
            except ValueError:
                pass
        conn.sendall(first)
        threading.Thread(target=_pump_in, args=(conn, proc), daemon=True).start()
        while True:
            chunk = proc.stdout.read1(1 << 16)
            if not chunk:
                break
            conn.sendall(chunk)
    except OSError:
        pass                              # the flight went away
    finally:
        _stop(proc)


def serve(srv: socket.socket, cmd: Sequence[str], scene: Optional[str] = None,
          once: bool = False, log: Callable[[str], None] = print) -> None:
    """Give each connection its own worker, one connection at a time."""
    while True:
        conn, _ = srv.accept()
        log("flight connected: starting the worker")
        try:
            _bridge(conn, cmd, scene, log)
        finally:
            conn.close()
        log("flight disconnected: worker stopped")
        if once:
            return


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", required=True, help="the scene folder")
    ap.add_argument("--python", default=sys.executable,
                    help="a python with torch and gsplat; default: this one")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    a = ap.parse_args(argv)

    from .bundle import SceneBundle
    from .renderer import worker_command

    bundle = SceneBundle.load(a.bundle)
    cmd = worker_command(bundle, python=a.python)
    print("worker: %s\n" % " ".join(cmd), flush=True)
    try:
        hello = check_worker(cmd)
    except (RuntimeError, ValueError) as exc:
        print("\nthe render worker does not start: %s" % exc)
        return 1
    print("\nworker ready: backend %s, %.6f m per unit"
          % (hello.get("backend"), float(hello.get("scale_to_metres", 0.0))))
    try:
        srv = listen(a.port)
    except OSError as exc:
        print("cannot listen on port %d: %s. Is another render server running?"
              % (a.port, exc))
        return 1
    print("serving %s on 127.0.0.1:%d. Fly with --worker-port %d; Ctrl-C here "
          "to stop.\n" % (bundle.name, a.port, a.port), flush=True)
    try:
        serve(srv, cmd, scene=bundle.name,
              log=lambda m: print(m, flush=True))
    except KeyboardInterrupt:
        pass
    finally:
        srv.close()
    return 0


if __name__ == "__main__":                                   # pragma: no cover
    raise SystemExit(main())
