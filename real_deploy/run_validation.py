"""
Launch server + client in sequence (single command, one machine).

From project root:
  python real_deploy/run_validation.py

Requires: torch, torchvision (see real_deploy/requirements.txt)

Parses REAL_DEPLOY_TIMING lines, prints a summary table + conclusions, runs
a single-process full-model baseline, and writes thesis assets under figures/:
real_deploy_last.json, real_deploy_table.md, and PNG charts (needs matplotlib).
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PORT = 29500

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TIMING_RE = re.compile(r"^REAL_DEPLOY_TIMING:(.+)$", re.MULTILINE)


def _parse_timing_block(text: str) -> dict | None:
    m = TIMING_RE.search(text or "")
    if not m:
        return None
    try:
        return json.loads(m.group(1).strip())
    except json.JSONDecodeError:
        return None


def _start_server_stdout_drain(
    proc: subprocess.Popen,
) -> tuple[threading.Event, list[str], threading.Thread]:
    """
    Continuously read server stdout in a thread so the child never blocks on a full pipe.
    Returns (listening_event, all_chunks, reader_thread).
    """
    listening = threading.Event()
    chunks: list[str] = []

    def _reader() -> None:
        r = proc.stdout
        if r is None:
            return
        for line in iter(r.readline, ""):
            if not line:
                break
            chunks.append(line)
            sys.stdout.write(line)
            sys.stdout.flush()
            if "listening" in line:
                listening.set()

    th = threading.Thread(target=_reader, daemon=True)
    th.start()
    return listening, chunks, th


def main() -> int:
    py = sys.executable
    import os

    # Reduce mojibake / encode errors on Windows consoles (GBK default).
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    if hasattr(sys.stderr, "reconfigure"):
        try:
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    env = {**dict(os.environ), "PYTHONUNBUFFERED": "1"}

    print("Starting server process (wait until model loaded and port listening)...", flush=True)
    srv = subprocess.Popen(
        [py, "-m", "real_deploy.two_process_alexnet", "server", "--port", str(PORT)],
        cwd=str(ROOT),
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    listening_evt, srv_chunks, srv_reader = _start_server_stdout_drain(srv)
    if not listening_evt.wait(timeout=600.0):
        try:
            srv.kill()
        except OSError:
            pass
        print(
            "\n[error] Server did not reach listening state in time (model download or bind failed?).",
            file=sys.stderr,
        )
        return 1

    print("Starting client process...", flush=True)
    cli = subprocess.run(
        [py, "-m", "real_deploy.two_process_alexnet", "client", "--port", str(PORT)],
        cwd=str(ROOT),
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    if cli.stdout:
        sys.stdout.write(cli.stdout)
    if cli.stderr:
        sys.stderr.write(cli.stderr)

    # Avoid relying on wait(timeout) alone: poll until the server child exits or force-kill.
    srv_code: int | None = None
    for _ in range(600):
        srv_code = srv.poll()
        if srv_code is not None:
            break
        time.sleep(0.1)
    else:
        try:
            srv.kill()
        except OSError:
            pass
        srv.wait(timeout=10)
        print(
            "\n[error] Server process did not exit within ~60s after client finished.",
            file=sys.stderr,
        )
        return 1

    srv_reader.join(timeout=10.0)
    out_srv = "".join(srv_chunks)

    if cli.returncode != 0:
        print("Client failed with code", cli.returncode)
        return cli.returncode
    if srv_code not in (0, None):
        print("Server exit code:", srv_code)
        return srv_code or 0

    server_t = _parse_timing_block(out_srv or "")
    client_t = _parse_timing_block(cli.stdout or "")

    if not server_t or server_t.get("role") != "server":
        print("\n[warn] Could not parse server REAL_DEPLOY_TIMING; raw server output above.")
        return 1
    if not client_t or client_t.get("role") != "client":
        print("\n[warn] Could not parse client REAL_DEPLOY_TIMING; raw client output above.")
        return 1

    print("\nRunning single-process full-model baseline (same repo code)...")
    from real_deploy.summary_metrics import build_payload, print_summary
    from real_deploy.thesis_export import export_thesis_assets
    from real_deploy.two_process_alexnet import benchmark_full_forward_ms

    baseline = benchmark_full_forward_ms(warmup=1, runs=5, seed=42)
    payload = build_payload(server_t, client_t, baseline)
    print_summary(payload)

    assets = export_thesis_assets(payload, ROOT)
    print("\n---------- 论文/答辩导出（与仿真 figures/ 同目录） ----------")
    print(f"  JSON:    {assets['json']}")
    print(f"  Markdown: {assets['markdown']}")
    if assets["figures"]:
        for fp in assets["figures"]:
            print(f"  Figure:  {fp}")
    else:
        print("  (未生成 PNG：请 pip install matplotlib 后重新运行或执行 plot_real_deploy_results.py)")

    print("\nValidation finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
