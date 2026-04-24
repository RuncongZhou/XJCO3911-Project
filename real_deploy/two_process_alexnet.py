"""
Two-process real AlexNet inference on one machine (localhost TCP).
Process A: features + avgpool + flatten -> send tensor.
Process B: receive tensor -> classifier -> logits.

Usage:
  python -m real_deploy.two_process_alexnet server [--port 29500]
  python -m real_deploy.two_process_alexnet client [--port 29500]

Or from project root:
  python real_deploy/run_validation.py
"""
from __future__ import annotations

import argparse
import io
import json
import socket
import struct
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

# Project root on path for imports if needed
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# One line per process for run_validation.py to parse (machine-readable).
TIMING_LINE_PREFIX = "REAL_DEPLOY_TIMING:"


def _emit_timing(payload: dict) -> None:
    print(TIMING_LINE_PREFIX + json.dumps(payload, ensure_ascii=False), flush=True)


def benchmark_full_forward_ms(
    warmup: int = 1,
    runs: int = 5,
    seed: int = 42,
) -> dict:
    """
    Same machine, single process, full AlexNet forward (baseline vs split pipeline).
    Returns dict with mean_ms, stdev_ms, runs, warmup.
    """
    model = load_alexnet()
    model.eval()
    device = torch.device("cpu")
    model.to(device)
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)
    x = torch.randn(1, 3, 224, 224, generator=rng, device=device)
    with torch.inference_mode():
        for _ in range(warmup):
            _ = model(x)
    samples: list[float] = []
    with torch.inference_mode():
        for _ in range(runs):
            t0 = time.perf_counter()
            _ = model(x)
            samples.append((time.perf_counter() - t0) * 1000.0)
    return {
        "mean_ms": float(statistics.mean(samples)),
        "stdev_ms": float(statistics.stdev(samples)) if len(samples) > 1 else 0.0,
        "runs": runs,
        "warmup": warmup,
    }


def load_alexnet() -> nn.Module:
    try:
        from torchvision.models import alexnet, AlexNet_Weights

        return alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
    except Exception:
        from torchvision.models import alexnet

        return alexnet(pretrained=True)


def part1_forward(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    x = model.features(x)
    x = model.avgpool(x)
    return torch.flatten(x, 1)


def part2_forward(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    return model.classifier(x)


def send_tensor(sock: socket.socket, tensor: torch.Tensor) -> None:
    buf = io.BytesIO()
    torch.save(tensor.cpu().detach(), buf)
    payload = buf.getvalue()
    sock.sendall(struct.pack("!I", len(payload)) + payload)


def recv_tensor(sock: socket.socket) -> torch.Tensor:
    raw_len = _recv_exact(sock, 4)
    (n,) = struct.unpack("!I", raw_len)
    payload = _recv_exact(sock, n)
    bio = io.BytesIO(payload)
    try:
        return torch.load(bio, map_location="cpu", weights_only=False)
    except TypeError:
        bio.seek(0)
        return torch.load(bio, map_location="cpu")


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    chunks = []
    remaining = n
    while remaining > 0:
        chunk = sock.recv(min(65536, remaining))
        if not chunk:
            raise ConnectionError("socket closed before %d bytes received" % n)
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def run_server(host: str, port: int) -> None:
    model = load_alexnet()
    model.eval()
    device = torch.device("cpu")
    model.to(device)

    rng = torch.Generator(device=device)
    rng.manual_seed(42)
    x = torch.randn(1, 3, 224, 224, generator=rng, device=device)

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(1)
    print(
        f"[server] listening on {host}:{port} (AlexNet part1: features+avgpool+flatten)",
        flush=True,
    )
    conn, addr = srv.accept()
    print(f"[server] client connected from {addr}", flush=True)

    t0 = time.perf_counter()
    with torch.inference_mode():
        h = part1_forward(model, x)
    t1 = time.perf_counter()
    comp_ms = (t1 - t0) * 1000.0

    t_send0 = time.perf_counter()
    send_tensor(conn, h)
    t_send1 = time.perf_counter()
    send_ms = (t_send1 - t_send0) * 1000.0

    print(
        f"[server] part1 compute: {comp_ms:.2f} ms | tensor shape: {tuple(h.shape)} | send: {send_ms:.2f} ms",
        flush=True,
    )

    _emit_timing(
        {
            "role": "server",
            "part1_ms": round(comp_ms, 4),
            "send_ms": round(send_ms, 4),
            "tensor_shape": list(h.shape),
        }
    )

    conn.close()
    srv.close()
    print("[server] done.", flush=True)


def run_client(host: str, port: int) -> None:
    model = load_alexnet()
    model.eval()
    device = torch.device("cpu")

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    t_conn0 = time.perf_counter()
    sock.connect((host, port))
    t_conn1 = time.perf_counter()
    conn_ms = (t_conn1 - t_conn0) * 1000.0

    t_recv0 = time.perf_counter()
    h = recv_tensor(sock)
    t_recv1 = time.perf_counter()
    recv_ms = (t_recv1 - t_recv0) * 1000.0

    h = h.to(device)
    model.to(device)

    t_inf0 = time.perf_counter()
    with torch.inference_mode():
        logits = part2_forward(model, h)
    t_inf1 = time.perf_counter()
    inf_ms = (t_inf1 - t_inf0) * 1000.0

    top = int(logits.argmax(dim=-1).item())
    print(
        f"[client] connect: {conn_ms:.2f} ms | recv tensor: {recv_ms:.2f} ms | classifier: {inf_ms:.2f} ms",
        flush=True,
    )
    print(
        f"[client] logits shape: {tuple(logits.shape)} | argmax class index: {top} (ImageNet label)",
        flush=True,
    )
    print(f"[client] validation OK: real two-process inference completed.", flush=True)

    _emit_timing(
        {
            "role": "client",
            "connect_ms": round(conn_ms, 4),
            "recv_ms": round(recv_ms, 4),
            "classifier_ms": round(inf_ms, 4),
            "argmax": top,
        }
    )

    sock.close()


def main() -> None:
    p = argparse.ArgumentParser(description="Two-process AlexNet validation (localhost)")
    p.add_argument("role", choices=("server", "client"))
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=29500)
    args = p.parse_args()

    if args.role == "server":
        run_server(args.host, args.port)
    else:
        run_client(args.host, args.port)


if __name__ == "__main__":
    main()
