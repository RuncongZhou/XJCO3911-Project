"""Build structured payload from server/client/baseline timings for print + thesis export."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def _fmt_row(cols: list[str], widths: list[int]) -> str:
    parts = []
    for c, w in zip(cols, widths):
        parts.append(str(c).ljust(w))
    return " | ".join(parts)


def build_payload(server: dict, client: dict, baseline: dict) -> dict[str, Any]:
    c_conn = float(client["connect_ms"])
    c_recv = float(client["recv_ms"])
    c_cls = float(client["classifier_ms"])
    s_p1 = float(server["part1_ms"])
    s_send = float(server["send_ms"])

    total_pipeline = c_conn + c_recv + c_cls
    full_mean = float(baseline["mean_ms"])
    overhead = total_pipeline - full_mean
    pct_recv = 100.0 * c_recv / total_pipeline if total_pipeline > 0 else 0.0
    pct_cls = 100.0 * c_cls / total_pipeline if total_pipeline > 0 else 0.0
    pct_conn = 100.0 * c_conn / total_pipeline if total_pipeline > 0 else 0.0

    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "server": server,
        "client": client,
        "baseline": baseline,
        "derived": {
            "connect_ms": c_conn,
            "recv_ms": c_recv,
            "classifier_ms": c_cls,
            "part1_ms": s_p1,
            "send_ms": s_send,
            "total_pipeline_ms": total_pipeline,
            "full_mean_ms": full_mean,
            "full_stdev_ms": float(baseline["stdev_ms"]),
            "overhead_ms": overhead,
            "pct_recv": pct_recv,
            "pct_cls": pct_cls,
            "pct_conn": pct_conn,
        },
    }


def print_summary(payload: dict) -> None:
    d = payload["derived"]
    baseline = payload["baseline"]
    server = payload["server"]
    c_conn = d["connect_ms"]
    c_recv = d["recv_ms"]
    c_cls = d["classifier_ms"]
    s_p1 = d["part1_ms"]
    s_send = d["send_ms"]
    total_pipeline = d["total_pipeline_ms"]
    full_mean = d["full_mean_ms"]
    overhead = d["overhead_ms"]
    pct_recv = d["pct_recv"]
    pct_cls = d["pct_cls"]
    pct_conn = d["pct_conn"]

    w = [22, 12, 38]
    sep = "-+-".join("-" * x for x in w)
    print()
    print("========== 真实双进程 AlexNet — 结果汇总表 ==========")
    print(_fmt_row(["阶段", "耗时 (ms)", "说明"], w))
    print(sep)
    print(
        _fmt_row(
            [
                "TCP 连接 (client)",
                f"{c_conn:.2f}",
                "客户端建连（量级通常很小）",
            ],
            w,
        )
    )
    print(
        _fmt_row(
            [
                "接收张量 (client)",
                f"{c_recv:.2f}",
                "含等待对端 part1+发送、TCP、torch.load 反序列化",
            ],
            w,
        )
    )
    print(
        _fmt_row(
            [
                "Classifier (client)",
                f"{c_cls:.2f}",
                "后半段分类器前向",
            ],
            w,
        )
    )
    print(sep)
    print(
        _fmt_row(
            [
                "合计（可相加）",
                f"{total_pipeline:.2f}",
                "connect + recv + classifier",
            ],
            w,
        )
    )
    print()
    print("—— server 侧分项（与上一行「接收」在时间上重叠，勿再相加 ——）")
    print(_fmt_row(["阶段", "耗时 (ms)", "说明"], w))
    print(sep)
    print(_fmt_row(["Part1 计算 (server)", f"{s_p1:.2f}", "features+avgpool+flatten"], w))
    print(_fmt_row(["序列化+发送 (server)", f"{s_send:.2f}", "torch.save + sendall"], w))
    print()
    print("—— 单进程整网基准（同机同输入分布，多次取平均） ——")
    print(
        _fmt_row(
            [
                f"整网一次前向 (×{baseline['runs']})",
                f"{full_mean:.2f} +/- {baseline['stdev_ms']:.2f}",
                "单进程、无 TCP/无二次加载模型开销（与双进程可比）",
            ],
            w,
        )
    )
    print(sep)
    print(
        _fmt_row(
            [
                "差值 (合计 - 整网)",
                f"{overhead:+.2f}",
                ">0 表示多进程与传输带来额外耗时",
            ],
            w,
        )
    )
    print()
    print("========== 结论（可直接写入实验记录 / 论文讨论） ==========")
    print(
        f"1. 本机双进程链路的端到端耗时（连接+接收+分类）约为 {total_pipeline:.2f} ms；"
        f"其中「接收张量」占 {pct_recv:.1f}%（含对端计算与传输），"
        f"「Classifier」占 {pct_cls:.1f}%，「连接」占 {pct_conn:.1f}%。"
    )
    print(
        f"2. 同机单进程整网前向平均约 {full_mean:.2f} ms；双进程合计与整网相差 {overhead:+.2f} ms。"
        + (
            " 说明在当前切分与实现下，多进程、序列化与本机 TCP 引入了额外开销（符合预期）。"
            if overhead > 0
            else " 说明双进程路径未明显高于整网（可能与计时噪声或调度有关，可多次运行取平均）。"
        )
    )
    print(
        "3. server 的 part1 与 send 已包含在 client 的 recv 等待时间内，"
        "表中「合计」仅对 connect / recv / classifier 三列求和，避免重复计算。"
    )
    print("============================================================")
