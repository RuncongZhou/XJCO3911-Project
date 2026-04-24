# Distributed DNN partition and device ordering - single device per stage
import itertools
from datetime import datetime

import numpy as np
import pandas as pd

# s: device order array (set by caller)
# first_band: first-hop bandwidth per device (set by caller)


class HiveMind:
    """DNN partition optimizer using shortest-path formulation."""

    LAYER_COUNT = 13  # AlexNet feature layers

    def __init__(self, num_devices: int):
        self.num_devices = num_devices
        self.bandwidth_matrix = np.zeros((num_devices, num_devices))
        self.layer_flops = np.zeros(0)
        self.layer_data_sizes = np.zeros(0)
        self.device_capabilities = np.zeros(0)
        self.path_cost = np.full((num_devices + 1, self.LAYER_COUNT + 1), np.inf)
        self.best_stop_index = np.zeros((num_devices, self.LAYER_COUNT + 1))
        self.Ls = []
        self.Le = []
        self.Ti = np.zeros(1)

    @property
    def band(self):
        return self.bandwidth_matrix

    @band.setter
    def band(self, val):
        self.bandwidth_matrix = val

    @property
    def cl(self):
        return self.layer_flops

    @cl.setter
    def cl(self, val):
        self.layer_flops = val

    @property
    def dl(self):
        return self.layer_data_sizes

    @dl.setter
    def dl(self, val):
        self.layer_data_sizes = val

    @property
    def fn(self):
        return self.device_capabilities

    @fn.setter
    def fn(self, val):
        self.device_capabilities = val

    @property
    def cost(self):
        return self.path_cost

    @cost.setter
    def cost(self, val):
        self.path_cost = val

    @property
    def npi(self):
        return self.best_stop_index

    @npi.setter
    def npi(self, val):
        self.best_stop_index = val

    @property
    def L(self):
        return self.LAYER_COUNT

    def assignment(self):
        """Initialize device specs and load layer profile."""
        n_half = self.num_devices * (self.num_devices - 1) // 2
        bw_vals = np.random.randint(21, 31, size=n_half)
        self.device_capabilities = np.linspace(41, 60, num=self.num_devices)

        off = 0
        for i in range(self.num_devices - 1):
            for j in range(i + 1, self.num_devices):
                self.bandwidth_matrix[i, j] = bw_vals[off]
                off += 1
        for j in range(self.num_devices - 1):
            for i in range(j + 1, self.num_devices):
                self.bandwidth_matrix[i, j] = self.bandwidth_matrix[j, i]

        df = pd.read_csv("data/AlexNet.csv")
        self.layer_flops = np.array(df["Flops"])
        self.layer_data_sizes = np.array(df["DataSize"])
        print("设备性能(GFlops/s):", self.device_capabilities)
        print("设备传输带宽(MB/s):", self.bandwidth_matrix)
        print("DNN层数为:", self.LAYER_COUNT)

    def _compute_flop_time(self, start_idx: int, end_idx: int, dev_idx: int) -> float:
        return np.sum(self.layer_flops[start_idx:end_idx] / self.device_capabilities[s[dev_idx]] / 1e9)

    def _compute_transfer_time(self, layer_idx: int, from_dev: int, to_dev: int) -> float:
        return self.layer_data_sizes[layer_idx] / self.bandwidth_matrix[s[from_dev], s[to_dev]]

    def enhancedDijkstraTime(self):
        """Optimize partition for minimum inference latency."""
        self.path_cost = np.full((self.num_devices + 1, self.LAYER_COUNT + 1), np.inf)
        self.best_stop_index = np.zeros((self.num_devices, self.LAYER_COUNT + 1))

        # Last device
        for i in range(self.LAYER_COUNT):
            self.path_cost[self.num_devices, i] = self._compute_flop_time(i, self.LAYER_COUNT, self.num_devices - 1)
            self.best_stop_index[self.num_devices - 1, i] = self.LAYER_COUNT
        self.path_cost[self.num_devices, self.LAYER_COUNT] = 0
        self.best_stop_index[self.num_devices - 1, self.LAYER_COUNT] = -1

        # Middle devices (backward)
        for dev_idx in range(self.num_devices - 2, 0, -1):
            cached_stop, cached_start = -1, 0
            for split_start in range(self.LAYER_COUNT + 1):
                if cached_stop > split_start:
                    self.path_cost[dev_idx + 1, split_start] = (
                        self.path_cost[dev_idx + 1, cached_start]
                        - np.sum(self.layer_flops[cached_start:split_start] / self.device_capabilities[s[dev_idx]] / 1e9)
                    )
                    self.best_stop_index[dev_idx, split_start] = cached_stop
                    continue
                best_val = float("inf")
                for split_end in range(split_start, self.LAYER_COUNT + 1):
                    comp = self._compute_flop_time(split_start, split_end, dev_idx)
                    trans = self._compute_transfer_time(split_end, dev_idx, dev_idx + 1)
                    total = comp + trans + self.path_cost[dev_idx + 2, split_end]
                    if total < best_val:
                        best_val = total
                        self.best_stop_index[dev_idx, split_start] = -1 if split_start == split_end else split_end
                self.path_cost[dev_idx + 1, split_start] = best_val
                cached_stop = int(self.best_stop_index[dev_idx, split_start])
                cached_start = split_start

        # First device
        best_total, best_idx = float("inf"), 0
        for i in range(1, self.LAYER_COUNT + 1):
            comp = self._compute_flop_time(0, i, 0)
            trans = self._compute_transfer_time(i, 0, 1)
            self.path_cost[1, i] = comp + trans + self.path_cost[2, i]
            self.best_stop_index[0, i] = i
            if self.path_cost[1, i] < best_total:
                best_total, best_idx = self.path_cost[1, i], i

        first_hop_time = self.layer_data_sizes[0] / first_band[s[0]]
        self.path_cost[0, 0] = best_total + first_hop_time

        # Build partition
        ls = [0] * self.num_devices
        le = [0] * self.num_devices
        ls[0], le[0] = 1, best_idx
        idx = best_idx
        for i in range(1, self.num_devices):
            le[i] = int(self.best_stop_index[i, idx])
            ls[i] = idx + 1
            if le[i] != -1:
                idx = int(le[i])
        ls = list(map(int, ls))
        le = list(map(int, le))
        print("流水线上每个设备的起始层:", ls)
        print("流水线上每个设备的结束层:", le)

        # Throughput
        if self.path_cost[0, 0] == float("inf"):
            max_throughput = 0
        else:
            max_throughput = float("inf")
            for i in range(self.num_devices):
                if i == 0:
                    max_throughput = 1 / max(
                        self.layer_data_sizes[0] / first_band[s[0]],
                        self._compute_flop_time(0, le[0], 0)
                    )
                    continue
                if le[i] == -1:
                    stage_time = self.layer_data_sizes[ls[i] - 1] / self.bandwidth_matrix[s[i - 1], s[i]]
                else:
                    stage_time = max(
                        self.layer_data_sizes[ls[i] - 1] / self.bandwidth_matrix[s[i - 1], s[i]],
                        self._compute_flop_time(ls[i] - 1, le[i], i)
                    )
                max_throughput = min(max_throughput, 1 / stage_time)

        return max_throughput, self.path_cost[0, 0]

    def enhancedDijkstraThroughput(self):
        """Optimize partition for maximum throughput."""
        self.path_cost = np.full((self.num_devices + 1, self.LAYER_COUNT + 1), np.inf)
        self.best_stop_index = np.zeros((self.num_devices, self.LAYER_COUNT + 1))

        for i in range(self.LAYER_COUNT):
            comp = self._compute_flop_time(i, self.LAYER_COUNT, self.num_devices - 1)
            comm = self.layer_data_sizes[i] / self.bandwidth_matrix[s[self.num_devices - 2], s[self.num_devices - 1]]
            self.path_cost[self.num_devices, i] = max(comp, comm)
            self.best_stop_index[self.num_devices - 1, i] = self.LAYER_COUNT
        self.path_cost[self.num_devices, self.LAYER_COUNT] = (
            self.layer_data_sizes[self.LAYER_COUNT] / self.bandwidth_matrix[s[self.num_devices - 2], s[self.num_devices - 1]]
        )
        self.best_stop_index[self.num_devices - 1, self.LAYER_COUNT] = -1

        for dev_idx in range(self.num_devices - 2, 0, -1):
            for split_start in range(self.LAYER_COUNT + 1):
                best_val = float("inf")
                for split_end in range(split_start, self.LAYER_COUNT + 1):
                    comp = self._compute_flop_time(split_start, split_end, dev_idx)
                    comm = self.layer_data_sizes[split_start] / self.bandwidth_matrix[s[dev_idx - 1], s[dev_idx]]
                    total = max(comp, comm, self.path_cost[dev_idx + 2, split_end])
                    if total < best_val:
                        best_val = total
                        self.best_stop_index[dev_idx, split_start] = -1 if split_start == split_end else split_end
                self.path_cost[dev_idx + 1, split_start] = best_val

        best_total, best_idx = float("inf"), 0
        for i in range(1, self.LAYER_COUNT + 1):
            comp = self._compute_flop_time(0, i, 0)
            comm0 = self.layer_data_sizes[0] / first_band[s[0]]
            self.path_cost[1, i] = max(comp, comm0, self.path_cost[2, i])
            self.best_stop_index[0, i] = i
            if self.path_cost[1, i] < best_total:
                best_total, best_idx = self.path_cost[1, i], i
        self.path_cost[0, 0] = best_total

        ls = [0] * self.num_devices
        le = [0] * self.num_devices
        ls[0], le[0] = 1, best_idx
        idx = best_idx
        for i in range(1, self.num_devices):
            le[i] = int(self.best_stop_index[i, idx])
            ls[i] = idx + 1
            if le[i] != -1:
                idx = int(le[i])
        ls = list(map(int, ls))
        le = list(map(int, le))
        print("流水线上每个设备的起始层:", ls)
        print("流水线上每个设备的结束层:", le)

        if self.path_cost[0, 0] == float("inf"):
            infer_time = float("inf")
        else:
            infer_time = 0
            for i in range(self.num_devices):
                if i == 0:
                    infer_time = self.layer_data_sizes[0] / first_band[s[0]] + self._compute_flop_time(0, le[0], 0)
                    continue
                if le[i] == -1:
                    infer_time += self.layer_data_sizes[ls[i] - 1] / self.bandwidth_matrix[s[i - 1], s[i]]
                else:
                    infer_time += (
                        self.layer_data_sizes[ls[i] - 1] / self.bandwidth_matrix[s[i - 1], s[i]]
                        + self._compute_flop_time(ls[i] - 1, le[i], i)
                    )
        return 1 / self.path_cost[0, 0], infer_time


if __name__ == '__main__':
    N = 5
    np.random.seed(1)
    first_band = np.random.randint(1, 50, size=N)
    first_band[np.random.randint(N - 1, size=N // 2)] = 0.01
    print("卸载带宽:", first_band)
    engine = HiveMind(N)
    engine.assignment()
    s = list(range(N))
    P = list(itertools.permutations(s))
    print("所有排序组合:", P)

    start = datetime.now()
    th1, ti1, ni1 = 0, float("inf"), 0
    th2, ti2, ni2 = float("inf"), 0, 0
    for m in range(len(P)):
        s = np.array(P[m])
        th, ti = engine.enhancedDijkstraTime()
        if ti < ti1 or (ti == ti1 and th > th1):
            th1, ti1, ni1 = th, ti, m
        if ti > ti2:
            th2, ti2, ni2 = th, ti, m

    print("HiveMind最小推理时延下最优排序:", ni1, P[ni1])
    print("吞吐量:", th1)
    print("单个推理时间:%fs" % ti1)
    print("HiveMind最小推理时延下最差排序:", ni2, P[ni2])
    print("吞吐量:", th2)
    print("单个推理时间:%fs" % ti2)
    print("程序运行时间：", (datetime.now() - start).total_seconds())
