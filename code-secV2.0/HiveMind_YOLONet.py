# Distributed DNN partition - YOLONet variant
import itertools
from datetime import datetime

import numpy as np
import pandas as pd

# s, first_band: set by caller


class HiveMind:
    """DNN partition optimizer (YOLONet: 52 layers)."""

    LAYER_COUNT = 52

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
        n_half = self.num_devices * (self.num_devices - 1) // 2
        bw_vals = np.random.randint(41, 51, size=n_half)
        self.device_capabilities = np.linspace(41, 60, num=self.num_devices)
        off = 0
        for i in range(self.num_devices - 1):
            for j in range(i + 1, self.num_devices):
                self.bandwidth_matrix[i, j] = bw_vals[off]
                off += 1
        for j in range(self.num_devices - 1):
            for i in range(j + 1, self.num_devices):
                self.bandwidth_matrix[i, j] = self.bandwidth_matrix[j, i]
        df = pd.read_csv("data/YOLONet.csv")
        self.layer_flops = np.array(df["Flops"])
        self.layer_data_sizes = np.array(df["DataSize"])

    def _comp_time(self, lo: int, hi: int, dev: int) -> float:
        return np.sum(self.layer_flops[lo:hi] / self.device_capabilities[s[dev]] / 1e9)

    def _xfer_time(self, layer_idx: int, from_dev: int, to_dev: int) -> float:
        return self.layer_data_sizes[layer_idx] / self.bandwidth_matrix[s[from_dev], s[to_dev]]

    def enhancedDijkstraTime(self):
        self.path_cost = np.full((self.num_devices + 1, self.LAYER_COUNT + 1), np.inf)
        self.best_stop_index = np.zeros((self.num_devices, self.LAYER_COUNT + 1))

        for i in range(self.LAYER_COUNT):
            self.path_cost[self.num_devices, i] = self._comp_time(i, self.LAYER_COUNT, self.num_devices - 1)
            self.best_stop_index[self.num_devices - 1, i] = self.LAYER_COUNT
        self.path_cost[self.num_devices, self.LAYER_COUNT] = 0
        self.best_stop_index[self.num_devices - 1, self.LAYER_COUNT] = -1

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
                    tot = self._comp_time(split_start, split_end, dev_idx) + self._xfer_time(split_end, dev_idx, dev_idx + 1) + self.path_cost[dev_idx + 2, split_end]
                    if tot < best_val:
                        best_val = tot
                        self.best_stop_index[dev_idx, split_start] = -1 if split_start == split_end else split_end
                self.path_cost[dev_idx + 1, split_start] = best_val
                cached_stop = int(self.best_stop_index[dev_idx, split_start])
                cached_start = split_start

        best_total, best_idx = float("inf"), 0
        for i in range(1, self.LAYER_COUNT + 1):
            self.path_cost[1, i] = self._comp_time(0, i, 0) + self._xfer_time(i, 0, 1) + self.path_cost[2, i]
            self.best_stop_index[0, i] = i
            if self.path_cost[1, i] < best_total:
                best_total, best_idx = self.path_cost[1, i], i

        self.path_cost[0, 0] = best_total + self.layer_data_sizes[0] / first_band[s[0]]

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

        if self.path_cost[0, 0] == float("inf"):
            max_tp = 0
        else:
            max_tp = float("inf")
            for i in range(self.num_devices):
                if i == 0:
                    max_tp = 1 / max(self.layer_data_sizes[0] / first_band[s[0]], self._comp_time(0, le[0], 0))
                    continue
                if le[i] == -1:
                    stage_t = self.layer_data_sizes[ls[i] - 1] / self.bandwidth_matrix[s[i - 1], s[i]]
                else:
                    stage_t = max(self.layer_data_sizes[ls[i] - 1] / self.bandwidth_matrix[s[i - 1], s[i]], self._comp_time(ls[i] - 1, le[i], i))
                max_tp = min(max_tp, 1 / stage_t)

        return max_tp, self.path_cost[0, 0]

    def enhancedDijkstraThroughput(self):
        self.path_cost = np.full((self.num_devices + 1, self.LAYER_COUNT + 1), np.inf)
        self.best_stop_index = np.zeros((self.num_devices, self.LAYER_COUNT + 1))
        for i in range(self.LAYER_COUNT):
            comp = self._comp_time(i, self.LAYER_COUNT, self.num_devices - 1)
            comm = self.layer_data_sizes[i] / self.bandwidth_matrix[s[self.num_devices - 2], s[self.num_devices - 1]]
            self.path_cost[self.num_devices, i] = max(comp, comm)
            self.best_stop_index[self.num_devices - 1, i] = self.LAYER_COUNT
        self.path_cost[self.num_devices, self.LAYER_COUNT] = self.layer_data_sizes[self.LAYER_COUNT] / self.bandwidth_matrix[s[self.num_devices - 2], s[self.num_devices - 1]]
        self.best_stop_index[self.num_devices - 1, self.LAYER_COUNT] = -1

        for dev_idx in range(self.num_devices - 2, 0, -1):
            for split_start in range(self.LAYER_COUNT + 1):
                best_val = float("inf")
                for split_end in range(split_start, self.LAYER_COUNT + 1):
                    comp = self._comp_time(split_start, split_end, dev_idx)
                    comm = self.layer_data_sizes[split_start] / self.bandwidth_matrix[s[dev_idx - 1], s[dev_idx]]
                    tot = max(comp, comm, self.path_cost[dev_idx + 2, split_end])
                    if tot < best_val:
                        best_val = tot
                        self.best_stop_index[dev_idx, split_start] = -1 if split_start == split_end else split_end
                self.path_cost[dev_idx + 1, split_start] = best_val

        best_total, best_idx = float("inf"), 0
        for i in range(1, self.LAYER_COUNT + 1):
            comp = self._comp_time(0, i, 0)
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

        if self.path_cost[0, 0] == float("inf"):
            infer_t = float("inf")
        else:
            infer_t = 0
            for i in range(self.num_devices):
                if i == 0:
                    infer_t = self.layer_data_sizes[0] / first_band[s[0]] + self._comp_time(0, le[0], 0)
                    continue
                if le[i] == -1:
                    infer_t += self.layer_data_sizes[ls[i] - 1] / self.bandwidth_matrix[s[i - 1], s[i]]
                else:
                    infer_t += self.layer_data_sizes[ls[i] - 1] / self.bandwidth_matrix[s[i - 1], s[i]] + self._comp_time(ls[i] - 1, le[i], i)
        return 1 / self.path_cost[0, 0], infer_t
