# Pipeline-based DNN partition with device ordering
import itertools
from datetime import datetime

import numpy as np
import pandas as pd

# s: device order (set by caller)
# first_band: first-hop bandwidth (set by caller)


class Pipeline:
    """Edge pipeline DNN partition via dynamic programming."""

    LAYER_COUNT = 13

    def __init__(self, num_devices: int):
        self.nd = num_devices
        self.bw_matrix = np.zeros((num_devices, num_devices))
        self.flops_per_layer = np.zeros(0)
        self.data_per_layer = np.zeros(0)
        self.device_perf = np.zeros(0)
        self.Ls = []
        self.Le = []
        self.Ti = np.zeros(1)

    @property
    def deviceNumber(self):
        return self.nd

    @property
    def band(self):
        return self.bw_matrix

    @band.setter
    def band(self, val):
        self.bw_matrix = val

    @property
    def cl(self):
        return self.flops_per_layer

    @cl.setter
    def cl(self, val):
        self.flops_per_layer = val

    @property
    def dl(self):
        return self.data_per_layer

    @dl.setter
    def dl(self, val):
        self.data_per_layer = val

    @property
    def fn(self):
        return self.device_perf

    @fn.setter
    def fn(self, val):
        self.device_perf = val

    @property
    def L(self):
        return self.LAYER_COUNT

    def assignment(self):
        """Load device and layer parameters."""
        n_half = self.nd * (self.nd - 1) // 2
        bw_vals = np.random.randint(11, 21, size=n_half)
        self.device_perf = np.linspace(41, 60, num=self.nd)

        off = 0
        for i in range(self.nd - 1):
            for j in range(i + 1, self.nd):
                self.bw_matrix[i, j] = bw_vals[off]
                off += 1
        for j in range(self.nd - 1):
            for i in range(j + 1, self.nd):
                self.bw_matrix[i, j] = self.bw_matrix[j, i]

        df = pd.read_csv("data/AlexNet.csv")
        self.flops_per_layer = np.array(df["Flops"])
        self.data_per_layer = np.array(df["DataSize"])
        print("设备性能(GFlops/s):", self.device_perf)
        print("设备传输带宽(MB/s):", self.bw_matrix)
        print("DNN层数为:", self.LAYER_COUNT)

    def _comp_time(self, lo: int, hi: int, dev: int) -> float:
        return np.sum(self.flops_per_layer[lo:hi] / self.device_perf[s[dev]] / 1e9)

    def dynamic_planning(self):
        """DP-based partition optimization."""
        L, nd = self.LAYER_COUNT, self.nd
        h = np.full((L, nd), np.inf)
        hs = np.zeros((L, nd))
        self.Ti = np.full((L, nd), np.inf)

        # First device
        for j in range(L):
            t_stage = max(
                self.data_per_layer[0] / first_band[s[0]],
                self._comp_time(0, j + 1, 0)
            )
            t_infer = self.data_per_layer[0] / first_band[s[0]] + self._comp_time(0, j + 1, 0)
            h[j, 0] = t_stage
            hs[j, 0] = 0
            self.Ti[j, 0] = t_infer

        # Remaining devices
        for i in range(1, nd):
            for j in range(L):
                best_kt = 0
                for k in range(j + 1):
                    tcomp = self._comp_time(k + 1, j + 1, i)
                    tcomm = self.data_per_layer[k + 1] / self.bw_matrix[s[i - 1], s[i]]
                    t_stage = max(tcomp, tcomm)
                    t_infer = tcomp + tcomm
                    cand = max(h[k, i - 1], t_stage)
                    if cand < h[j, i]:
                        h[j, i] = cand
                        best_kt = k
                        self.Ti[j, i] = self.Ti[k, i - 1] + t_infer
                hs[j, i] = best_kt + 1

        # Extract partition
        ls = [0] * nd
        le = [0] * nd
        kt = 0
        for i in range(nd - 1):
            rev = nd - i - 1
            if i == 0:
                if hs[L - 1, rev] < L:
                    ls[rev] = int(hs[L - 1, rev]) + 1
                    le[rev] = L
                else:
                    ls[rev], le[rev] = 0, 0
                kt = int(hs[L - 1, rev]) - 1
            else:
                if hs[int(kt), rev] < kt + 1:
                    ls[rev] = int(hs[int(kt), rev]) + 1
                    le[rev] = int(kt) + 1
                else:
                    ls[rev], le[rev] = 0, 0
                kt = int(hs[int(kt), rev]) - 1
        ls[0], le[0] = 1, int(kt) + 1
        ls = list(map(int, ls))
        le = list(map(int, le))
        self.Ls, self.Le = ls, le
        print("流水线上每个设备的起始层:", ls)
        print("流水线上每个设备的结束层:", le)

        throughput = 1 / h[L - 1, nd - 1]
        latency = self.Ti[L - 1, nd - 1]
        return throughput, latency

    def average_splitting(self):
        """Uniform layer split baseline."""
        n = self.LAYER_COUNT // self.nd
        r = self.LAYER_COUNT % self.nd
        tha = max(
            int(np.sum(self.flops_per_layer[0:n + 1] / self.device_perf[s[0]] / 1e9)),
            self.data_per_layer[0] / first_band[s[0]]
        )
        tia = np.sum(self.flops_per_layer[0:n + 1] / self.device_perf[s[0]] / 1e9) + self.data_per_layer[n] / first_band[s[0]]
        for i in range(1, r):
            tha = max(
                int(np.sum(self.flops_per_layer[i * (n + 1):i * (n + 1) + (n + 1)] / self.device_perf[s[i]] / 1e9)),
                self.data_per_layer[i * (n + 1)] / self.bw_matrix[s[i - 1], s[i]],
                tha
            )
            tia += np.sum(self.flops_per_layer[i * (n + 1):i * (n + 1) + (n + 1)] / self.device_perf[s[i]] / 1e9) + self.data_per_layer[i * (n + 1)] / self.bw_matrix[s[i - 1], s[i]]
        for i in range(r, self.nd):
            temp = max(
                int(np.sum(self.flops_per_layer[i * n:i * n + n] / self.device_perf[s[i]] / 1e9)),
                self.data_per_layer[i * n] / self.bw_matrix[s[i - 1], s[i]],
                tha
            )
            tha = max(tha, temp)
            tia += np.sum(self.flops_per_layer[i * n:i * n + n] / self.device_perf[s[i]] / 1e9) + self.data_per_layer[i * n] / self.bw_matrix[s[i - 1], s[i]]
        return 1 / tha, tia

    def random_splitting(self):
        """Random partition baseline."""
        endpoints = np.sort(np.random.choice(range(1, self.LAYER_COUNT), self.nd - 1, replace=False))
        endpoints = np.append(endpoints, self.LAYER_COUNT)
        tha = max(
            int(np.sum(self.flops_per_layer[0:endpoints[0]] / self.device_perf[s[0]] / 1e9)),
            self.data_per_layer[0] / first_band[s[0]]
        )
        tia = np.sum(self.flops_per_layer[0:endpoints[0]] / self.device_perf[s[0]] / 1e9) + self.data_per_layer[0] / first_band[s[0]]
        for i in range(1, self.nd):
            tha = max(
                int(np.sum(self.flops_per_layer[endpoints[i - 1]:endpoints[i]] / self.device_perf[s[0]] / 1e9)),
                self.data_per_layer[i] / first_band[s[i]],
                tha
            )
            tia += np.sum(self.flops_per_layer[endpoints[i - 1]:endpoints[i]] / self.device_perf[s[0]] / 1e9) + self.data_per_layer[i] / first_band[s[i]]
        return 1 / tha, tia


if __name__ == '__main__':
    N = 5
    np.random.seed(1)
    first_band = np.random.randint(1, 50, size=N)
    print("卸载带宽:", first_band)
    pipe = Pipeline(N)
    pipe.assignment()
    s = list(range(N))
    P = list(itertools.permutations(s))

    start = datetime.now()
    th1, ti1, ni1 = 0, float("inf"), 0
    th2, ti2, ni2 = float("inf"), 0, 0
    for m in range(len(P)):
        s = np.array(P[m])
        th, ti = pipe.dynamic_planning()
        if th > th1 or (th == th1 and ti < ti1):
            th1, ti1, ni1 = th, ti, m
        if th < th2:
            th2, ti2, ni2 = th, ti, m

    print("DP分区下最优排序:", ni1, P[ni1])
    print("吞吐量:", th1)
    print("单个推理时间:%fs" % ti1)
    print("DP分区下最差排序:", ni2, P[ni2])
    print("吞吐量:", th2)
    print("单个推理时间:%fs" % ti2)
    print("程序运行时间：", (datetime.now() - start).total_seconds())
