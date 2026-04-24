# Pipeline-based DNN partition - VGG19 variant
import itertools
from datetime import datetime

import numpy as np
import pandas as pd

# s, first_band: set by caller


class Pipeline:
    """Edge pipeline DNN partition (VGG19: 37 layers)."""

    LAYER_COUNT = 37

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
        n_half = self.nd * (self.nd - 1) // 2
        bw_vals = np.random.randint(1, 11, size=n_half)
        self.device_perf = np.linspace(41, 60, num=self.nd)
        off = 0
        for i in range(self.nd - 1):
            for j in range(i + 1, self.nd):
                self.bw_matrix[i, j] = bw_vals[off]
                off += 1
        for j in range(self.nd - 1):
            for i in range(j + 1, self.nd):
                self.bw_matrix[i, j] = self.bw_matrix[j, i]
        df = pd.read_csv("data/vgg19.csv")
        self.flops_per_layer = np.array(df["Flops"])
        self.data_per_layer = np.array(df["DataSize"])

    def _comp_time(self, lo: int, hi: int, dev: int) -> float:
        return np.sum(self.flops_per_layer[lo:hi] / self.device_perf[s[dev]] / 1e9)

    def dynamic_planning(self):
        L, nd = self.LAYER_COUNT, self.nd
        h = np.full((L, nd), np.inf)
        hs = np.zeros((L, nd))
        self.Ti = np.full((L, nd), np.inf)

        for j in range(L):
            t_stage = max(self.data_per_layer[0] / first_band[s[0]], self._comp_time(0, j + 1, 0))
            t_infer = self.data_per_layer[0] / first_band[s[0]] + self._comp_time(0, j + 1, 0)
            h[j, 0] = t_stage
            hs[j, 0] = 0
            self.Ti[j, 0] = t_infer

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

        return 1 / h[L - 1, nd - 1], self.Ti[L - 1, nd - 1]
