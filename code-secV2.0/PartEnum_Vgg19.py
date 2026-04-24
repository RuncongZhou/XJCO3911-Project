# Stage-wise partition by full enumeration (Vgg19)
from datetime import datetime
import numpy as np
import pandas as pd


class StagePartition:
    """DP-based stage partition for Vgg19 (conv/pool height rules)."""

    def __init__(self):
        self.bw_matrix = np.zeros((N, N))
        self.flops_per_layer = np.zeros(0)
        self.data_per_layer = np.zeros(0)  # 每一层输入数据大小
        self.layer_height = np.zeros(0)  # 每一层的层高度
        self.layer_type = np.zeros(0)  # 每一层的层类型
        self.device_perf = np.zeros(0)  # 设备性能按由小到大排序，索引对应设备名
        self.dev_stage_map = np.zeros((1, N))  # 索引对应设备号，值对应设备在管道所处的阶段
        self.num_stages = 1  # 当前粒子对应管道阶段数，初始为1
        self.row_start = np.zeros((N, L))  # 设备负责每一层输入特征起始行数
        self.row_end = np.zeros((N, L))  # 设备负责每一层输入特征的终止行数
        self.stage_time_mat = np.zeros(1)  # 动态规划矩阵:j层DNN在前i个阶段上执行的最慢阶段时间
        self.split_idx_mat = np.zeros(1)  # 动态规划矩阵:最优解对应的前i个阶段上执行的层数
        self.infer_time_mat = np.zeros(1)  # 动态规划矩阵:j层DNN在前i个阶段上执行的最短推理时间

    def load_config(self):
        temp = np.random.randint(21, 31, size=int(N * (N - 1) / 2))
        self.device_perf = np.linspace(81, 100, num=N)
        print("Device perf:", self.device_perf.shape)
        t1, t2 = 0, 0
        for i in range(N - 1):
            for j in range(i + 1, N):
                self.bw_matrix[i, j] = temp[t1]
                t1 += 1
        for j in range(N - 1):
            for i in range(j + 1, N):
                self.bw_matrix[i, j] = temp[t2]
                t2 += 1
        print("Link bandwidth:", self.bw_matrix.shape)
        df = pd.read_csv("data/vgg19.csv")
        self.flops_per_layer = np.array(df["Flops"])
        self.data_per_layer = np.array(df["DataSize"])
        self.layer_height = np.array(df["height"])
        self.layer_type = np.array(df["type"])

        # with open('data\\vgg19.csv','r') as csvfile:
        #     reader = csv.DictReader(csvfile)
        #     self.flops_per_layer = [row['Flops'] for row in reader]
        #     self.data_per_layer = [row['DataSize'] for row in reader]
        #     该函数返回的结果遍历一次之后，再次遍历返回的结果是空列表。
        # print(self.flops_per_layer)
        # print(self.data_per_layer)

    def align_stage_order(self):
        self.num_stages = 1  # 每次要更新为初始值
        # print("posInt", self.posInt, self.posInt.shape)
        idx = np.argsort(posInt)  # 值越小的设备属于阶段越靠前
        # print("idx:", idx, idx.shape)
        if not type(idx) is np.ndarray:
            idx = np.array(idx)
        temp = posInt[0, idx[0, 0]]  # 当前阶段对应的值
        self.dev_stage_map[0, idx[0, 0]] = self.num_stages
        for i in range(1, idx.size):
            if posInt[0, idx[0, i]] == temp:
                self.dev_stage_map[0, idx[0, i]] = self.num_stages
            else:  # 位置不相等则在管道下一阶段,stage先自增再赋值
                temp = posInt[0, idx[0, i]]
                self.num_stages += 1
                self.dev_stage_map[0, idx[0, i]] = self.num_stages

    def run_dp(self):
        curr_stage = np.array(np.where(self.dev_stage_map.reshape(-1) == 1)).reshape(-1)
        # print("curr_stage", curr_stage, curr_stage.shape)
        curr_num = curr_stage.size
        for j in range(L):
            if curr_num == 1:  # 第一阶段只有一个设备
                # 计算DNN执行时间时注意要将cl值从Flops转换为GFlops
                tcomm = self.data_per_layer[0]/first_band[curr_stage[0]]
                tcomp = np.sum(self.flops_per_layer[0:j + 1] / self.device_perf[curr_stage[0]] / 1e9)
                t_stage = max(tcomp, tcomm)        # 当前阶段时间
                t_infer = tcomp + tcomm            # 单个输入阶段推理时间
                self.stage_time_mat[j, 0] = t_stage             # 通信时间和计算时间去最大值为阶段时间
                self.split_idx_mat[j, 0] = 0                  # 初始化为0而不是1,索引从0开始的
                self.infer_time_mat[j, 0] = t_infer            # 第一阶段推理时间
            else:  # 第一阶段多个设备，按照计算能力进行层内拆分
                # 得出每个设备性能比例，根据当前阶段最后一层进行划分，并回溯至当前阶段第一层
                # 独立划分最后一层的输出特征,并转为输入特征范围，第一阶段不考虑通信时间，计算负载直接按比例分配
                arr_device = np.zeros(curr_num)  # 存储每个设备对于当前阶段分配的计算量
                arr_fn = [self.device_perf[curr_stage[i]] for i in range(curr_num)]  # 当前阶段设备性能
                sum_fn = sum(arr_fn)
                index = 0  # 起始行数为第一行
                for i in range(curr_num):
                    self.row_start[curr_stage[i], j] = index  # 分配给设备curr_stage[i]的起始行数
                    index = index + np.round(arr_fn[i] / sum_fn * self.layer_height[j])
                    if index <= self.layer_height[j]:  # 分配给设备curr_stage[i]的终止行数
                        self.row_end[curr_stage[i], j] = index
                    else:
                        index = self.row_end[curr_stage[i], j] = self.layer_height[j]  # 下一个设备的Hs要更新为self.layer_height[j]
                # 激活层和批归一化层输入输出一致
                if self.layer_type[j] == 3:  # 卷积层
                    self.row_start[curr_stage, j] = self.row_start[curr_stage, j] - 1  # 对于Vgg19
                    self.row_end[curr_stage, j] = self.row_end[curr_stage, j] + 1
                if self.layer_type[j] == 2:  # 池化层
                    self.row_start[curr_stage, j] = self.row_start[curr_stage, j] * 2  # 对于Vgg19
                    self.row_end[curr_stage, j] = self.row_end[curr_stage, j] * 2
                arr_device += (self.row_end[curr_stage, j] - self.row_start[curr_stage, j]) / self.layer_height[j] * self.flops_per_layer[j]
                # 填充数据不需要传输，Hs，He中小于1的赋值1，大于hlayer的赋值为hlayer
                self.row_start[curr_stage, j]=np.clip(self.row_start[curr_stage, j], a_min=0, a_max=10000)
                self.row_end[curr_stage, j]=np.clip(self.row_end[curr_stage, j], a_min=-10000, a_max=self.layer_height[j])
                # 从第j-1层回溯至第1层，得到每一层设备所需输入范围
                for m in range(1, j + 1):  # j-m=0->m=j+1
                    if self.layer_type[j - m] == 3:  # 卷积层
                        self.row_start[curr_stage, j - m] = self.row_start[curr_stage, j - m + 1] - 1  # 对于Vgg19
                        self.row_end[curr_stage, j - m] = self.row_end[curr_stage, j - m + 1] + 1
                    if self.layer_type[j - m] == 2:  # 池化层
                        self.row_start[curr_stage, j - m] = self.row_start[curr_stage, j - m + 1] * 2  # 对于Vgg19
                        self.row_end[curr_stage, j - m] = self.row_end[curr_stage, j - m + 1] * 2
                    # 激活层输入输出不变
                    if self.layer_type[j - m] == 1:  # 激活层
                        # 对于Vgg19
                        self.row_start[curr_stage, j - m] = self.row_start[curr_stage, j - m + 1]
                        self.row_end[curr_stage, j - m] = self.row_end[curr_stage, j - m + 1]

                    arr_device += (self.row_end[curr_stage, j - m] - self.row_start[curr_stage, j - m]) / self.layer_height[j - m] * self.flops_per_layer[j - m]
                    self.row_start[curr_stage, j-m]=np.clip(self.row_start[curr_stage, j - m], a_min=0, a_max=10000)
                    self.row_end[curr_stage, j-m]=np.clip(self.row_end[curr_stage, j - m], a_min=-10000, a_max=self.layer_height[j - m])
                # 按照当前阶段第一层每个设备的输入范围得到最大传输时间
                tcomm = 0
                for n in range(curr_num):
                    temp = self.data_per_layer[0] * (self.row_end[curr_stage[n], 0] - self.row_start[curr_stage[n], 0]) / \
                           self.layer_height[0] / first_band[curr_stage[n]]
                    if temp > tcomm:
                        tcomm = temp
                # 当前阶段的计算时间，取计算时间最大的设备时间
                tcomp = 0
                for n in range(curr_num):
                    temp = arr_device[n] / arr_fn[n] / 1e9
                    if temp > tcomp:
                        tcomp = temp
                t_stage = max(tcomp, tcomm)  # 当前阶段时间
                t_infer = tcomp + tcomm  # 单个输入阶段推理时间
                # 每个设备计算时间几乎相同，因为按照计算性能进行行数拆分
                self.stage_time_mat[j, 0] = t_stage
                self.split_idx_mat[j, 0] = 0  # 初始化为0而不是1,索引从0开始的
                self.infer_time_mat[j, 0] = t_infer
        #print("第一阶段执行所有层吞吐量:",self.stage_time_mat[L-1, 0])

        # 动态规划，从第二个阶段一直遍历到最后一个阶段
        for s in range(1, self.num_stages):
            # 计算第一个阶段处理前j层的最大周期，采用了自适应层内拆分
            curr_stage = np.array(np.where(self.dev_stage_map.reshape(-1) == (s + 1))).reshape(-1)
            # print("curr_stag:",curr_stage)
            prev_stage = np.array(np.where(self.dev_stage_map.reshape(-1) == s)).reshape(-1)  # 上一阶段设备
            # print("prev_stage",prev_stage)
            curr_num = curr_stage.size
            prev_num = prev_stage.size
            for j in range(L):  # 第一阶段肯定有执行层，不是中继设备，输入数据很小
                if curr_num > 1 and prev_num > 1:
                    print("Skip this ordering")
                    return 0
                for k in range(j + 1):  # 前s-1个阶段处理前(k+1)层的情况，遍历得到最优k值
                    # 首先通过拆分当前阶段最后一层求出每个设备的max(通信,计算)时间，然后得到最慢设备时间，即当前阶段时间
                    if curr_num == 1:  # 当前阶段只有一个设备，则前一阶段可能为多个设备
                        arr_fn = [self.device_perf[prev_stage[i]] for i in range(prev_num)]  # 上一阶段设备性能
                        sum_fn = sum(arr_fn)
                        tcomm = 0
                        # 按照前一阶段设备的输出范围等比分配传输数据大小,计算通信大概时间
                        for n in range(prev_stage.size):
                            temp = self.data_per_layer[k + 1] * arr_fn[n] / sum_fn / self.bw_matrix[prev_stage[n], curr_stage[0]]
                            if temp > tcomm:
                                tcomm = temp
                        # 计算DNN执行时间时注意要将cl值从Flops转换为GFlops
                        if k < j:
                            tcomp = np.sum(self.flops_per_layer[k + 1:j + 1]) / self.device_perf[curr_stage[0]] / 1e9  # 当前阶段的计算时间
                        else:  # 当前阶段仅仅充当中继节点
                            tcomp = 0
                        t_stage = max(tcomp, tcomm)  # 当前阶段时间
                        t_infer = tcomp + tcomm  # 单个输入阶段推理时间
                        if max(self.stage_time_mat[k, s - 1], t_stage) < self.stage_time_mat[j, s]:  # 暂时没考虑不同k值吞吐量相等=的情况（约束中相同优先取单个推理时间小值）
                            self.stage_time_mat[j, s] = max(self.stage_time_mat[k, s - 1], t_stage)
                            self.split_idx_mat[j, s] = k + 1  # 层数从零开始要+1
                            self.infer_time_mat[j, s] = self.infer_time_mat[k, s - 1] + t_infer
                    else:  # 当前阶段为多个设备，则不可能为中继节点，前一阶段为一个设备
                        # 得出每个设备性能比例，根据当前阶段最后一层进行划分，并回溯至当前阶段第一层才能计算传输时间
                        # 独立划分最后一层的输出特征,并转为输入特征范围
                        arr_device = np.zeros(curr_num)  # 存储每个设备对于当前阶段分配的计算量
                        arr_fn = [self.device_perf[curr_stage[i]] for i in range(curr_num)]  # 当前阶段设备性能
                        sum_fn = sum(arr_fn)
                        index = 0  # 起始行数为第一行
                        for i in range(curr_num):
                            self.row_start[curr_stage[i], j] = index  # 分配给设备curr_stage[i]的起始行数
                            index = index + np.round(arr_fn[i] / sum_fn * self.layer_height[j])
                            if index <= self.layer_height[j]:  # 分配给设备curr_stage[i]的终止行数
                                self.row_end[curr_stage[i], j] = index
                            else:
                                index = self.row_end[curr_stage[i], j] = self.layer_height[j]  # 下一个设备的Hs要更新为self.layer_height[j]
                        if k < j:  # 当前阶段参与计算
                            # 激活层和批归一化层输入输出一致
                            if self.layer_type[j] == 3:  # 卷积层
                                # 对于Vgg19
                                self.row_start[curr_stage, j] = self.row_start[curr_stage, j] - 1
                                self.row_end[curr_stage, j] = self.row_end[curr_stage, j] + 1
                            if self.layer_type[j] == 2:  # 池化层
                                # 对于Vgg19
                                self.row_start[curr_stage, j] = self.row_start[curr_stage, j] * 2
                                self.row_end[curr_stage, j] = self.row_end[curr_stage, j] * 2
                            arr_device += (self.row_end[curr_stage, j] - self.row_start[curr_stage, j]) / self.layer_height[j] * self.flops_per_layer[j]
                            # 填充数据不需要传输，Hs，He中小于1的赋值1，大于hlayer的赋值为hlayer
                            self.row_start[curr_stage, j] = np.clip(self.row_start[curr_stage, j], a_min=0, a_max=10000)
                            self.row_end[curr_stage, j] = np.clip(self.row_end[curr_stage, j], a_min=-10000,a_max=self.layer_height[j])
                            # 计算DNN执行时间时注意要将cl值从Flops转换为GFlops
                            # 从第j-1层回溯至第(k+1)层，得到每一层设备所需输入范围
                            for m in range(1, j - k):  # j-m=k+1->m=j-k-1
                                if self.layer_type[j - m] == 3:  # 卷积层
                                    # 对于Vgg19
                                    self.row_start[curr_stage, j - m] = self.row_start[curr_stage, j - m + 1] - 1
                                    self.row_end[curr_stage, j - m] = self.row_end[curr_stage, j - m + 1] + 1
                                if self.layer_type[j - m] == 2:  # 池化层
                                    # 对于Vgg19
                                    self.row_start[curr_stage, j - m] = self.row_start[curr_stage, j - m + 1] * 2
                                    self.row_end[curr_stage, j - m] = self.row_end[curr_stage, j - m + 1] * 2
                                # 激活层输入输出不变
                                if self.layer_type[j - m] == 1:  # 激活层
                                    # 对于Vgg19
                                    self.row_start[curr_stage, j - m] = self.row_start[curr_stage, j - m + 1]
                                    self.row_end[curr_stage, j - m] = self.row_end[curr_stage, j - m + 1]
                                arr_device += (self.row_end[curr_stage, j - m] - self.row_start[curr_stage, j - m]) / self.layer_height[
                                    j - m] * self.flops_per_layer[j - m]
                                self.row_start[curr_stage, j - m] = np.clip(self.row_start[curr_stage, j - m], a_min=0, a_max=10000)
                                self.row_end[curr_stage, j - m] = np.clip(self.row_end[curr_stage, j - m], a_min=-10000,a_max=self.layer_height[j - m])
                            # 按照当前阶段第一层每个设备的输入范围得到最大传输时间
                            tcomm = 0
                            for n in range(curr_num):
                                temp = self.data_per_layer[k + 1] * (
                                            self.row_end[curr_stage[n], k + 1] - self.row_start[curr_stage[n], k + 1]) / \
                                       self.layer_height[k + 1] / self.bw_matrix[prev_stage[0], curr_stage[n]]
                                if temp > tcomm:
                                    tcomm = temp
                            # 当前阶段的计算时间，每个设备计算时间几乎相同
                            # tcomp = np.sum(self.flops_per_layer[k + 1:j + 1]) * arr_fn[0] / sum_fn / arr_fn[0] / 1e9
                            tcomp = 0
                            for n in range(curr_num):
                                temp = arr_device[n] / arr_fn[n] / 1e9
                                if temp > tcomp:
                                    tcomp = temp
                            t_stage = max(tcomp, tcomm)  # 当前阶段时间
                            t_infer = tcomp + tcomm  # 单个输入阶段推理时间
                            # print("self.stage_time_mat[k, s - 1]",self.stage_time_mat[k, s - 1])
                            # print("self.stage_time_mat[j, s]",self.stage_time_mat[j, s])
                        else:  # 当前阶段仅仅充当中继节点
                            # 按照当前阶段第一层每个设备的输入范围得到最大传输时间
                            tcomm = 0
                            for n in range(curr_stage.size):
                                temp = self.data_per_layer[k + 1] * (self.row_end[curr_stage[n], k] - self.row_start[curr_stage[n], k]) / \
                                       self.layer_height[k] / self.bw_matrix[curr_stage[0], curr_stage[n]]
                                if temp > tcomm:
                                    tcomm = temp
                            tcomp = 0
                            t_stage = max(tcomp, tcomm)  # 当前阶段时间
                            t_infer = tcomp + tcomm  # 单个输入阶段推理时间
                        if max(self.stage_time_mat[k, s - 1], t_stage) < self.stage_time_mat[j, s]:  # 暂时没考虑不同k值吞吐量相等=的情况（约束中相同优先取单个推理时间小值）
                            self.stage_time_mat[j, s] = max(self.stage_time_mat[k, s - 1], t_stage)
                            self.infer_time_mat[j, s] = self.infer_time_mat[k, s - 1] + t_infer
                            self.split_idx_mat[j, s] = k + 1  # 层数从零开始要+1

        # 打印每个阶段上执行的起始层和终止层
        ls = [0] * self.num_stages  # 每一个阶段上的起始DNN层
        le = [0] * self.num_stages  # 每一个阶段上的最后DNN层
        kt = 0  # 上一阶段最后一层的索引，要-1
        for i in range(self.num_stages - 1):  # 从最后一个阶段前溯,一直到第二个阶段
            if i == 0:  # 最后一个设备
                if self.split_idx_mat[L - 1, self.num_stages - i - 1] < L:
                    ls[self.num_stages - i - 1] = self.split_idx_mat[L - 1, self.num_stages - i - 1] + 1  # +1因为hs为上一阶段最后一层
                    le[self.num_stages - i - 1] = L
                else:
                    ls[self.num_stages - i - 1], le[self.num_stages - i - 1] = 0, 0  # 为0代表只作为中继节点
                kt = self.split_idx_mat[L - 1, self.num_stages - i - 1] - 1
            else:
                if self.split_idx_mat[int(kt), self.num_stages - i - 1] < kt + 1:  # 索引Index不能用浮点数,应该改为整型
                    ls[self.num_stages - i - 1] = self.split_idx_mat[int(kt), self.num_stages - i - 1] + 1
                    le[self.num_stages - i - 1] = kt + 1
                else:
                    ls[self.num_stages - i - 1], le[self.num_stages - i - 1] = 0, 0
                kt = self.split_idx_mat[int(kt), self.num_stages - i - 1] - 1
        ls[0], le[0] = 1, kt + 1  # 第一个设备起始层和终止层
        ls = list(map(int, ls))  # 将浮点数转换为整数并返回列表
        le = list(map(int, le))
        # print("每个阶段的起始层:", ls)
        # print("每个阶段的终止层:", le)
        print("Throughput:", 1 / self.stage_time_mat[L - 1, self.num_stages - 1])
        print("Latency (s):", self.infer_time_mat[L - 1, self.num_stages - 1])
        return 1 / self.stage_time_mat[L - 1, self.num_stages - 1]

    def compute_throughput(self):
        self.align_stage_order()
        self.row_start = np.zeros((N, L))
        self.row_end = np.zeros((N, L))
        self.stage_time_mat = np.full((L, self.num_stages), np.inf)
        self.split_idx_mat = np.zeros((L, self.num_stages))
        self.infer_time_mat = np.full((L, self.num_stages), np.inf)
        return self.run_dp()


if __name__ == '__main__':
    N = 3
    L = 37   # Vgg19 feature layers
    np.random.seed(1)
    first_band = np.random.randint(1, 50, size=N)
    first_band[np.random.randint(N - 1, size=int(N / 2))] = 0.01
    print("Upload BW:", first_band)
    engine = StagePartition()
    engine.load_config()
    posInt = np.zeros((1, N))
    th1, th2 = 0, float("inf")
    ti1, ti2 = 0, 0
    pipe1 = np.zeros((1, N))
    pipe2 = np.zeros((1, N))
    count = 1
    start = datetime.now()
    for a in range(1, N + 1):
        for b in range(1, N + 1):
            for c in range(1, N + 1):
                for d in range(1, N + 1):
                    for e in range(1, N + 1):
                        for f in range(1, N + 1):
                                print("Trial {}".format(count))
                                count += 1
                                posInt[0, 0], posInt[0, 1], posInt[0, 2] = a, b, c
                                posInt[0, 3], posInt[0, 4], posInt[0, 5] = d, e, f
                                print("Order:", posInt)
                                th = engine.compute_throughput()
                                if th > th1 or (th == th1 and ti1 > engine.infer_time_mat[L - 1, engine.num_stages - 1]):
                                    th1, pipe1 = th, posInt.copy()
                                    ti1 = engine.infer_time_mat[L - 1, engine.num_stages - 1]
                                if th < th2:
                                    th2, pipe2 = th, posInt.copy()
                                    ti2 = engine.infer_time_mat[L - 1, engine.num_stages - 1]
    end = datetime.now()
    print("Enum time: {:.2f}s".format((end - start).total_seconds()))
    print("Best:", pipe1, "Throughput:", th1, "Latency: {:.4f}s".format(ti1))
    print("Worst:", pipe2, "Throughput:", th2, "Latency: {:.4f}s".format(ti2))
    print("Best/Worst: {:.2%} / {:.2%}".format(th1 / th2, ti1 / ti2))

    # duration = 1000         # 持续时间以毫秒为单位，这里是1s
    # freq = 200              # Hz
    # winsound.Beep(freq, duration)
