# PSO + DP stage partition (YOLONet, kernel/stride/padding)
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D


class PSOAgent:
    def __init__(self, cl, dl, height, tlayer, band, fn, x_max, max_vel, dim):
        self.flops_per_layer = cl                                           # 每一层所需要浮点运算次数
        self.data_per_layer = dl                                           # 每一层输入数据大小
        self.layer_height = height                                   # 每一层的层高度
        self.layer_type = tlayer                                   # 每一层的层类型
        self.bw_matrix = band                                       # 带宽矩阵
        self.device_perf = fn                                           # 等差数列,设备性能按由小到大依次排序，索引对应设备名
        self.pos = np.random.uniform(1, x_max, (1, dim))       # 粒子的当前位置,管道的阶段数最大为设备数
        self.posInt = np.around(self.pos)                      # 取整后的粒子位置
        self.vel = np.random.uniform(-max_vel, max_vel, (1, dim))  # 粒子的速度
        self.bestPos = self.pos.copy()                             # 粒子最好的位置
        self.dev_stage_map = np.zeros((1, dim))                       # 索引对应设备号，值对应设备在管道所处的阶段
        self.num_stages = 1                                             # 当前粒子对应管道阶段数，初始为1
        self.Ls = []                                               # 每一个设备的起始层,索引对应设备号
        self.Le = []                                               # 每一个设备的终止层
        self.Iv = np.zeros((N, N))                                 # 任务决策变量1：设备顺序关系
        self.Hv = np.zeros((N, L))                                 # 任务决策变量2：DNN层分配关系,Hv=He-Hs
        self.row_start = np.zeros((N, L))                                 # 设备负责每一层输入特征起始行数，从上往下
        self.row_end = np.zeros((N, L))                                 # 设备负责每一层输入特征的终止行数
        self.stage_time_mat = np.zeros(1)                              # 动态规划所用矩阵,h(j,i)代表j层DNN在前i个阶段上执行的最慢阶段时间
        self.split_idx_mat = np.zeros(1)                             # 动态规划所用矩阵hs[j,i]最优解对应的前i个阶段上执行的层数
        self.infer_time_mat = np.zeros(1)                             # 动态规划所用矩阵,Ti(j,i)代表j层DNN在前i个阶段上执行的最短推理时间
        self.fitnessValue = 0                             # 最优适应度函数值，个体极值
        self.bestTime = 0                                 # 最优适应度所对应推理时间，相同取推理时间最小值

    # 将小数解四舍五入取整，并将设备与管道阶段对应
    def align_stage_order(self):
        self.num_stages = 1
        self.posInt = np.around(self.pos)
        print("Agent position:", self.posInt)
        # print("posInt", self.posInt, self.posInt.shape)
        idx = np.argsort(self.posInt)  # 值越小的设备属于阶段越靠前
        # print("idx:", idx, idx.shape)
        if not type(idx) is np.ndarray:
            idx = np.array(idx)
        temp = self.posInt[0, idx[0, 0]]  # 当前阶段对应的值
        self.dev_stage_map[0, idx[0, 0]] = self.num_stages
        for i in range(1, idx.size):
            if self.posInt[0, idx[0, i]] == temp:
                self.dev_stage_map[0, idx[0, i]] = self.num_stages
            else:  # 位置不相等则在管道下一阶段,stage先自增再赋值
                temp = self.posInt[0, idx[0, i]]
                self.num_stages += 1
                self.dev_stage_map[0, idx[0, i]] = self.num_stages

    def set_pos(self, value):
        self.pos = value

    def get_pos(self):
        return self.pos

    def set_best_pos(self, value):
        self.bestPos = value

    def get_best_pos(self):
        return self.bestPos

    def set_vel(self, value):
        self.vel = value

    def get_vel(self):
        return self.vel

    def set_fitness_value(self, value):
        self.fitnessValue = value

    def get_fitness_value(self):  # 返回个体极值
        return self.fitnessValue

    def get_infer_time(self):  # 返回单个推理时间
        return self.infer_time_mat[L - 1, self.num_stages - 1]

    def run_dp(self):
        # 计算第一个阶段处理前j层的最大周期，采用了自适应层内拆分
        # print("sortDevice:", self.dev_stage_map, self.dev_stage_map.shape)
        # 找到当前阶段(1)设备的索引号，此处为1，参数为-1转化为一维数组
        curr_stage = np.array(np.where(self.dev_stage_map.reshape(-1) == 1)).reshape(-1)
        # print("curr_stage", curr_stage, curr_stage.shape)
        curr_num = curr_stage.size   # 不加括号
        # print("第一阶段设备数:",curr_num)
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
                # 得出每个设备性能比例，根据比例划分当前阶段最后一层，并回溯至当前阶段第一层
                # 独立划分最后一层的输出特征,并转为输入特征范围，计算负载直接按比例分配
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
                if self.layer_type[j] == 3 or self.layer_type[j] == 2:  # 卷积层和池化层，对于YOLONet
                    self.row_start[curr_stage, j] = self.row_start[curr_stage, j]*solver.stride_sz[j]-solver.pad_sz[j]    # 对于YOLONet
                    self.row_end[curr_stage, j] = (self.row_end[curr_stage, j] - 1)*solver.stride_sz[j]+solver.kernel_sz[j]-solver.pad_sz[j]
                arr_device += (self.row_end[curr_stage, j] - self.row_start[curr_stage, j]) / self.layer_height[j] * self.flops_per_layer[j]
                # 填充数据不需要传输，Hs，He中小于1的赋值1，大于hlayer的赋值为hlayer
                self.row_start[curr_stage, j] = np.clip(self.row_start[curr_stage, j], a_min=0, a_max=10000)
                self.row_end[curr_stage, j] = np.clip(self.row_end[curr_stage, j], a_min=-10000, a_max=self.layer_height[j])
                #print("设备计算量", arr_device, arr_device.shape)

                # 从第j-1层回溯至第1层，得到每一层设备所需输入范围，curr_stage不加索引为了整体对应计算
                for m in range(1, j+1):  # j-m=0->m=j+1
                    if self.layer_type[j - m] == 3 or self.layer_type[j - m] == 2:  # 卷积层和池化层
                        self.row_start[curr_stage, j - m] = self.row_start[curr_stage, j - m + 1]*solver.stride_sz[j - m]-solver.pad_sz[j - m]
                        self.row_end[curr_stage, j - m] = (self.row_end[curr_stage, j - m + 1] - 1)*solver.stride_sz[j - m] + \
                            solver.kernel_sz[j - m]-solver.pad_sz[j - m]
                    else:  # 激活层输入输出不变
                        self.row_start[curr_stage, j - m] = self.row_start[curr_stage, j - m + 1]
                        self.row_end[curr_stage, j - m] = self.row_end[curr_stage, j - m + 1]
                    # 在这里解决之前计算时间的公式错误，每向前回溯一次计算一下每个设备执行当前层分配计算量的累积时间
                    # 每个设备的计算时间都要计算，用一个数组来表示，到每一个阶段第一层后开始计算最慢设备所用时间。
                    # add = ((self.row_end[curr_stage, j - m] - self.row_start[curr_stage, j - m]) / self.layer_height[j - m] * self.flops_per_layer[j - m])
                    # print("增量",add,add.shape)
                    # print("设备计算量",arr_device,arr_device.shape)
                    arr_device += (self.row_end[curr_stage, j - m] - self.row_start[curr_stage, j - m]) / self.layer_height[j - m] * self.flops_per_layer[j - m]
                    self.row_start[curr_stage, j - m] = np.clip(self.row_start[curr_stage, j - m], a_min=0, a_max=10000)
                    self.row_end[curr_stage, j - m] = np.clip(self.row_end[curr_stage, j - m], a_min=-10000,a_max=self.layer_height[j - m])

                # 按照当前阶段第一层每个设备的输入范围得到最大传输时间
                tcomm = 0
                for n in range(curr_num):
                    # print("第{}个设备的带宽:{}".format(curr_stage[n],first_band[curr_stage[n]]))
                    temp = self.data_per_layer[0] * (self.row_end[curr_stage[n], 0] - self.row_start[curr_stage[n], 0]) / \
                           self.layer_height[0] / first_band[curr_stage[n]]
                    if temp > tcomm:
                        tcomm = temp
                # print("最大通信时间",tcomm)
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

        # 动态规划，从第二个阶段一直遍历到最后一个阶段
        for s in range(1, self.num_stages):
            # 计算第一个阶段处理前j层的最大周期，采用了自适应层内拆分
            curr_stage = np.array(np.where(self.dev_stage_map.reshape(-1) == (s+1))).reshape(-1)
            # print("curr_stag:",curr_stage)
            prev_stage = np.array(np.where(self.dev_stage_map.reshape(-1) == s)).reshape(-1)  # 上一阶段设备
            # print("prev_stage",prev_stage)
            curr_num = curr_stage.size
            prev_num = prev_stage.size
            for j in range(L):   # 第一阶段肯定有执行层，不是中继设备，输入数据很小
                if curr_num > 1 and prev_num > 1:  # 两个相连阶段是多设备则直接返回0，跳过当前粒子
                    # print("跳过当前粒子")
                    return 0
                for k in range(j + 1):  # 前s-1个阶段处理前(k+1)层的情况，遍历得到最优k值
                    # 首先通过拆分当前阶段最后一层求出每个设备的max(通信,计算)时间，然后得到最慢设备时间，即当前阶段时间
                    if curr_num == 1:  # 当前阶段只有一个设备，则前一阶段可能为多个设备
                        arr_fn = [self.device_perf[prev_stage[i]] for i in range(prev_num)]  # 上一阶段设备性能
                        sum_fn = sum(arr_fn)
                        tcomm = 0
                        # 按照前一阶段设备的输出范围等比分配传输数据大小,计算通信大概时间
                        for n in range(prev_stage.size):
                            temp = self.data_per_layer[k+1] * arr_fn[n] / sum_fn / self.bw_matrix[prev_stage[n], curr_stage[0]]
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
                    else:  # 当前阶段为多个设备，前一阶段为一个设备
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
                            if self.layer_type[j] == 3 or self.layer_type[j] == 2:  # 卷积层和池化层
                                self.row_start[curr_stage, j] = self.row_start[curr_stage, j] * solver.stride_sz[j] - solver.pad_sz[j]  # 对于YOLONet
                                self.row_end[curr_stage, j] = (self.row_end[curr_stage, j] - 1) * solver.stride_sz[j] + \
                                    solver.kernel_sz[j] - solver.pad_sz[j]
                            arr_device += (self.row_end[curr_stage, j] - self.row_start[curr_stage, j]) / self.layer_height[j] * self.flops_per_layer[j]
                            # 填充数据不需要传输，Hs，He中小于1的赋值1，大于hlayer的赋值为hlayer
                            self.row_start[curr_stage, j] = np.clip(self.row_start[curr_stage, j], a_min=0, a_max=10000)
                            self.row_end[curr_stage, j] = np.clip(self.row_end[curr_stage, j], a_min=-10000,a_max=self.layer_height[j])
                            # 计算DNN执行时间时注意要将cl值从Flops转换为GFlops
                            #print("设备计算量", arr_device, arr_device.shape)
                            # 从第j-1层回溯至第(k+1)层，得到每一层设备所需输入范围
                            for m in range(1, j - k):  # j-m=k+1->m=j-k-1
                                if self.layer_type[j-m] == 3 or self.layer_type[j-m] == 2:  # 卷积层和池化层
                                    self.row_start[curr_stage, j-m] = self.row_start[curr_stage, j-m+1] * solver.stride_sz[j-m] - solver.pad_sz[j-m]  # 对于YOLONet
                                    self.row_end[curr_stage, j-m] = (self.row_end[curr_stage, j-m+1] - 1) * solver.stride_sz[j-m] + \
                                        solver.kernel_sz[j-m] - solver.pad_sz[j-m]
                                else:  # 激活层输入输出不变
                                    self.row_start[curr_stage, j - m] = self.row_start[curr_stage, j - m + 1]
                                    self.row_end[curr_stage, j - m] = self.row_end[curr_stage, j - m + 1]

                                arr_device += (self.row_end[curr_stage, j - m] - self.row_start[curr_stage, j - m]) / self.layer_height[
                                    j - m] * self.flops_per_layer[j - m]
                                #add = ((self.row_end[curr_stage, j - m] - self.row_start[curr_stage, j - m]) / self.layer_height[j - m] * self.flops_per_layer[j - m]).reshape(-1)
                                #print("增量", add, add.shape)
                                #print("设备计算量", arr_device, arr_device.shape)
                                self.row_start[curr_stage, j - m] = np.clip(self.row_start[curr_stage, j - m], a_min=0, a_max=10000)
                                self.row_end[curr_stage, j - m] = np.clip(self.row_end[curr_stage, j - m], a_min=-10000,a_max=self.layer_height[j - m])

                            # 按照当前阶段第一层每个设备的输入范围得到最大传输时间
                            tcomm = 0
                            for n in range(curr_num):
                                temp = self.data_per_layer[k+1] * (self.row_end[curr_stage[n], k + 1] - self.row_start[curr_stage[n], k + 1]) / \
                                       self.layer_height[k + 1] / self.bw_matrix[prev_stage[0], curr_stage[n]]
                                if temp > tcomm:
                                    tcomm = temp
                            # 当前阶段的计算时间，取计算时间最大的设备时间
                            tcomp = 0
                            for n in range(curr_num):
                                temp = arr_device[n] / arr_fn[n] / 1e9
                                if temp > tcomp:
                                    tcomp = temp
                            t_stage = max(tcomp, tcomm)  # 当前阶段时间
                            # if s==1 and j ==L-1:
                            #     print("计算时间：",tcomp)
                            #     print("通信时间：",tcomm)
                            #     print("阶段时间:",t_stage)
                            t_infer = tcomp + tcomm  # 单个输入阶段推理时间
                            # print("self.stage_time_mat[k, s - 1]",self.stage_time_mat[k, s - 1])
                            # print("self.stage_time_mat[j, s]",self.stage_time_mat[j, s])
                        else:  # 当前阶段仅仅充当中继节点
                            # 按照当前阶段第一层每个设备的输入范围得到最大传输时间
                            tcomm = 0
                            for n in range(curr_stage.size):
                                temp = self.data_per_layer[k+1] * (self.row_end[curr_stage[n], k] - self.row_start[curr_stage[n], k]) / \
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
        # ls = [0] * self.num_stages  # 每一个阶段上的起始DNN层
        # le = [0] * self.num_stages  # 每一个阶段上的最后DNN层
        # kt = 0  # 上一阶段最后一层的索引，要-1
        # for i in range(self.num_stages - 1):  # 从最后一个阶段前溯,一直到第二个阶段
        #     if i == 0:  # 最后一个设备
        #         if self.split_idx_mat[L - 1, self.num_stages - i - 1] < L:
        #             ls[self.num_stages - i - 1] = self.split_idx_mat[L - 1, self.num_stages - i - 1] + 1  # +1因为hs为上一阶段最后一层
        #             le[self.num_stages - i - 1] = L
        #         else:
        #             ls[self.num_stages - i - 1], le[self.num_stages - i - 1] = 0, 0  # 为0代表只作为中继节点
        #         kt = self.split_idx_mat[L - 1, self.num_stages - i - 1] - 1
        #     else:
        #         if self.split_idx_mat[int(kt), self.num_stages - i - 1] < kt + 1:  # 索引Index不能用浮点数,应该改为整型
        #             ls[self.num_stages - i - 1] = self.split_idx_mat[int(kt), self.num_stages - i - 1] + 1
        #             le[self.num_stages - i - 1] = kt + 1
        #         else:
        #             ls[self.num_stages - i - 1], le[self.num_stages - i - 1] = 0, 0
        #         kt = self.split_idx_mat[int(kt), self.num_stages - i - 1] - 1
        # ls[0], le[0] = 1, kt + 1  # 第一个设备起始层和终止层
        # ls = list(map(int, ls))  # 将浮点数转换为整数并返回列表
        # le = list(map(int, le))
        # print("每个阶段的起始层:", ls)
        # print("每个阶段的终止层:", le)
        print("吞吐量为:", 1 / self.stage_time_mat[L - 1, self.num_stages - 1])
        # print("单个输入推理时间:",self.infer_time_mat[L - 1, self.num_stages - 1])
        return 1 / self.stage_time_mat[L - 1, self.num_stages - 1]  # 返回吞吐量，时间的倒数

    def evaluate_fitness(self):
        self.align_stage_order()  # 先找到每个设备对应的阶段数
        self.row_start = np.zeros((N, L))  # 设备负责每一层输入特征起始行数，从上往下，计算量跟输入范围正比例
        self.row_end = np.zeros((N, L))  # 设备负责每一层输入特征的终止行数
        self.stage_time_mat = np.full((L, self.num_stages), np.inf)  # 动态规划所用矩阵,h(j,i)代表j层DNN在前i个阶段上执行的最短时间
        self.split_idx_mat = np.zeros((L, self.num_stages))  # 动态规划所用矩阵hs[j,i]最优解对应的前i个阶段上执行的层数
        self.infer_time_mat = np.full((L, self.num_stages), np.inf)  # 动态规划所用矩阵,h(j,i)代表j层DNN在前i个阶段上执行的最短推理时间
        return self.run_dp()


class SwarmDPSolver:  # 粒子群和动态规划融合算法
    def __init__(self, max_vel, tol, C1=2, C2=2, W=1):
        # Particle Swarm Optimization With Placement Sorting
        self.c1 = C1
        self.c2 = C2
        self.w = W
        self.size = size  # 粒子个数
        self.iter_num = iter_num  # 迭代次数
        self.x_max = max_pos  # 粒子最大位置
        self.max_vel = max_vel  # 粒子最大速度
        self.tol = tol  # 截止条件
        self.best_fitness_value = 0  # 最优自适应度值，吞吐量=时间的倒数
        self.best_position = np.zeros((1, N))  # 种群最优位置,小数解
        self.best_intPosition = np.zeros((1, N))  # 种群最优位置,整数解
        self.fitness_val_list = []  # 每次迭代最优适应值
        self.inference_time = 0  # 最优适应值对应推理时间
        # 粒子种群,元素为每个粒子的对象
        self.agent_list = []

        # Self-Adaptive Dynamic Programming Partition
        self.bw_matrix = np.zeros((N, N))  # 生成带宽矩阵
        self.flops_per_layer = np.zeros(0)         # 每一层所需要浮点运算次数cl
        self.data_per_layer = np.zeros(0)         # 每一层输出数据大小dl
        self.layer_height = np.zeros(0)  # 每一层的层高度
        self.layer_type = np.zeros(0)  # 每一层的层类型
        self.device_perf = np.zeros(0)      # 等差数列,设备性能按由小到大依次排序，索引对应设备名
        self.kernel_sz = np.zeros(0)   # 卷积核大小
        self.stride_sz = np.zeros(0)        # 卷积步长
        self.pad_sz = np.zeros(0)      # 填充大小，单边

    def load_config(self):
        # 随机生成n*(n-1)/2=10个1-50中的随机整数作为设备之间的带宽值,a到b和b到a的速度相等(MB/s)
        # temp = np.full(int(n * (n - 1) / 2), 10)  # 带宽都相同情况
        temp = np.random.randint(21, 31, size=int(N * (N - 1) / 2))
        # print(temp, temp.shape, temp.dtype)
        # 初始化设备计算性能和带宽矩阵值(GFlops)
        self.device_perf = np.linspace(81, 100, num=N)
        print("设备性能(GFlops/s):", self.device_perf, self.device_perf.shape, self.device_perf.dtype)
        t1, t2 = 0, 0
        # 将带宽值依次赋予矩阵,其索引为设备名
        for i in range(N - 1):  # 按行给上三角赋值
            for j in range(i + 1, N):
                self.bw_matrix[i, j] = temp[t1]
                t1 += 1
        for j in range(N - 1):  # 按列给下三角赋值
            for i in range(j + 1, N):
                self.bw_matrix[i, j] = temp[t2]
                t2 += 1
        print("设备传输带宽(MB/s):", self.bw_matrix, self.bw_matrix.shape, self.bw_matrix.dtype)

        # 导入层参数(每一层所需要浮点运算次数cl和输出数据大小dl)
        #df = pd.read_csv("data\\YOLONet.csv")
        df = pd.read_csv("data/YOLONet.csv")
        # df.values.tolist()
        self.flops_per_layer = np.array(df["Flops"])  # 获取某一列的值
        # print("DNN层数为:", L)
        print("层浮点运算次数Flops:", self.flops_per_layer, self.flops_per_layer.shape, self.flops_per_layer.dtype)
        self.data_per_layer = np.array(df["DataSize"])  # 这一列数据是自己计算加上去的,当前层计算所需要的输入数据大小
        print("层输入数据大小MB:", self.data_per_layer, self.data_per_layer.shape, self.data_per_layer.dtype)
        self.layer_height = np.array(df["height"])  # 每一层的层高度
        print("层高度:", self.layer_height, self.layer_height.shape, self.layer_height.dtype)
        self.layer_type = np.array(df["type"])  # 每一层的层类型
        print("层类型:", self.layer_type, self.layer_type.shape, self.layer_type.dtype)
        self.kernel_sz = np.array(df["Kernel_size"])
        self.stride_sz = np.array(df["stride"])
        self.pad_sz = np.array(df["padding"])

        # with open('data\\vgg19.csv','r') as csvfile:
        #     reader = csv.DictReader(csvfile)
        #     self.flops_per_layer = [row['Flops'] for row in reader]
        #     self.data_per_layer = [row['DataSize'] for row in reader]
        #     该函数返回的结果遍历一次之后，再次遍历返回的结果是空列表。
        # print(self.flops_per_layer)
        # print(self.data_per_layer)

        # 初始化粒子种群，第一次随机生成位置和速度
        self.agent_list = [PSOAgent(self.flops_per_layer, self.data_per_layer, self.layer_height, self.layer_type, self.bw_matrix, self.device_perf, self.x_max,
                                       self.max_vel, N) for _ in range(self.size)]

    def set_best_fitness_value(self, value):
        self.best_fitness_value = value

    def get_best_fitness_value(self):
        return self.best_fitness_value

    def set_best_position(self, value):
        self.best_position = value

    def get_best_position(self):
        return self.best_position

    def set_best_int_position(self, value):
        self.best_intPosition = value

    def get_best_int_position(self):
        return self.best_intPosition

    # 更新速度,要用小数值
    def update_vel(self, agent):
        vel_value = self.w * agent.get_vel() + self.c1 * np.random.rand() * (agent.get_best_pos() - agent.get_pos()) \
                    + self.c2 * np.random.rand() * (self.get_best_position() - agent.get_pos())
        vel_value[vel_value > self.max_vel] = self.max_vel  # 限制速度的变化范围
        vel_value[vel_value < -self.max_vel] = -self.max_vel
        # print("速度变化值:", vel_value)
        agent.set_vel(vel_value)

    # 更新位置，要用小数值
    def update_pos(self, agent):
        pos_value = agent.get_pos() + agent.get_vel()
        pos_value[pos_value > self.x_max] = self.x_max  # 限制位置的变化范围
        pos_value[pos_value < 1] = 1
        agent.set_pos(pos_value)
        # print("更新后小数位置:", agent.get_pos())
        value = agent.evaluate_fitness()  # 返回的是吞吐量
        # print("更新后整数位置:", np.around(agent.get_pos()))
        if value > agent.get_fitness_value():  # 与局部极值比较并更新
            agent.set_fitness_value(value)
            agent.set_best_pos(pos_value)
            agent.bestTime = agent.get_infer_time()
            print("个体极值更新")
            print("value值为:", value)
            if value > self.get_best_fitness_value():  # 更新后才与全局极值比较
                self.set_best_fitness_value(value)
                self.set_best_position(pos_value)
                self.inference_time = agent.get_infer_time()  # 记录对应单个推理时间
                print("全局极值更新")
        elif value == agent.get_fitness_value() and agent.get_infer_time() < agent.bestTime:
            agent.set_best_pos(pos_value)
            agent.bestTime = agent.get_infer_time()
            print("个体极值更新")
            print("value值为:", value)
            if value > self.get_best_fitness_value() or \
                    (value == self.get_best_fitness_value() and agent.get_infer_time() < self.inference_time):
                self.set_best_fitness_value(value)
                self.set_best_position(pos_value)
                self.inference_time = agent.get_infer_time()  # 记录对应单个推理时间
                print("全局极值更新")
        else:
            print("个体极值不变")

    # 更新位置，要用小数值
    def update_pos_random(self, agent):
        pos_value = np.random.uniform(1, max_pos, (1, N))
        agent.set_pos(pos_value)
        # print("更新后小数位置:", agent.get_pos())
        value = agent.evaluate_fitness()  # 返回的是吞吐量
        # print("更新后整数位置:", np.around(agent.get_pos()))
        if value > agent.get_fitness_value():  # 与局部极值比较并更新
            agent.set_fitness_value(value)
            agent.set_best_pos(pos_value)
            agent.bestTime = agent.get_infer_time()
            print("个体极值更新")
            print("value值为:", value)
            if value > self.get_best_fitness_value():  # 更新后才与全局极值比较
                self.set_best_fitness_value(value)
                self.set_best_position(pos_value)
                self.inference_time = agent.get_infer_time()  # 记录对应单个推理时间
                print("全局极值更新")
        elif value == agent.get_fitness_value() and agent.get_infer_time() < agent.bestTime:
            agent.set_best_pos(pos_value)
            agent.bestTime = agent.get_infer_time()
            print("个体极值更新")
            print("value值为:", value)
            if value > self.get_best_fitness_value() or \
                    (value == self.get_best_fitness_value() and agent.get_infer_time() < self.inference_time):
                self.set_best_fitness_value(value)
                self.set_best_position(pos_value)
                self.inference_time = agent.get_infer_time()  # 记录对应单个推理时间
                print("全局极值更新")
        else:
            print("个体极值不变")

    def update_ndim(self):
        print("初始化...")        # 因为可能第一次初始值中包含最优值
        for agent in self.agent_list:
            agent.fitnessValue = agent.evaluate_fitness()
            agent.bestTime = agent.get_infer_time()   # 初始化个体极值对应推理时间
            if agent.get_fitness_value() > self.best_fitness_value:
                self.best_fitness_value = agent.get_fitness_value()
                self.inference_time = agent.bestTime
                self.best_position = agent.get_pos()
        print('初始种群最佳适应值为{}'.format(self.get_best_fitness_value()))
        self.fitness_val_list.append(self.get_best_fitness_value())  # 初始值

        for i in range(1, self.iter_num + 1):
            print("第{}次迭代:".format(i))
            c = 1
            arr = np.random.randint(2, size=size)
            for agent in self.agent_list:
                print('第{}个粒子计算:'.format(c))
                if arr[c - 1] == 0:
                    self.update_vel(agent)
                    self.update_pos(agent)
                    # print("第{}个粒子的局部极值:{}".format(c, agent.get_fitness_value()))
                else:
                    self.update_pos_random(agent)
                c += 1
            self.fitness_val_list.append(self.get_best_fitness_value())  # 每次迭代完把当前的最优适应度存到列表
            print('第{}次迭代后最佳适应值为{}:'.format(i, self.get_best_fitness_value()))
            print("单个推理时间:", self.inference_time)
            print('第{}次迭代后全局最优位置:'.format(i), np.around(self.get_best_position()))
            # 暂时设置为相邻迭代变化小于阈值，也可以设置为当前迭代自适应度值大于阈值
            # if (self.fitness_val_list[-1]-self.fitness_val_list[-2]) < self.tol:
            #     break
        return self.fitness_val_list, self.get_best_position()

    # 可视化绘图
    def visual(self):
        # 绘制图形(采用指数分布,横坐标为外部迭代次数，纵坐标为）
        # 创建画布
        fig = plt.figure(num=1, figsize=(12, 6))
        # 字体设置为中文黑体
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        plt.xlabel("迭代次数", fontsize=15)
        plt.ylabel("吞吐量", fontsize=15)
        plt.title("PSDPA(粒子数={},迭代次数={}),seed={}".format(size,iter_num,seed), fontsize=25, backgroundcolor='#3c7f99',
                  fontweight='bold', color='white', verticalalignment="baseline")
        # plt.semilogy(np.arange(1, x + 1), y1, "r")
        plt.plot(np.arange(1, self.iter_num + 1), self.fitness_val_list[1:], "r")
        plt.show()


if __name__ == '__main__':
    N = 7            # 粒子的维度=设备数量
    L = 52           # 当前DNN特征提取层的层数(卷积层Conv、池化层Pool、批量归一化层BN、激活层Relu)vgg19模型
    size = 20        # 粒子数量
    iter_num = 20    # 迭代次数
    max_pos = N      # 最大位置范围
    seed=1
    start = datetime.now()
    np.random.seed(seed)
    first_band = np.random.randint(1, 50, size=N)  # 任务产生设备与协作推理设备之间传输带宽
    # print(np.random.randint(N - 1, size=int(N / 2)))   元素可能相同，大于一半的设备能卸载
    first_band[np.random.randint(N - 1, size=int(N / 2))] = 0.01  # 任务产生设备只与部分设备能通信
    print("Upload BW:", first_band)
    solver = SwarmDPSolver(max_pos / N, 1e-4)
    solver.load_config()
    fit_var_list, best_pos = solver.update_ndim()
    print("Best position:", best_pos)
    print("Best throughput:", fit_var_list[-1])
    print("Latency:", solver.inference_time)
    end = datetime.now()
    print("Solver time: {:.2f}s".format((end - start).total_seconds()))
    # solver.visual()
    # duration = 1000                          # 持续时间以毫秒为单位，这里是1秒
    # freq = 200                               # Hz
    # winsound.Beep(freq, duration)
