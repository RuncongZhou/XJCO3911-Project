V1.0
带宽temp = np.random.randint(1, 50, size=int(N * (N - 1) / 2)) np.random.seed(seed)
性能self.fn = np.linspace(20, 100, num=N)

V2.0
将第二版本的np.seed(1) 带宽改为1-10 11-20  21-30  31-40 41-50这种情况np.random.randint(1,11)
设备性能 1-20 21-40 41-60 61- 80 81-100 np.linspace(1,20）
但不是所有情况都要跑完，实验量太大，针对不同的带宽情况，只取固定的一种设备性能区间，反之固定的带宽情况下取不同的带宽区间

he=index-1 改为he=index

first_band[np.random.randint(N-1, size=int(N/2))] = 0改为first_band[np.random.randint(N-1, size=int(N/2))] = 0.01

V3.0
np.clip(self.Hs[curr_stage, j - m], a_min=0, a_max=10000)改为self.Hs[curr_stage, j - m]=np.clip(self.Hs[curr_stage, j - m], a_min=0, a_max=10000)

alexnet和YOLONet回溯遍历中第一阶段的j-m写成了j!!!!!
第一阶段回溯代码写错，(self.He[curr_stage, j - m]-1)改为(self.He[curr_stage, j - m + 1]-1)，此外激活层的回溯要用else，否则每次回溯失效（这里也导致了之前的实验效果异常好）
所有实验重新跑

np.random.randint(1, 11,np.random.seed(1)