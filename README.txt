Version Notes

V1.0
- Bandwidth: temp = np.random.randint(1, 50, size=int(N * (N - 1) / 2)), np.random.seed(seed)
- Performance: self.fn = np.linspace(20, 100, num=N)

V2.0
- Use np.seed(1); bandwidth ranges: 1-10, 11-20, 21-30, 31-40, 41-50 via np.random.randint(1,11) etc.
- Device performance ranges: 1-20, 21-40, 41-60, 61-80, 81-100 via np.linspace(1,20) etc.
- For experiments, fix one range and vary the other to reduce workload.

- he=index-1 changed to he=index
- first_band[np.random.randint(N-1, size=int(N/2))] = 0 changed to 0.01

V3.0
- self.Hs[curr_stage, j - m] = np.clip(self.Hs[curr_stage, j - m], a_min=0, a_max=10000)
- AlexNet and YOLONet: first-stage backtrack uses j-m (was incorrectly j)
- First-stage backtrack: (self.He[curr_stage, j - m]-1) -> (self.He[curr_stage, j - m + 1]-1)
- Activation layer backtrack uses else branch
- Re-run all experiments
