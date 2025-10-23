import numpy as np
from toy_edge_size_simulator import ToySimulator
import matplotlib.pyplot as plt

THETA = [0.8, 0.2, 0.5, 0.2, 0.4, 0.2]


TS = ToySimulator(edge_list = [[5,5]], theta = THETA, force_label = None)

N_STEPS = int(1e6)
majority_1 = [0]
majority_0 = [0]
balanced   = [1]

for rep in range(N_STEPS):
    TS.simulate(n_samples = 1)
    e = TS.edge_list[-1]
    if e[0] > e[1]:
        majority_0 += [majority_0[-1] + 1]
        majority_1 += [majority_1[-1]]
        balanced   += [balanced[-1]]
    elif e[1] > e[0]:
        majority_1 += [majority_1[-1] + 1]
        majority_0 += [majority_0[-1]]
        balanced   += [balanced[-1]]
    else:
        balanced   += [balanced[-1] + 1]
        majority_0 += [majority_0[-1]]
        majority_1 += [majority_1[-1]]

for i in range(len(balanced)):
    total = majority_0[i] + majority_1[i] + balanced[i]
    majority_0[i] /= total
    majority_1[i] /= total
    balanced[i]   /= total


fig, ax = plt.subplots(1, 1, figsize = (8, 6))
ax.plot(majority_0, label = "Majority 0")
ax.plot(majority_1, label = "Majority 1")
ax.plot(balanced,   label = "Balanced")
ax.semilogx()
ax.set_xlabel("Time step")
ax.set_ylabel("Cumulative count")
ax.set_title("Majority Label Over Time")
ax.legend()
plt.savefig("fig/majority-label-over-time.png", dpi = 300)