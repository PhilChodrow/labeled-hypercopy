import sys 
sys.path.append("src")
sys.path.append("scripts")
import poisson_hypergraph as ph 
from linear_map import matrix_of_linear_map

import xgi 
from collections import Counter
import numpy as np
from matplotlib import pyplot as plt
plt.style.use('seaborn-v0_8-whitegrid')


theta = [0.7, 0.3, 0.5, 0.3, 0.2, 0.1]

H = xgi.Hypergraph([[0, 1]])
H.set_node_attributes({0 : 0, 1 : 1}, name = "label")
g = ph.GH(H, [0, 1], theta[0], theta[1])

# compute the slope of the degree distribution from the matrix of the linear map

k_max = 10
M = matrix_of_linear_map(k_max, theta)
vals, E = np.linalg.eig(M)
idx = vals.argsort()[::-1]
v = E[:, idx[0]].real
v = v / v.sum()
P = v.reshape((k_max + 1, k_max + 1))


# compute some moments of the 

K_0 = np.arange(k_max + 1)[:, None] 
K_1 = np.arange(k_max + 1)[None, :]
K = K_0 + K_1

rho_0 = np.nansum((K_0**2/K)*P)
rho_1 = np.nansum((K_1**2/K)*P)
rho_01 = np.nansum((K_0*K_1/K)*P)

mean_d = np.nansum(K*P)

a = (1 + theta[0]*(rho_0 + rho_1 - 1) + theta[1]*rho_01)/mean_d

# need to review math and re-check this, probably not right as is
zeta = 1 + (theta[4] + theta[5]) / a


def degree_sequence(g):
    edge_list = g.H.edges.members()
    deg_seq = Counter()
    for edge in edge_list:
        for node in edge:
            deg_seq[node] += 1
    return np.array(list(deg_seq.values()))

def log_binned_histogram(degree_sequence, interval = 5, num_bins = 20):
    hist, bins = np.histogram(degree_sequence, bins = min(int(len(degree_sequence)/interval), num_bins))
    bins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
    hist, bins = np.histogram(degree_sequence, bins = bins)
    binwidths = bins[1:] - bins[:-1]
    hist = hist / binwidths
    p = hist/hist.sum()

    return bins[:-1], p

for i in range(10000): 
    g.add_hyperedge(2, theta[2], theta[3], theta[4], theta[5])
    # print(i)
    
d = degree_sequence(g)

fig, ax = plt.subplots(figsize = (6,5))

centers, heights = log_binned_histogram(d, num_bins = 30)

dmax = d.max()
dmin = d.min()

upper = dmax 
lower = dmax / 5

offset = (0.2, .05)

x = np.linspace(max(100, lower), upper, 10)*offset[0]


y = (x ** (-zeta)) 

y = offset[1]*y / y.sum()

ax.plot(x, y, linestyle = "--", color = "black", zorder = -10, linewidth=1)
ax.annotate(fr"$\zeta=${zeta:.2f}", xy = (x[0]*2, y[0]/2))


ax.scatter(centers, heights, facecolors='none', edgecolors =  'cornflowerblue', linewidth = 2)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Degree")
ax.set_ylabel("Probability")


title = fr"""
Degree distribution
$\eta_+ = {theta[0]}, \eta_- = {theta[1]}$
$\lambda_+ = {theta[2]}, \lambda_- = {theta[3]}$
$\gamma_+ = {theta[4]}, \gamma_- = {theta[5]}$
"""

ax.set_title(title, fontsize = 10)


plt.tight_layout()

plt.savefig("fig/degree_distribution.png")
