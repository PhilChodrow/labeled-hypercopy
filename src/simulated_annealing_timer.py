# import sys
# import os

# project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
# sys.path.insert(0, project_root)

from poisson_hypergraph import GH
from NMI_func import NMI
from simulated_annealing import simulated_annealing_full_likelihood, simulated_annealing_approx_likelihood
import xgi
import numpy as np
import matplotlib.pyplot as plt

def generate_graph(true_theta, timesteps):
    true_p, true_q, gamma_nu, gamma_nr, gamma_eu, gamma_er = true_theta
    H = xgi.Hypergraph([[0, 1]])
    H.set_node_attributes({0 : 0, 1 : 1}, name = "label")
    g = GH(H, [0, 1], true_p, true_q)
    g.add_hyperedge(timesteps, gamma_nu, gamma_nr, gamma_eu, gamma_er)
    return g

def generate_graph_26_starting_nodes(true_theta, timesteps):
    true_p, true_q, gamma_nu, gamma_nr, gamma_eu, gamma_er = true_theta
    H = xgi.Hypergraph([[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]])
    H.set_node_attributes({0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0,10:0,11:0,12:0,13:1,14:1,15:1,16:1,17:1,18:1,19:1,20:1,21:1,22:1,23:1,24:1,25:1}, name="label")
    g=GH(H, [0,1], true_p, true_q)
    g.add_hyperedge(timesteps, gamma_nu, gamma_nr, gamma_eu, gamma_er)

    return g

true_p = .8
true_q = .2
gamma_nu, gamma_nr, gamma_eu, gamma_er = .75, .25, 1, 0.25
true_theta = [true_p, true_q, gamma_nu, gamma_nr, gamma_eu, gamma_er]
timesteps = 50
g = generate_graph(true_theta, timesteps)

graph_sizes_to_consider = [20, 100, 200]

approx_times = []
full_times = []

import time
print("test")

for size in graph_sizes_to_consider:
    g = generate_graph(true_theta, size-2)
    start = time.time()

    simulated_annealing_approx_likelihood(g, true_theta)

    approx_times.append(time.time() - start)

    print("done")

    start = time.time()

    simulated_annealing_full_likelihood(g, true_theta)

    full_times.append(time.time() - start)

import csv
with open('simulated_annealing_timing.csv', 'a', newline="") as file:
    writer = csv.writer(file)
    for i in range(len(graph_sizes_to_consider)):
        row = [graph_sizes_to_consider[i], approx_times[i], full_times[i]]
        writer.writerows([row]) 
