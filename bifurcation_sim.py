import xgi
import poisson_hypergraph
import matplotlib.pyplot as plt
import numpy as np
import csv

beta_vals = np.arange(0, 1, step = 0.01)
r = 3
data = [["beta", "mean majority size", "mean minority size"]]
for beta in beta_vals:
    # print("beta: " + str(beta))
    majority_list = []
    minority_list = []
    for experiment in range(0, 10):
        # print(" exp: " + str(experiment))
        H = xgi.Hypergraph([[0, 1]])
        H.set_node_attributes({0 : 0, 1 : 1}, name = "label")
        GH = poisson_hypergraph.GH(H, [0, 1], 0.6, 0.2)

        timesteps = 10000
        GH.add_hyperedge(timesteps, r * beta, r * (1 - beta), r * beta, r * (1 - beta))
        
        k_0_list = []
        k_1_list = []
        labels = GH.get_labels()
        edges = GH.get_edges()
        for edge in edges:
            edge_labels = [labels[node] for node in edge]
            k_1_list.append(sum(edge_labels))
            k_0_list.append(len(edge_labels) - sum(edge_labels))
        majority = max(sum(k_0_list)/len(k_0_list), sum(k_1_list)/len(k_1_list))
        minority = min(sum(k_0_list)/len(k_0_list), sum(k_1_list)/len(k_1_list))
        majority_list.append(majority)
        minority_list.append(minority)
    data.append([beta, sum(majority_list)/len(majority_list), sum(minority_list)/len(majority_list)])

with open('simulation_results_10_000_edges_lambda3.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(data)

