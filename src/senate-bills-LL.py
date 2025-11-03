import xgi
import src.poisson_hypergraph as poisson_hypergraph
from src.NMI_func import NMI
import numpy as np
import networkx as nx
import sys
import csv

H = xgi.load_xgi_data("senate-bills")

# Violet's Code Congress Sem
party_affs = H.nodes.attrs('affiliation').asdict()
new_nodes = sorted([int(node) - 1 for node in H.nodes])
new_edges = [{int(node) - 1 for node in edge} for edge in H.edges.members()]

# record dem as 0 and rep as 1 for all nodes
labels = []
for party in list(party_affs.values()):
    if party == 'Democrat':
        labels.append(0)
    if party == 'Republican':
        labels.append(1)

# create new dict using our binary labels
label_dict = dict(zip(new_nodes, labels))
sorted_label_dict = dict(sorted(label_dict.items()))

# make new hypergraph
new_H = xgi.Hypergraph(new_edges)
new_H.set_node_attributes(sorted_label_dict, name = "label")

# turn the data set into an object of the GH class (so we can perform SEM on it)
g = poisson_hypergraph.GH(new_H, [0, 1], 0, 0)

true_thetas = [[.9, .1, .001, .001, 1, .25],
[.8, .2, .001, .001, 1, .25],
[.7, .3, .001, .001, 1, .25],
[.6, .4, .001, .001, 1, .25],
[.43, .37, 0.001, .001, .91, .65]]

theta_index = int(sys.argv[1])%5

true_theta = true_thetas[theta_index]


print("Senate bills LL " + str(g.total_log_likelihood(true_theta, g.get_labels())))
