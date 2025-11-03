from src.poisson_hypergraph import GH
from src.NMI_func import NMI
import xgi
import numpy as np
import networkx as nx
import sys
import csv

# function that generates an intstance of the class GH with the parameters stored in true_theta and for timesteps times resulting in a graph with timesteps+1 edges
# the graph starts with 2 nodes but will have more as novel nodes are added
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

true_p = .9
true_q = .1
# gamma_nu is poisson weight for the distribution for number of like labeled novel nodes of choosen u novel nodes added
# gamma_nr is poisson weight for the distribution for number of opposite labeled nodes of choosen u novel nodes added
# gamma_eu is poisson weight for the distribution for number of like labeled nodes of choosen u external nodes added
# gamma_er is poisson weight for the distribution for number of opposite labeled nodes of choosen u external nodes added
gamma_nu, gamma_nr, gamma_eu, gamma_er = .01, .01, 1, 0.25

# should be in this order
true_theta = [true_p, true_q, gamma_nu, gamma_nr, gamma_eu, gamma_er]
timesteps = 10

g = generate_graph(true_theta, 100)

# input is an instance of the class GH defined in poisson_hypergraph.py... output a tuple of lists of total likelihood and nmi indexed by timestep
def greedy_community_detection_algo(g, generate_likelihoods):
    total_likelihoods = []
    nmis = []

    null_labels = np.random.choice([0,1], size=len(g.get_labels()))
    true_labels = g.get_labels()

    greedy_steps = 1000
    step_num = 0
    while step_num < greedy_steps:
        delta_E = 0
        e_index = np.random.choice(range(1, len(g.get_edges())))
        
        v_index = np.random.choice(list(g.get_edges()[e_index]))

        canidate_f = []
        for f_index in range(e_index):
            if len(g.get_edges()[f_index].intersection(g.get_edges()[e_index])) != 0:
                canidate_f.append(f_index)


        for f_index in canidate_f:
            delta_E += g.greedy_expectation_step_given_f(v_index, f_index, e_index, true_theta, null_labels) / len(canidate_f)
        
        if (delta_E > 0):
            null_labels[v_index] = 1 - null_labels[v_index]
            # print("swapped label of " + str(v_index) + " from edge " + str(e_index))

        step_num+=1

        # if (step_num % 25 == 0):
        if generate_likelihoods:
            total_likelihoods.append(g.total_log_likelihood(true_theta, null_labels))
            nmis.append(NMI(g.get_labels(), null_labels, g))
        
    # print("greedy labels likelihood: " + str(g.expected_log_likelihood_total(true_theta, null_labels)))
    # print("true labels likelihood: " + str(g.expected_log_likelihood_total(true_theta,true_labels)))
    # print(null_labels)
    # print(true_labels)
    if generate_likelihoods:
        return total_likelihoods, nmis
    else:
        return null_labels
    

LLs = greedy_community_detection_algo(g, False)

csv_name = sys.argv[1]
with open('test_slurm.csv', 'a', newline="") as file:
    writer = csv.writer(file)
    writer.writerows([LLs])

print(LLs)
