import xgi
# import src.poisson_hypergraph as poisson_hypergraph
# from src.NMI_func import NMI
# from src.simulated_annealing import simulated_annealing_approx_likelihood, simulated_annealing_full_likelihood
import poisson_hypergraph as poisson_hypergraph
from NMI_func import NMI
from simulated_annealing import simulated_annealing_approx_likelihood, simulated_annealing_full_likelihood, SimulatedAnnealingApprox
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



# params, should vary with job number
# theta_index = int(sys.argv[1])%5
theta_index = 0

# [.43, .37, 0.001, .001, .91, .65] Violet's approx
true_thetas = [[.9, .1, .5, .25, 1, .25],
[.8, .2, .5, .25, 1, .25],
[.7, .3, .5, .25, 1, .25],
[.6, .4, .5, .25, 1, .25],
[.43, .37, 0.5, .25, .91, .65]]

true_theta = true_thetas[theta_index]

def greedy_community_detection_algo_with_posterior_prob(g, generate_likelihoods):
    null_labels = np.random.choice([0,1], size=len(g.get_labels()))
    true_labels = g.get_labels()

    greedy_steps = 2000
    step_num = 0

    # TODO remove, for testing
    total_likelihoods = []
    nmis = []
    while step_num < greedy_steps:
        delta_E = 0
        e_index = np.random.choice(range(1, len(g.get_edges())))
        v_index = np.random.choice(list(g.get_edges()[e_index]))

        new_labels = null_labels.copy()
        new_labels[v_index] = 1 - new_labels[v_index]

        f_probs = g.f_prob_array_given_e_array(e_index, true_theta, new_labels)

        canidate_f = []
        for f_index in range(e_index):
            if len(g.get_edges()[f_index].intersection(g.get_edges()[e_index])) != 0:
                canidate_f.append(f_index)

        for f_index in canidate_f:
            delta_E += g.greedy_expectation_step_given_f(v_index, f_index, e_index, true_theta, null_labels) * f_probs[f_index]


        
        if (delta_E > 0):
            null_labels[v_index] = 1 - null_labels[v_index]
        
        # if (step_num % 25 == 0):
        if generate_likelihoods:
            total_likelihoods.append(g.total_log_likelihood(true_theta, null_labels))
            nmis.append(NMI(g.get_labels(), null_labels, g))

            
        step_num+=1

        if step_num%25 == 0:
            print(step_num)

    if generate_likelihoods:
        return total_likelihoods, nmis
    else:
        return null_labels
    
def greedy_community_detection_algo_with_posterior_prob_true_label_warmstart(g, generate_likelihoods):
    null_labels = g.get_labels()
    true_labels = g.get_labels()

    greedy_steps = 10000
    step_num = 0

    # TODO remove, for testing
    total_likelihoods = []
    nmis = []
    while step_num < greedy_steps:
        delta_E = 0
        e_index = np.random.choice(range(1, len(g.get_edges())))
        v_index = np.random.choice(list(g.get_edges()[e_index]))

        new_labels = null_labels.copy()
        new_labels[v_index] = 1 - new_labels[v_index]

        f_probs = g.f_prob_array_given_e_array(e_index, true_theta, new_labels)

        canidate_f = []
        for f_index in range(e_index):
            if len(g.get_edges()[f_index].intersection(g.get_edges()[e_index])) != 0:
                canidate_f.append(f_index)

        for f_index in canidate_f:
            delta_E += g.greedy_expectation_step_given_f(v_index, f_index, e_index, true_theta, null_labels) * f_probs[f_index]


        
        if (delta_E > 0):
            null_labels[v_index] = 1 - null_labels[v_index]
        
        # if (step_num % 25 == 0):
        if generate_likelihoods:
            total_likelihoods.append(g.total_log_likelihood(true_theta, null_labels))
            nmis.append(NMI(g.get_labels(), null_labels, g))

            
        step_num+=1

        if step_num%25 == 0:
            print(step_num)

    if generate_likelihoods:
        return total_likelihoods, nmis
    else:
        return null_labels
    
def greedy_community_detection_algo_3_edge(g, true_theta):
    generate_likelihoods = False
    NUM_ES = 20

    null_labels = np.random.choice([0,1], size=len(g.get_labels()))
    true_labels = g.get_labels()

    greedy_steps = 5000
    step_num = 0

    # TODO remove, for testing
    total_likelihoods = []
    nmis = []
    while step_num < greedy_steps:
        delta_E = 0
        
        v_index = np.random.choice(list(range(len(g.get_labels()))))

        canidate_e = []
        for e_index in range(1, len(g.get_edges())):
            if v_index in g.get_edges()[e_index]:
                canidate_e.append(e_index)

        e_indexes = np.random.choice(canidate_e, min(NUM_ES, len(canidate_e)))

        new_labels = null_labels.copy()
        new_labels[v_index] = 1 - new_labels[v_index]

        for e_index in e_indexes:
            f_probs = g.f_prob_array_given_e(e_index, true_theta, new_labels)

            canidate_f = []
            for f_index in range(e_index):
                if len(g.get_edges()[f_index].intersection(g.get_edges()[e_index])) != 0:
                    canidate_f.append(f_index)

            for f_index in canidate_f:
                delta_E += g.greedy_expectation_step_given_f(v_index, f_index, e_index, true_theta, null_labels) * f_probs[f_index]


        if (delta_E > 0):
            null_labels[v_index] = 1 - null_labels[v_index]
        
        # if (step_num % 25 == 0):
        if generate_likelihoods:
            total_likelihoods.append(g.total_log_likelihood(true_theta, null_labels))
            nmis.append(NMI(g.get_labels(), null_labels, g))

            
        step_num+=1
        with open('senate_NMIs_per_step.csv', 'a', newline="") as file:
            writer = csv.writer(file)
            row = [NMI(null_labels, g.get_labels(), g)]
            writer.writerows([row])

    if generate_likelihoods:
        return total_likelihoods, nmis
    else:
        return null_labels


def choose_k_similar_edges(k, g, e_index):
    e = g.get_edges()[e_index]
    n = len(e)

    other_es = []
    for f_index in range(1, e_index):
        f = g.get_edges()[f_index]

        if (len(f.intersection(e)) > n*.75):
            other_es.append(f_index)
    
    return np.random.choice(other_es, min(k, len(other_es)))


import random
def greedy_algo_flip_all_labels_smart(g, true_theta):
    likelihood_each_step = False
    STEPS_PER_E = 1
    ES_PER_E = 5
    PROB_FLIP_STEP = .2
    null_labels = np.random.choice([0,1], size=len(g.get_labels()))
    true_labels = g.get_labels()

    greedy_steps = 5000
    step_num = 0

    LLs = []
    NMIs = []
    flip_steps = []
    max_likelihood = -np.inf
    max_likelihood_labels = None
    while step_num < greedy_steps:
        
        e_index = np.random.choice(range(1, len(g.get_edges())))
        

       
        new_labels = null_labels.copy()
        delta_E = 0

    
        if random.random() < PROB_FLIP_STEP:
            for v_index in g.get_edges()[e_index]:
                new_labels[v_index] = 1 - new_labels[v_index]

            
            
            other_es = choose_k_similar_edges(ES_PER_E, g, e_index)

            tot = 0
            for e_index_consider in other_es:
                f_probs = g.f_prob_array_given_e_array(e_index_consider, true_theta, null_labels)
                canidate_f = []
                for f_index in range(e_index_consider):
                    if len(g.get_edges()[f_index].intersection(g.get_edges()[e_index_consider])) != 0:
                        canidate_f.append(f_index)
                for f_index in canidate_f:
                    temp = g.expected_log_likelihood_given_f(e_index_consider, f_index, true_theta, new_labels) * f_probs[f_index]
                    delta_E += temp
                    tot += temp
                    delta_E -= g.expected_log_likelihood_given_f(e_index_consider, f_index, true_theta, null_labels) * f_probs[f_index]


            f_probs = g.f_prob_array_given_e_array(e_index, true_theta, null_labels)
            canidate_f = []
            for f_index in range(e_index):
                if len(g.get_edges()[f_index].intersection(g.get_edges()[e_index])) != 0:
                    canidate_f.append(f_index)
            for f_index in canidate_f:
                temp = g.expected_log_likelihood_given_f(e_index, f_index, true_theta, new_labels) * f_probs[f_index]
                delta_E += temp
                tot += temp
                delta_E -= g.expected_log_likelihood_given_f(e_index, f_index, true_theta, null_labels) * f_probs[f_index]
                

            if (delta_E > 0):
                null_labels = new_labels
                flip_steps.append(step_num)
        else:
            v_index = np.random.choice(list(g.get_edges()[e_index]))
            
            new_labels[v_index] = 1 - new_labels[v_index]
            canidate_f = []
            for f_index in range(e_index):
                if len(g.get_edges()[f_index].intersection(g.get_edges()[e_index])) != 0:
                    canidate_f.append(f_index)
            
            f_probs = g.f_prob_array_given_e_array(e_index, true_theta, null_labels)

            for f_index in canidate_f:
                delta_E += g.greedy_expectation_step_given_f(v_index, f_index, e_index, true_theta, null_labels) * f_probs[f_index]
            if (delta_E > 0):
                null_labels = new_labels
          
        if likelihood_each_step:
            LLs.append(g.total_log_likelihood(true_theta, null_labels))
            NMIs.append(NMI(true_labels, null_labels, g))

        step_num+=1
        with open('senate_NMIs_per_step_smart.csv', 'a', newline="") as file:
            writer = csv.writer(file)
            row = [NMI(null_labels, g.get_labels(), g)]
            writer.writerows([row])

    if likelihood_each_step:
        return LLs, NMIs, flip_steps
    else:
        return null_labels
    
def clique_projection_modularity_maximization_algo(g):
    simple_graph = xgi.to_bipartite_graph(g.H)

    partition = nx.community.greedy_modularity_communities(simple_graph)
    z = np.array([0 if node in partition[0] else 1 for node in simple_graph.nodes()])
    z = z[0:len(g.get_labels())]

    return z
    


# algo_labels = simulated_annealing_approx_likelihood(g, true_theta)
# mod_max_labels = clique_projection_modularity_maximization_algo(g)
# with open('senate_nmi_results.csv', 'a', newline="") as file:
#     writer = csv.writer(file)
#     writer.writerows([[theta_index, NMI(algo_labels, g.get_labels(), g)]])
# print(NMI(mod_max_labels, g.get_labels(), g))

# print("Likelihood before algo: " + str(g.total_log_likelihood(true_theta, g.get_labels())))
# print(NMI(mod_max_labels, g.get_labels(), g))

sa = SimulatedAnnealingApprox(g, true_theta)

for step_num in range(len(g.nodes)*20):
    print("step num: " + str(step_num))

    sa.step()

    # with open('senate_ari_results_approx_index' + str(theta_index) + '.csv', 'a', newline="") as file:
    with open('best_senate_bills_attempt.csv', 'a', newline="") as file:
        writer = csv.writer(file)
        writer.writerows([[step_num, sa.likelihoods_per_step[-1], sa.aris_per_step[-1]]])



