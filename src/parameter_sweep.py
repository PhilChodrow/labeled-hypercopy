# Evaluate algorithm performance on different parameters



import sys
import xgi
from poisson_hypergraph import GH
from NMI_func import NMI
import numpy as np
import csv
import random
import networkx as nx
from simulated_annealing import simulated_annealing_full_likelihood, simulated_annealing_approx_likelihood, simulated_annealing_likelihood_batch_approx, SimulatedAnnealingApprox
import time
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
# takes in the algorithm and the timesteps to generate the graph from
# if best likelihood true, returns the best label set from entire run... expensive
# lambda and mu have default values...
# all algorithm results are averaged

parameter_index = int(sys.argv[1])
GRID_SIZE = 11
NUM_RUNS_PER_PARAM_SET = 50

def generate_graph_26_starting_nodes(true_theta, timesteps):
    true_p, true_q, gamma_nu, gamma_nr, gamma_eu, gamma_er = true_theta
    H = xgi.Hypergraph([[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]])
    H.set_node_attributes({0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0,10:0,11:0,12:0,13:1,14:1,15:1,16:1,17:1,18:1,19:1,20:1,21:1,22:1,23:1,24:1,25:1}, name="label")
    g=GH(H, [0,1], true_p, true_q)
    g.add_hyperedge(timesteps, gamma_nu, gamma_nr, gamma_eu, gamma_er)

    return g

def greedy_community_detection_algo_3_edge(g, true_theta, best_labels):
    NUM_ES = 5

    null_labels = np.random.choice([0,1], size=len(g.get_labels()))
    true_labels = g.get_labels()

    greedy_steps = 2000
    step_num = 0

    # TODO remove, for testing
    total_likelihoods = []
    nmis = []
    max_likelihood = -np.inf
    max_likelihood_labels = None
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
        if best_labels:
            LL = g.total_log_likelihood(true_theta, null_labels)
            if LL > max_likelihood:
                max_likelihood = LL
                max_likelihood_labels = null_labels

            
        step_num+=1

    # if generate_likelihoods:
    #     return total_likelihoods, nmis
    if best_labels:
        return max_likelihood_labels
    else:
        return null_labels
    

def greedy_community_detection_algo_with_posterior_prob(g, true_theta, best_labels):
    null_labels = np.random.choice([0,1], size=len(g.get_labels()))
    true_labels = g.get_labels()

    greedy_steps = 2000
    step_num = 0

    # TODO remove, for testing
    total_likelihoods = []
    nmis = []
    max_likelihood = -np.inf
    max_likelihood_labels = None
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

        if best_labels:
            LL = g.total_log_likelihood(true_theta, null_labels)
            if LL > max_likelihood:
                max_likelihood = LL
                max_likelihood_labels = null_labels

        step_num+=1

    if best_labels:
        return max_likelihood_labels
    else:
        return null_labels
    
def greedy_posterior_prob_with_e_threshold(g, true_theta, best_labels):
    null_labels = np.random.choice([0,1], size=len(g.get_labels()))
    true_labels = g.get_labels()

    greedy_steps = 2000
    step_num = 0

    # TODO remove, for testing
    total_likelihoods = []
    nmis = []
    max_likelihood = -np.inf
    max_likelihood_labels = None
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

        # 25% cutoff of edges
        PERCENT_EDGES_CONSIDERED = .75
        sorted_f_probs = sorted(f_probs)
        prob_cutoff = f_probs[int(len(canidate_f)*PERCENT_EDGES_CONSIDERED)]

        # sum 25% of edges approach
        # TODO

        for f_index in canidate_f:
            if (f_probs[f_index] > prob_cutoff):
                delta_E += g.greedy_expectation_step_given_f(v_index, f_index, e_index, true_theta, null_labels) * f_probs[f_index]


        
        if (delta_E > 0):
            null_labels[v_index] = 1 - null_labels[v_index]
        
        # if (step_num % 25 == 0):

        if best_labels:
            LL = g.total_log_likelihood(true_theta, null_labels)
            if LL > max_likelihood:
                max_likelihood = LL
                max_likelihood_labels = null_labels

        step_num+=1

    if best_labels:
        return max_likelihood_labels
    else:
        return null_labels
    
def greedy_algo_flip_all_labels(g, true_theta, best_labels):
    STEPS_PER_E = 10
    PROB_FLIP_STEP = .1
    null_labels = np.random.choice([0,1], size=len(g.get_labels()))
    true_labels = g.get_labels()

    greedy_steps = 3000
    step_num = 0

    max_likelihood = -np.inf
    max_likelihood_labels = None
    while step_num < greedy_steps:
        
        e_index = np.random.choice(range(1, len(g.get_edges())))
        f_probs = g.f_prob_array_given_e_array(e_index, true_theta, null_labels)

        for _ in range(STEPS_PER_E):
            new_labels = null_labels.copy()
            delta_E = 0
            if random.random() < PROB_FLIP_STEP:
                for v_index in g.get_edges()[e_index]:
                    new_labels[v_index] = 1 - new_labels[v_index]


                canidate_f = []
                for f_index in range(e_index):
                    if len(g.get_edges()[f_index].intersection(g.get_edges()[e_index])) != 0:
                        canidate_f.append(f_index)

                for f_index in canidate_f:
                    delta_E += g.expected_log_likelihood_given_f(e_index, f_index, true_theta, new_labels) * f_probs[f_index]
                    delta_E -= g.expected_log_likelihood_given_f(e_index, f_index, true_theta, null_labels) * f_probs[f_index]

                
                if (delta_E > 0):
                    null_labels = new_labels
                    
            else:
                v_index = np.random.choice(list(g.get_edges()[e_index]))
                
                new_labels[v_index] = 1 - new_labels[v_index]
                canidate_f = []
                for f_index in range(e_index):
                    if len(g.get_edges()[f_index].intersection(g.get_edges()[e_index])) != 0:
                        canidate_f.append(f_index)

                for f_index in canidate_f:
                    delta_E += g.greedy_expectation_step_given_f(v_index, f_index, e_index, true_theta, null_labels) * f_probs[f_index]


            
                if (delta_E > 0):
                    null_labels = new_labels
          

        if best_labels:
            LL = g.total_log_likelihood(true_theta, null_labels)
            if LL > max_likelihood:
                max_likelihood = LL
                max_likelihood_labels = null_labels

        step_num+=1

    if best_labels:
        return max_likelihood_labels
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

def greedy_algo_flip_all_labels_smart(g, true_theta, best_labels):
    STEPS_PER_E = 1
    ES_PER_E = 5
    PROB_FLIP_STEP = .2
    null_labels = np.random.choice([0,1], size=len(g.get_labels()))
    true_labels = g.get_labels()

    greedy_steps = 2000
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
          
        if best_labels:
            LL = g.total_log_likelihood(true_theta, null_labels)
            if LL > max_likelihood:
                max_likelihood = LL
                max_likelihood_labels = null_labels

        step_num+=1

    if best_labels:
        return max_likelihood_labels
    else:
        return null_labels

# create param list
def parameter_list_init():
    param_list = []
    for a in range((GRID_SIZE)):
        eta_plus = .5 + a/GRID_SIZE/2
        eta_minus = .5 - a/GRID_SIZE/2
        for b in range((GRID_SIZE)):
            lambda_plus = 1 + b/GRID_SIZE
            lambda_minus = 1 - b/GRID_SIZE

            param_list.append([eta_plus, eta_minus, lambda_plus, lambda_minus])

    return param_list

params = parameter_list_init()

def parameter_sweep(algorithm, timesteps, best_likelihood):
    true_theta = [params[parameter_index][0], params[parameter_index][1], .001, .001, params[parameter_index][2], params[parameter_index][3]]
    results_ari = []
    results_LL = []

    for _ in range(NUM_RUNS_PER_PARAM_SET):
        # true_theta_generate = [.9, .1, .001, .001, 1, .25] # for wrong params only
        g = generate_graph_26_starting_nodes(true_theta, timesteps)

        # labels = algorithm(g, true_theta, best_likelihood)
        labels = algorithm(g, true_theta)
        results_ari.append(adjusted_rand_score(labels, g.get_labels()))
        results_LL.append(g.total_log_likelihood(true_theta, labels))

        print(_)

        print(NMI(labels, g.get_labels(), g))

        # labels = greedy_community_detection_algo_with_posterior_prob(g, true_theta, best_likelihood)
        # print("bad: " + str(NMI(labels, g.get_labels(), g)))

    mean_ari = np.mean(results_ari)
    mean_LL = np.mean(results_LL)

    return mean_ari, mean_LL

def parameter_sweep_greedy(timesteps):
    true_theta = [params[parameter_index][0], params[parameter_index][1], .001, .001, params[parameter_index][2], params[parameter_index][3]]
    results_ari = []
    results_LL = []

    for _ in range(NUM_RUNS_PER_PARAM_SET):
        # true_theta_generate = [.9, .1, .001, .001, 1, .25] # for wrong params only
        g = generate_graph_26_starting_nodes(true_theta, timesteps)

        sa = SimulatedAnnealingApprox(g, true_theta)

        for _ in range(len(g.nodes)*50):
            sa.step()

        # labels = algorithm(g, true_theta, best_likelihood)
        results_ari.append(sa.aris_per_step[-1])
        results_LL.append(sa.likelihoods_per_step[-1])

    mean_ari = np.mean(results_ari)
    mean_LL = np.mean(results_LL)

    return mean_ari, mean_LL

print(params[parameter_index])

# used as a baseline to justify the algorithm
def clique_projection_modularity_maximization_algo(g, true_theta):
    simple_graph = xgi.to_bipartite_graph(g.H)

    partition = nx.community.greedy_modularity_communities(simple_graph)
    z = np.array([0 if node in partition[0] else 1 for node in simple_graph.nodes()])
    z = z[0:len(g.get_labels())]

    return z


# nmi, LL = parameter_sweep(greedy_posterior_prob_with_e_threshold, 30, True)
time.sleep(-1*parameter_index)
# ari, LL = parameter_sweep(simulated_annealing_approx_likelihood, 20, True)
ari, LL = parameter_sweep_greedy(20)

# write results to CSV
with open('relaxed_annealing.csv', 'a', newline="") as file:
    writer = csv.writer(file)
    row = [parameter_index, ari, LL]
    row = row + params[parameter_index]
    writer.writerows([row])



