# import sys
# import os

# project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
# sys.path.insert(0, project_root)

import xgi
# from src.poisson_hypergraph import GH
# from src.NMI_func import NMI
from poisson_hypergraph import GH
from NMI_func import NMI
import numpy as np
import csv
import random
import networkx as nx
import math
from sklearn.metrics import normalized_mutual_info_score


def generate_step(labels):
    node_to_switch = np.random.choice(range(0,len(labels)))

    # has to be a better way here
    copied_labels = labels.copy()
    copied_labels[node_to_switch] = 1 - copied_labels[node_to_switch]
    return copied_labels

def simulated_annealing_full_likelihood(g, theta):
    NUM_STEPS = len(g.nodes) * 50
    labels = list(np.random.choice([0,1], size=len(g.nodes)))

    likelihoods_per_step = []
    nmis_per_step = []
    likelihoods_per_step.append((g.total_log_likelihood_complete(theta, labels)))
    nmis_per_step.append(normalized_mutual_info_score(g.get_labels(), labels))
    for step in range(1, NUM_STEPS):
        # if step%25 == 0:
        #     print(step)
        T = 1-(step)/(NUM_STEPS)

        

        new_labels = generate_step(labels)

        new_likelihood = g.total_log_likelihood_complete(theta, new_labels)
        # print(math.exp(-(new_likelihood - likelihoods_per_step[-1])/T))
        # print(-(new_likelihood - likelihoods_per_step[-1])/T)
        # print(math.exp(-(new_likelihood - likelihoods_per_step[-1])/(new_likelihood+likelihoods_per_step[-1])/T))

        prob_bad_acceptance = math.exp(-(-1*new_likelihood+likelihoods_per_step[-1])/T)

        print(prob_bad_acceptance)
        if new_likelihood > likelihoods_per_step[-1]:
            # print("better")
            # print(labels)
            # print(g.get_labels())
            # accept
            labels = new_labels
            likelihoods_per_step.append(new_likelihood)
            nmis_per_step.append(normalized_mutual_info_score(g.get_labels(), labels))
        
        
        elif T**2 > random.random():
            # accept
            # print("worse but temp")
            # print(labels)
            # print(g.get_labels())
            labels = new_labels
            likelihoods_per_step.append(new_likelihood)
            nmis_per_step.append(normalized_mutual_info_score(g.get_labels(), labels))
        
        else:
            # print("worse with temp")
            # print(labels)
            # print(g.get_labels())
            likelihoods_per_step.append(likelihoods_per_step[-1])
            nmis_per_step.append(nmis_per_step[-1])

    return labels
    return nmis_per_step, likelihoods_per_step


def simulated_annealing_approx_likelihood(g, theta):
    NUM_STEPS = len(g.nodes) * 50
    labels = list(np.random.choice([0,1], size=len(g.nodes)))

    likelihoods_per_step = []
    nmis_per_step = []
    likelihoods_per_step.append((g.total_log_likelihood_approx(theta, labels)))
    nmis_per_step.append(normalized_mutual_info_score(g.get_labels(), labels))
    for step in range(1, NUM_STEPS):
        # if step%25 == 0:
        #     print(step)
        # T = (1-(step)/(NUM_STEPS))*1000
        T0 = NUM_STEPS
        # cooling_factor = .995
        cooling_factor = (1 - (1/NUM_STEPS)*10)
        T = T0 * cooling_factor**step

        new_labels = generate_step(labels)

        new_likelihood = g.total_log_likelihood_approx(theta, new_labels)


        # print(math.exp(-(new_likelihood - likelihoods_per_step[-1])/T))
        # print(-(new_likelihood - likelihoods_per_step[-1])/T)
        # print(math.exp(-(new_likelihood - likelihoods_per_step[-1])/(new_likelihood+likelihoods_per_step[-1])/T))
        # delete later
        if new_likelihood < likelihoods_per_step[-1]:
            print("bad prop accept prob")
            print(math.exp((new_likelihood-likelihoods_per_step[-1])/T))


        # prob_bad_acceptance = math.exp(-(-1*new_likelihood+likelihoods_per_step[-1])/T)

        # print(prob_bad_acceptance)
        if new_likelihood > likelihoods_per_step[-1]:
            # print("better")
            # print(labels)
            # print(g.get_labels())
            # accept
            labels = new_labels
            likelihoods_per_step.append(new_likelihood)
            nmis_per_step.append(normalized_mutual_info_score(g.get_labels(), labels))
        
        
        elif math.exp((new_likelihood-likelihoods_per_step[-1])/T) > random.random():
            # print(math.exp(-(-1*new_likelihood+likelihoods_per_step[-1])/T))
            # accept
            # print("worse but temp")
            # print(labels)
            # print(g.get_labels())
            labels = new_labels
            likelihoods_per_step.append(new_likelihood)
            nmis_per_step.append(normalized_mutual_info_score(g.get_labels(), labels))
        
        else:
            # print("worse with temp")
            # print(labels)
            # print(g.get_labels())
            likelihoods_per_step.append(likelihoods_per_step[-1])
            nmis_per_step.append(nmis_per_step[-1])


        # with open('senate_bills_results.csv', 'a', newline="") as file:
        #     writer = csv.writer(file)
        #     row = [step, nmis_per_step[-1], likelihoods_per_step[-1]]
        #     writer.writerows([row]) 

        # print("step: " + str(step))
        # print("likelihood: " + str(likelihoods_per_step[-1]))
        # print("nmi: " + str(nmis_per_step[-1]))
        # print("")

    return labels
    return nmis_per_step, likelihoods_per_step

def simulated_annealing_likelihood_batch_approx(g, theta):
    NUM_STEPS = len(g.nodes) * 50
    labels = list(np.random.choice([0,1], size=len(g.nodes)))

    likelihoods_per_step = []
    nmis_per_step = []
    likelihoods_per_step.append((g.total_log_likelihood_approx(theta, labels)))
    nmis_per_step.append(normalized_mutual_info_score(g.get_labels(), labels))
    for step in range(1, NUM_STEPS):
        # if step%25 == 0:
        #     print(step)
        # T = (1-(step)/(NUM_STEPS))*1000
        T0 = NUM_STEPS
        # cooling_factor = .995
        cooling_factor = (1 - (1/NUM_STEPS)*10)
        T = T0 * cooling_factor**step

        new_labels = generate_step(labels)

        new_likelihood = g.total_log_likelihood_approx_batch(theta, new_labels)


        # print(math.exp(-(new_likelihood - likelihoods_per_step[-1])/T))
        # print(-(new_likelihood - likelihoods_per_step[-1])/T)
        # print(math.exp(-(new_likelihood - likelihoods_per_step[-1])/(new_likelihood+likelihoods_per_step[-1])/T))
        # delete later
        if new_likelihood < likelihoods_per_step[-1]:
            print("bad prop accept prob")
            print(math.exp((new_likelihood-likelihoods_per_step[-1])/T))


        # prob_bad_acceptance = math.exp(-(-1*new_likelihood+likelihoods_per_step[-1])/T)

        # print(prob_bad_acceptance)
        if new_likelihood > likelihoods_per_step[-1]:
            # print("better")
            # print(labels)
            # print(g.get_labels())
            # accept
            labels = new_labels
            likelihoods_per_step.append(new_likelihood)
            nmis_per_step.append(normalized_mutual_info_score(g.get_labels(), labels))
        
        
        elif math.exp((new_likelihood-likelihoods_per_step[-1])/T) > random.random():
            # print(math.exp(-(-1*new_likelihood+likelihoods_per_step[-1])/T))
            # accept
            # print("worse but temp")
            # print(labels)
            # print(g.get_labels())
            labels = new_labels
            likelihoods_per_step.append(new_likelihood)
            nmis_per_step.append(normalized_mutual_info_score(g.get_labels(), labels))
        
        else:
            # print("worse with temp")
            # print(labels)
            # print(g.get_labels())
            likelihoods_per_step.append(likelihoods_per_step[-1])
            nmis_per_step.append(nmis_per_step[-1])


        # with open('senate_bills_results.csv', 'a', newline="") as file:
        #     writer = csv.writer(file)
        #     row = [step, nmis_per_step[-1], likelihoods_per_step[-1]]
        #     writer.writerows([row]) 

        # print("step: " + str(step))
        # print("likelihood: " + str(likelihoods_per_step[-1]))
        # print("nmi: " + str(nmis_per_step[-1]))
        # print("")

    return labels
    return nmis_per_step, likelihoods_per_step
