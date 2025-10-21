# import sys
# import os

# project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
# sys.path.insert(0, project_root)

import xgi

try:
    from src.poisson_hypergraph import GH
    from src.NMI_func import NMI
    from src.f_e_pair import FEPair
except:
    from poisson_hypergraph import GH
    from NMI_func import NMI
    from f_e_pair import FEPair
import numpy as np
import csv
import random
import networkx as nx
import math
from sklearn.metrics import normalized_mutual_info_score
from scipy.stats import norm
import statistics


class SimulatedAnnealingApprox:
    def __init__(self, g, theta):
        self.g = g
        self.theta = theta

        self.steps_taken = 0
        # steps between flip flop steps
        self.FLIP_FLOP_STEPS = 10 

        # random initialization of node labels
        self.labels = list(np.random.choice([0,1], size=len(g.nodes)))

        self.likelihoods_per_step = []
        self.nmis_per_step = []
        self.f_e_pairs = self.initialize_f_e_pairs()

        # TODO add initial, random likelihood
        self.likelihoods_per_step.append(self.calculate_likelihood_with_f_e_pairs(self.labels))

        self.LL_change_data = []
        self.standard_deviation = None


    def generate_step(self):
        node_to_switch = np.random.choice(range(0,len(self.labels)))

        copied_labels = self.labels.copy()
        copied_labels[node_to_switch] = 1 - copied_labels[node_to_switch]
        return copied_labels

    def generate_step_flip_flop(self):
        edge_to_switch = np.random.choice(range(1, len(self.g.get_edges())))

        copied_nodes = self.labels.copy()
        for node_index in self.g.get_edges()[edge_to_switch]:
            copied_nodes[node_index] = 1 - copied_nodes[node_index]

        return copied_nodes
    
    def calculate_likelihood_with_f_e_pairs(self, new_labels):
        arr = np.zeros(len(self.g.get_edges())-1)

        for i in range(len(self.f_e_pairs)):
            pair = self.f_e_pairs[i]
            arr[pair.e_index-1] += self.f_e_pairs[i].calculate_prob(self.theta, new_labels) * self.f_e_pairs[i].weight
        
        return np.sum(np.log(arr))
    
    def calculate_likelihood_with_f_e_pairs_greedy(self, changed_label_index, changed_label_value):
        arr = np.zeros(len(self.g.get_edges())-1)

        for i in range(len(self.f_e_pairs)):
            pair = self.f_e_pairs[i]

            # print(self.f_e_pairs[i])
            # print(self.f_e_pairs[i].greedy_calculate_prob(changed_label_index, changed_label_value))

            arr[pair.e_index-1] += self.f_e_pairs[i].greedy_calculate_prob(changed_label_index, changed_label_value) * self.f_e_pairs[i].weight
        
        return np.sum(np.log(arr))
    
    def update_f_e_pairs_labels(self, changed_label_index, changed_label_value):
        for f_e_pair in self.f_e_pairs:
            f_e_pair.change_counts_given_label_changed(changed_label_index, changed_label_value)


    def step_not_greedy(self):
        new_labels = None
        if self.steps_taken % self.FLIP_FLOP_STEPS == 0:
            new_labels = self.generate_step_flip_flop()
        
        else:
            new_labels = self.generate_step()


        # TODO: TEMPORARY FOR TESTING
        node_to_switch = np.random.choice(range(0,len(self.labels)))

        new_labels = self.labels.copy()
        new_labels[node_to_switch] = 1 - new_labels[node_to_switch]

        # TODO: END TEMPORARY FOR TESTING 
     
        # calculate prob with f_e_pairs
        T0 = len(self.g.nodes) * 50
        # T = T0-self.steps_taken
        # T = .1-(self.steps_taken/T0)*.1


        new_likelihood = self.calculate_likelihood_with_f_e_pairs(new_labels)

        # greedy testing
        # new_likelihood = self.calculate_likelihood_with_f_e_pairs_greedy(node_to_switch, new_labels[node_to_switch])
        delta_likelihood = new_likelihood - self.likelihoods_per_step[-1]

        epoch = int(self.steps_taken/T0 * 20)

        bad_accept_prob = 1
        
        # print(epoch)
        # print(len(LL_change_data))
        self.LL_change_data.append(delta_likelihood)
      
        if epoch == 0:
            pass


        elif epoch > 0 and epoch < 5:
            if self.standard_deviation == None:
                self.standard_deviation = statistics.stdev(self.LL_change_data)
                # print(self.standard_deviation)
                if math.isnan(self.standard_deviation):
                    self.standard_deviation = 10

            # since bad accept prob, assume delta_likelihood is negative
            sd_away = abs(delta_likelihood / self.standard_deviation)
            bad_accept_prob = norm.pdf(sd_away, loc=0, scale=1)
            # print(bad_accept_prob)

        elif epoch >= 5:
            sd_away = abs(delta_likelihood / self.standard_deviation)
            bad_accept_prob = norm.pdf(sd_away, loc=0, scale=1-(min(epoch, 20)-5)/15)

        # to determine if annealing schedule is good
        # bad_accept_prob = .1

        # print(math.exp(-(new_likelihood - likelihoods_per_step[-1])/T))
        # print(-(new_likelihood - likelihoods_per_step[-1])/T)
        # print(math.exp(-(new_likelihood - likelihoods_per_step[-1])/(new_likelihood+likelihoods_per_step[-1])/T))
        # delete later
        # if new_likelihood < likelihoods_per_step[-1]:
        #     print("bad prop accept prob")
        #     print(math.exp(-1*(new_likelihood-likelihoods_per_step[-1])/(new_likelihood+likelihoods_per_step[-1])/T))


        # prob_bad_acceptance = math.exp(-(-1*new_likelihood+likelihoods_per_step[-1])/T)

        # print(prob_bad_acceptance)
        if delta_likelihood > 0:
            # print("better")
            # print(labels)
            # print(g.get_labels())
            # accept
            self.labels = new_labels
            self.likelihoods_per_step.append(new_likelihood)
            self.nmis_per_step.append(normalized_mutual_info_score(self.g.get_labels(), self.labels))


            self.update_f_e_pairs_labels(node_to_switch, new_labels[node_to_switch])
        
        
        # elif math.exp(-1*(new_likelihood-likelihoods_per_step[-1])/(new_likelihood+likelihoods_per_step[-1])/T) > random.random():
        elif bad_accept_prob > random.random():
            # print(math.exp(-(-1*new_likelihood+likelihoods_per_step[-1])/T))
            # accept
            # print("worse but temp")
            # print(labels)
            # print(g.get_labels())
            self.labels = new_labels
            self.likelihoods_per_step.append(new_likelihood)
            self.nmis_per_step.append(normalized_mutual_info_score(self.g.get_labels(), self.labels))

            self.update_f_e_pairs_labels(node_to_switch, new_labels[node_to_switch])
        
        else:
            # print("worse with temp")
            # print(labels)
            # print(g.get_labels())
            self.likelihoods_per_step.append(self.likelihoods_per_step[-1])
            self.nmis_per_step.append(self.nmis_per_step[-1])


        # with open('senate_bills_results.csv', 'a', newline="") as file:
        #     writer = csv.writer(file)
        #     row = [step, nmis_per_step[-1], likelihoods_per_step[-1]]
        #     writer.writerows([row]) 

        # print("step: " + str(step))
        # print("likelihood: " + str(likelihoods_per_step[-1]))
        # print("nmi: " + str(nmis_per_step[-1]))
        # print("")

        self.steps_taken += 1
        

    def step(self):
        new_labels = None
        if self.steps_taken % self.FLIP_FLOP_STEPS == 0:
            new_labels = self.generate_step_flip_flop()
        
        else:
            new_labels = self.generate_step()


        # TODO: TEMPORARY FOR TESTING
        node_to_switch = np.random.choice(range(0,len(self.labels)))

        new_labels = self.labels.copy()
        new_labels[node_to_switch] = 1 - new_labels[node_to_switch]

        # TODO: END TEMPORARY FOR TESTING 
     
        # calculate prob with f_e_pairs
        T0 = len(self.g.nodes) * 50
        # T = T0-self.steps_taken
        # T = .1-(self.steps_taken/T0)*.1


        # new_likelihood = self.calculate_likelihood_with_f_e_pairs(new_labels)

        # greedy testing
        new_likelihood = self.calculate_likelihood_with_f_e_pairs_greedy(node_to_switch, new_labels[node_to_switch])
        delta_likelihood = new_likelihood - self.likelihoods_per_step[-1]

        epoch = int(self.steps_taken/T0 * 20)

        bad_accept_prob = 1
        
        # print(epoch)
        # print(len(LL_change_data))
        self.LL_change_data.append(delta_likelihood)
      
        if epoch == 0:
            pass


        elif epoch > 0 and epoch < 5:
            if self.standard_deviation == None:
                self.standard_deviation = statistics.stdev(self.LL_change_data)
                # print(self.standard_deviation)
                if math.isnan(self.standard_deviation):
                    self.standard_deviation = 10

            # since bad accept prob, assume delta_likelihood is negative
            sd_away = abs(delta_likelihood / self.standard_deviation)
            bad_accept_prob = norm.pdf(sd_away, loc=0, scale=1)
            # print(bad_accept_prob)

        elif epoch >= 5:
            sd_away = abs(delta_likelihood / self.standard_deviation)
            bad_accept_prob = norm.pdf(sd_away, loc=0, scale=1-(min(epoch, 20)-5)/15)

        # to determine if annealing schedule is good
        # bad_accept_prob = .1

        # print(math.exp(-(new_likelihood - likelihoods_per_step[-1])/T))
        # print(-(new_likelihood - likelihoods_per_step[-1])/T)
        # print(math.exp(-(new_likelihood - likelihoods_per_step[-1])/(new_likelihood+likelihoods_per_step[-1])/T))
        # delete later
        # if new_likelihood < likelihoods_per_step[-1]:
        #     print("bad prop accept prob")
        #     print(math.exp(-1*(new_likelihood-likelihoods_per_step[-1])/(new_likelihood+likelihoods_per_step[-1])/T))


        # prob_bad_acceptance = math.exp(-(-1*new_likelihood+likelihoods_per_step[-1])/T)

        # print(prob_bad_acceptance)
        if delta_likelihood > 0:
            # print("better")
            # print(labels)
            # print(g.get_labels())
            # accept
            self.labels = new_labels
            self.likelihoods_per_step.append(new_likelihood)
            self.nmis_per_step.append(normalized_mutual_info_score(self.g.get_labels(), self.labels))


            self.update_f_e_pairs_labels(node_to_switch, new_labels[node_to_switch])
        
        
        # elif math.exp(-1*(new_likelihood-likelihoods_per_step[-1])/(new_likelihood+likelihoods_per_step[-1])/T) > random.random():
        elif bad_accept_prob > random.random():
            # print(math.exp(-(-1*new_likelihood+likelihoods_per_step[-1])/T))
            # accept
            # print("worse but temp")
            # print(labels)
            # print(g.get_labels())
            self.labels = new_labels
            self.likelihoods_per_step.append(new_likelihood)
            self.nmis_per_step.append(normalized_mutual_info_score(self.g.get_labels(), self.labels))

            self.update_f_e_pairs_labels(node_to_switch, new_labels[node_to_switch])
        
        else:
            # print("worse with temp")
            # print(labels)
            # print(g.get_labels())
            self.likelihoods_per_step.append(self.likelihoods_per_step[-1])
            self.nmis_per_step.append(self.nmis_per_step[-1])


        # with open('senate_bills_results.csv', 'a', newline="") as file:
        #     writer = csv.writer(file)
        #     row = [step, nmis_per_step[-1], likelihoods_per_step[-1]]
        #     writer.writerows([row]) 

        # print("step: " + str(step))
        # print("likelihood: " + str(likelihoods_per_step[-1]))
        # print("nmi: " + str(nmis_per_step[-1]))
        # print("")

        self.steps_taken += 1

    def initialize_f_e_pairs(self):
        f_e_pairs = []
        for e_index in range(1, len(self.g.get_edges())):
            e = self.g.get_edges()[e_index]
            k = 0
            canidate_f_indexes = []
            for f_index in range(e_index):
                f = self.g.get_edges()[f_index]
                inter_size = len(e.intersection(f))

                if inter_size == k:
                    canidate_f_indexes.append(f_index)
                elif inter_size > k:
                    k = inter_size
                    canidate_f_indexes = [f_index]
            if k > 0:
                for f_index in canidate_f_indexes:
                    f_e_pairs.append(FEPair(self.g, e_index, f_index, 1/len(canidate_f_indexes), self.labels, self.theta))

        return f_e_pairs
        



def generate_step(labels):
    node_to_switch = np.random.choice(range(0,len(labels)))

    # has to be a better way here
    copied_labels = labels.copy()
    copied_labels[node_to_switch] = 1 - copied_labels[node_to_switch]
    return copied_labels

def generate_step_flip_flop(labels, g):
    edge_to_switch = np.random.choice(range(1, len(g.get_edges())))

    copied_nodes = labels.copy()
    for node_index in g.get_edges()[edge_to_switch]:
        copied_nodes[node_index] = 1 - copied_nodes[node_index]

    return copied_nodes

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

        

        new_labels = generate_step_flip_flop(labels, g)

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

    LL_change_data = []
    standard_deviation = None
    for step in range(1, NUM_STEPS):
     
        # T = (1-(step)/(NUM_STEPS))*1000
        T0 = NUM_STEPS
        T = T0-step
        T = .1-(step/NUM_STEPS)*.1
        # cooling_factor = .995
        # cooling_factor = (1 - (1/NUM_STEPS)*10)
        # T = T0 * cooling_factor**step

        # try new move
        new_labels = generate_step(labels)

        if step % 10 == 0:
            new_labels = generate_step_flip_flop(labels, g)
        else:
            new_labels = generate_step(labels)
        new_likelihood = g.total_log_likelihood_approx(theta, new_labels)
        delta_likelihood = new_likelihood - likelihoods_per_step[-1]

        epoch = int(step/NUM_STEPS * 20)

        bad_accept_prob = 1
        

        
        # print(epoch)
        # print(len(LL_change_data))
        LL_change_data.append(delta_likelihood)
      
        if epoch == 0:
            pass


        elif epoch > 0 and epoch < 5:
            if standard_deviation == None:
                standard_deviation = statistics.stdev(LL_change_data)
                # print(standard_deviation)
                if math.isnan(standard_deviation):
                    standard_deviation = 10

            # since bad accept prob, assume delta_likelihood is negative
            sd_away = abs(delta_likelihood / standard_deviation)
            bad_accept_prob = norm.pdf(sd_away, loc=0, scale=1)
            # print(bad_accept_prob)

        elif epoch >= 5:
            sd_away = abs(delta_likelihood / standard_deviation)
            bad_accept_prob = norm.pdf(sd_away, loc=0, scale=1-(epoch-5)/15)

        # # to determine if annealing schedule is good
        # bad_accept_prob = .1

        # print(math.exp(-(new_likelihood - likelihoods_per_step[-1])/T))
        # print(-(new_likelihood - likelihoods_per_step[-1])/T)
        # print(math.exp(-(new_likelihood - likelihoods_per_step[-1])/(new_likelihood+likelihoods_per_step[-1])/T))
        # delete later
        # if new_likelihood < likelihoods_per_step[-1]:
        #     print("bad prop accept prob")
        #     print(math.exp(-1*(new_likelihood-likelihoods_per_step[-1])/(new_likelihood+likelihoods_per_step[-1])/T))


        # prob_bad_acceptance = math.exp(-(-1*new_likelihood+likelihoods_per_step[-1])/T)

        # print(prob_bad_acceptance)
        if delta_likelihood > 0:
            # print("better")
            # print(labels)
            # print(g.get_labels())
            # accept
            labels = new_labels
            likelihoods_per_step.append(new_likelihood)
            nmis_per_step.append(normalized_mutual_info_score(g.get_labels(), labels))
        
        
        # elif math.exp(-1*(new_likelihood-likelihoods_per_step[-1])/(new_likelihood+likelihoods_per_step[-1])/T) > random.random():
        elif bad_accept_prob > random.random():
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

        T = T0 - step
        # cooling_factor = .995
        # cooling_factor = (1 - (1/NUM_STEPS)*10)
        # T = T0 * cooling_factor**step

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
