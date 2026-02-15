
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
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from scipy.stats import norm
import statistics


class SimulatedAnnealingApprox:
    def __init__(self, g, theta):
        self.g = g
        self.theta = theta

        self.steps_taken = 0

        # random initialization of node labels
        self.labels = list(np.random.choice([0,1], size=len(g.nodes)))

        self.likelihoods_per_step = []
        self.nmis_per_step = []
        self.aris_per_step = []

        self.f_e_pairs = self.initialize_f_e_pairs()
        # self.f_e_pairs = self.initalize_f_e_pairs_adjustable_approx_bound(100)

        # TODO add initial, random likelihood
        self.likelihoods_per_step.append(self.calculate_likelihood_with_f_e_pairs(self.labels))
        self.aris_per_step.append(adjusted_rand_score(self.labels, self.g.get_labels()))

        with open('likelihood_of_good_labels.csv', 'a', newline="") as file:
            writer = csv.writer(file)
            row = [self.aris_per_step[0], self.likelihoods_per_step[0]]
            writer.writerows([row])

        # store best likelihood and labels
        self.max_LL = self.likelihoods_per_step[0]
        self.max_LL_corresponding_ari = self.aris_per_step[0]
        self.max_LL_labels = self.labels
        self.max_LL_step = 0

        self.LL_change_data = []
        self.standard_deviation = None


    def generate_step(self):
        node_to_switch = np.random.choice(range(0,len(self.labels)))

        copied_labels = self.labels.copy()
        copied_labels[node_to_switch] = 1 - copied_labels[node_to_switch]
        return copied_labels

    # DEPRACATED
    # def generate_step_flip_flop(self):
    #     edge_to_switch = np.random.choice(range(1, len(self.g.get_edges())))

    #     copied_nodes = self.labels.copy()
    #     for node_index in self.g.get_edges()[edge_to_switch]:
    #         copied_nodes[node_index] = 1 - copied_nodes[node_index]

    #     return copied_nodes
    
    def calculate_likelihood_with_f_e_pairs(self, new_labels):
        arr = np.zeros(len(self.g.get_edges())-1)

        for i in range(len(self.f_e_pairs)):
            pair = self.f_e_pairs[i]
            arr[pair.e_index-1] += self.f_e_pairs[i].calculate_prob(self.theta, new_labels) * self.f_e_pairs[i].weight
        
        # TODO remove after testing
        epsilon = 10**-40
        # arr[arr == 0] = 1
        return np.sum(np.log(arr + epsilon))
    
    def calculate_likelihood_with_f_e_pairs_greedy(self, changed_label_index, changed_label_value):
        arr = np.zeros(len(self.g.get_edges())-1)

        for i in range(len(self.f_e_pairs)):
            pair = self.f_e_pairs[i]

            arr[pair.e_index-1] += self.f_e_pairs[i].greedy_calculate_prob(changed_label_index, changed_label_value) * self.f_e_pairs[i].weight

        epsilon = 10**-40
        return np.sum(np.log(arr+epsilon))
    
    def calculate_likelihood_with_f_e_pairs_multiple_nodes_greedy(self, changed_label_indexes, changed_label_values):
        arr = np.zeros(len(self.g.get_edges())-1)

        for i in range(len(self.f_e_pairs)):
            pair = self.f_e_pairs[i]
            arr[pair.e_index-1] += self.f_e_pairs[i].greedy_calculate_prob_multiple_nodes(changed_label_indexes, changed_label_values) * self.f_e_pairs[i].weight

        for i in range(len(self.f_e_pairs)):
            pair = self.f_e_pairs[i]
            pair.greedy_calculate_prob_multiple_nodes(changed_label_indexes, [1 if val == 0 else 0 for val in changed_label_values])
        
        # adding epsilon avoid numerical issues
        epsilon = 10**-10
        return np.sum(np.log(arr + epsilon))
    
    def update_f_e_pairs_labels(self, changed_label_index, changed_label_value):
        for f_e_pair in self.f_e_pairs:
            f_e_pair.change_counts_given_label_changed(changed_label_index, changed_label_value)

    def update_f_e_pairs_labels_multiple_nodes(self, changed_label_indexes, changed_label_values):
        for i in range(len(changed_label_indexes)):
            for f_e_pair in self.f_e_pairs:
                f_e_pair.change_counts_given_label_changed(changed_label_indexes[i], changed_label_values[i])

    def step_not_greedy(self):
        new_labels = None

        node_to_switch = np.random.choice(range(0,len(self.labels)))

        new_labels = self.labels.copy()
        new_labels[node_to_switch] = 1 - new_labels[node_to_switch]

        # calculate prob with f_e_pairs
        T0 = len(self.g.nodes) * 20


        new_likelihood = self.calculate_likelihood_with_f_e_pairs(new_labels)

        delta_likelihood = new_likelihood - self.likelihoods_per_step[-1]

        epoch = int(self.steps_taken/T0 * 20)

        bad_accept_prob = 1
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

        elif epoch >= 5:
            sd_away = abs(delta_likelihood / self.standard_deviation)
            bad_accept_prob = norm.pdf(sd_away, loc=0, scale=1-(min(epoch, 20)-5)/15)

        if delta_likelihood > 0:
            self.labels = new_labels
            self.likelihoods_per_step.append(new_likelihood)
            self.nmis_per_step.append(normalized_mutual_info_score(self.g.get_labels(), self.labels))


            self.update_f_e_pairs_labels(node_to_switch, new_labels[node_to_switch])

        elif bad_accept_prob > random.random():
            self.labels = new_labels
            self.likelihoods_per_step.append(new_likelihood)
            self.nmis_per_step.append(normalized_mutual_info_score(self.g.get_labels(), self.labels))

            self.update_f_e_pairs_labels(node_to_switch, new_labels[node_to_switch])
        
        else:
            self.likelihoods_per_step.append(self.likelihoods_per_step[-1])
            self.nmis_per_step.append(self.nmis_per_step[-1])

        self.steps_taken += 1

    def new_step(self):
        T0 = len(self.labels)*50
        node_to_switch = None
        node_to_switch = np.random.choice(range(0,len(self.labels)))
        
        new_labels = self.labels.copy()
        new_labels[node_to_switch] = 1 - new_labels[node_to_switch]

        new_likelihood = (self.calculate_likelihood_with_f_e_pairs_greedy(node_to_switch, new_labels[node_to_switch]))
        delta_likelihood = (math.exp(new_likelihood) - math.exp(self.likelihoods_per_step[-1]))

        print(new_likelihood - self.likelihoods_per_step[-1])
        print(delta_likelihood)

        accept_prob = (delta_likelihood/(self.steps_taken+1))
        print(accept_prob)
        
        if accept_prob > 0 or 1-abs(accept_prob) > random.random():
            self.labels = new_labels
            self.likelihoods_per_step.append(new_likelihood)

            self.aris_per_step.append(adjusted_rand_score(self.g.get_labels(), self.labels))

           
            self.update_f_e_pairs_labels(node_to_switch, new_labels[node_to_switch])
        
        else:
            self.likelihoods_per_step.append(self.likelihoods_per_step[-1])

            self.aris_per_step.append(self.aris_per_step[-1])


        if (self.likelihoods_per_step[-1] > self.max_LL):
            self.max_LL = self.likelihoods_per_step[-1]
            self.max_LL_corresponding_ari = self.aris_per_step[-1]
            self.max_LL_labels = self.labels

        self.steps_taken += 1
        

    def step(self):
        T0 = len(self.g.nodes) * 20

        new_labels = None
        flip_flop = False

        # if not flip flop
        node_to_switch = None
        # if flip flop
        nodes_to_switch = None

        node_to_switch = np.random.choice(range(0,len(self.labels)))
        
        new_labels = self.labels.copy()
        new_labels[node_to_switch] = 1 - new_labels[node_to_switch]
     
        new_likelihood = None
        delta_likelihood = None
        if flip_flop:
            node_values_switch_to = []
            for i in range(len(nodes_to_switch)):
                node_values_switch_to.append(new_labels[nodes_to_switch[i]])
            new_likelihood = self.calculate_likelihood_with_f_e_pairs_multiple_nodes_greedy(nodes_to_switch, node_values_switch_to)
            delta_likelihood = new_likelihood - self.likelihoods_per_step[-1]
        else:
            new_likelihood = self.calculate_likelihood_with_f_e_pairs_greedy(node_to_switch, new_labels[node_to_switch])
            delta_likelihood = new_likelihood - self.likelihoods_per_step[-1]

        epoch = int(self.steps_taken/T0 * 20)

        bad_accept_prob = 1
        
        if not flip_flop:
            self.LL_change_data.append(delta_likelihood)
      
        if epoch == 0:
            pass

        elif epoch > 0 and epoch < 5:
            if self.standard_deviation == None:
                self.standard_deviation = statistics.stdev(self.LL_change_data)
                print(self.standard_deviation)
                # print(self.standard_deviation)
                if math.isnan(self.standard_deviation):
                    print("back sd")
                    self.standard_deviation = 10

            # since bad accept prob, assume delta_likelihood is negative
            sd_away = abs(delta_likelihood / self.standard_deviation)
            bad_accept_prob = norm.pdf(sd_away, loc=0, scale=2)

        elif epoch >= 5:
            sd_away = abs(delta_likelihood / self.standard_deviation)
            bad_accept_prob = norm.pdf(sd_away, loc=0, scale=2-2*(min(epoch, 20))/20)

        if delta_likelihood > 0:
            self.labels = new_labels
            self.likelihoods_per_step.append(new_likelihood)
            self.nmis_per_step.append(normalized_mutual_info_score(self.g.get_labels(), self.labels))
            self.aris_per_step.append(adjusted_rand_score(self.g.get_labels(), self.labels))


            if flip_flop:
                self.update_f_e_pairs_labels_multiple_nodes(nodes_to_switch, node_values_switch_to)
            else:
                self.update_f_e_pairs_labels(node_to_switch, new_labels[node_to_switch])
        
        
        elif bad_accept_prob > random.random():
            self.labels = new_labels
            self.likelihoods_per_step.append(new_likelihood)
            self.nmis_per_step.append(normalized_mutual_info_score(self.g.get_labels(), self.labels))
            self.aris_per_step.append(adjusted_rand_score(self.g.get_labels(), self.labels))

            if flip_flop:
                self.update_f_e_pairs_labels_multiple_nodes(nodes_to_switch, node_values_switch_to)
            else:
                self.update_f_e_pairs_labels(node_to_switch, new_labels[node_to_switch])
        
        else:
            self.likelihoods_per_step.append(self.likelihoods_per_step[-1])
            self.nmis_per_step.append(self.nmis_per_step[-1])
            self.aris_per_step.append(self.aris_per_step[-1])


        if self.likelihoods_per_step[-1] > self.max_LL:
            self.max_LL = self.likelihoods_per_step[-1]
            self.max_LL_corresponding_ari = self.aris_per_step[-1]
            self.max_LL_labels = self.labels
            self.max_LL_step = self.steps_taken

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
                    f_e_pairs.append(FEPair(self.g, e_index, f_index, 1/e_index, self.labels, self.theta))

        return f_e_pairs

    def initalize_f_e_pairs_adjustable_approx_bound(self, bound):
        f_e_pairs = []
        for e_index in range(1, len(self.g.get_edges())):
            e = self.g.get_edges()[e_index]
            ks = []
            canidate_f_indexes = []
            for f_index in range(e_index):
                f = self.g.get_edges()[f_index]
                inter_size = len(e.intersection(f))

                if (inter_size != 0):
                    canidate_f_indexes.append((f_index, inter_size))
            
            canidate_f_indexes = sorted(canidate_f_indexes, key=lambda x: x[1])
            for f_pair in canidate_f_indexes[-min(bound, len(canidate_f_indexes)):]:
                f_e_pairs.append(FEPair(self.g, e_index, f_pair[0], 1/len(canidate_f_indexes), self.labels, self.theta))

        return f_e_pairs
        
