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
import torch


class GradientDescent:
    def __init__(self, theta, g, learning_rate, momentum):
        self.theta = theta
        self.g = g

        self.label_LLs_converted = []
        self.label_LLs = []
        self.label_aris = []

        # self.labels = np.random.uniform(0.4, 0.6, len(self.g.get_labels()))
        self.labels = np.full(len(g.get_labels()), .5) # set the labels perfecly at half
        self.labels[0] = .8 # break the symmetry
        self.tensor_labels = torch.tensor(self.labels, requires_grad=True)
        self.optimizer = torch.optim.Adam([self.tensor_labels], lr=learning_rate, betas=momentum)

        self.label_aris.append(adjusted_rand_score(self.g.get_labels(), self.convert_final_labels(self.tensor_labels)))
        self.label_LLs.append(self.differentiable_log_likelihood(self.tensor_labels))
        self.label_LLs_converted.append(self.g.total_log_likelihood_complete(self.theta, self.convert_final_labels(self.tensor_labels)))

    def generate_canidate_f_indexes(self, e_index):
        canidate_f = []
        e = self.g.get_edges()[e_index]
        for f_index in range(e_index):
            if len(self.g.get_edges()[f_index].intersection(e)) != 0:
                canidate_f.append(f_index)
            
        return canidate_f
    
    def generalized_comb(self, n, k):
        return math.gamma(n+1) / math.gamma(k+1) / math.gamma(n-k+1)

    def convert_final_labels(self, tensor_labels):
        labels = []
        for label in tensor_labels:
            if label > .5:
                labels.append(1)
            else:
                labels.append(0)

        return labels

    def differentiable_log_likelihood(self, tensor_labels):
        summ = torch.tensor([0.0], requires_grad=True)

        for e_index in range(1, len(self.g.get_edges())):
            canidate_f_indexes = self.generate_canidate_f_indexes(e_index)
            e_sum = torch.tensor([0.0], requires_grad=True)
            for f_index in canidate_f_indexes:
                e = self.g.get_edges()[e_index]
                f = self.g.get_edges()[f_index]

                prev_nodes = list(range(self.g.last_added[e_index - 1] + 1))
                p, q, gamma_nu, gamma_nr, gamma_eu, gamma_er = self.theta

                # novel nodes counting
                novel_nodes = set(e) - set(prev_nodes)
                t = torch.zeros(len(tensor_labels), dtype=torch.float32)
                t[list(novel_nodes)] = 1
                novel_labels = torch.mul(t,tensor_labels)
                # print(novel_labels)
                nov1 = sum(novel_labels)
                nov0 = len(novel_nodes) - nov1

                # external nodes counting
                external_nodes = set(prev_nodes) - set(f)
                t = torch.zeros(len(tensor_labels), dtype=torch.float32)
                t[list(external_nodes)] = 1
                external_node_labels = torch.mul(t,tensor_labels)
                posext1 = sum(external_node_labels)
                posext0 = len(external_nodes) - posext1

                external_nodes_added = external_nodes.intersection(e)
                t = torch.zeros(len(tensor_labels), dtype=torch.float32)
                t[list(external_nodes_added)] = 1
                external_nodes_added_labels = torch.mul(t,tensor_labels)
                ext1 = sum(external_nodes_added_labels)
                ext0 = len(external_nodes_added) - ext1

                # copied and not copied nodes counting
                copied_nodes = e.intersection(f)
                t = torch.zeros(len(tensor_labels), dtype=torch.float32)
                t[list(copied_nodes)] = 1
                copied_nodes_labels = torch.mul(t,tensor_labels)
                cop1 = sum(copied_nodes_labels)
                cop0 = len(copied_nodes) - cop1

                not_copied_nodes = set(f) - set(e)
                t = torch.zeros(len(tensor_labels), dtype=torch.float32)
                t[list(not_copied_nodes)] = 1
                not_copied_nodes_labels = torch.mul(t,tensor_labels)

                notcop1 = sum(not_copied_nodes_labels)
                notcop0 = len(not_copied_nodes) - notcop1

                # find probs of u being label 1 and 0
                t = torch.zeros(len(tensor_labels), dtype=torch.float32)
                f_intersect_e = f.intersection(e)
                t[list(f_intersect_e)] = 1
                f_labels = torch.mul(t,tensor_labels)
                prob_u_label_equals_1 = sum(f_labels) / len(f_intersect_e)
                prob_u_label_equals_0 = 1 - prob_u_label_equals_1

                prob_given_f_u_label_1 = (p**(cop1-1)) * ((1-p)**notcop1) * (q**cop0) * ((1-q)**notcop0)
                prob_given_f_u_label_1 = prob_given_f_u_label_1 * (gamma_eu**ext1) * math.exp(-gamma_eu) / math.gamma(ext1+1) / self.generalized_comb(posext1, ext1)
                prob_given_f_u_label_1 = prob_given_f_u_label_1 * (gamma_er**ext0) * math.exp(-gamma_er) / math.gamma(ext0+1) / self.generalized_comb(posext0, ext0)
                prob_given_f_u_label_1 = prob_given_f_u_label_1 * (gamma_nu**nov1) * math.exp(-gamma_nu) / math.gamma(nov1+1)
                prob_given_f_u_label_1 = prob_given_f_u_label_1 * (gamma_nr**nov0) * math.exp(-gamma_nr) / math.gamma(nov0+1)

                # calculate prob of e given f and u_label = 0
                prob_given_f_u_label_0 = (p**(cop0-1)) * ((1-p)**notcop0) * (q**cop1) * ((1-q)**notcop1)
                prob_given_f_u_label_0 = prob_given_f_u_label_0 * (gamma_eu**ext0) * math.exp(-gamma_eu) / math.gamma(ext0+1) / self.generalized_comb(posext0, ext0)
                prob_given_f_u_label_0 = prob_given_f_u_label_0 * (gamma_er**ext1) * math.exp(-gamma_er) / math.gamma(ext1+1) / self.generalized_comb(posext1, ext1)
                prob_given_f_u_label_0 = prob_given_f_u_label_0 * (gamma_nu**nov0) * math.exp(-gamma_nu) / math.gamma(nov0+1)
                prob_given_f_u_label_0 = prob_given_f_u_label_0 * (gamma_nr**nov1) * math.exp(-gamma_nr) / math.gamma(nov1+1)

                e_sum = e_sum + (prob_u_label_equals_1*prob_given_f_u_label_1 + prob_u_label_equals_0*prob_given_f_u_label_0)/e_index
                

            
            summ = summ + torch.log(e_sum)
        
        REGULARIZATION_CONSTANT = 1*len(self.g.get_edges())/5 # LL goes down based on the number of edges, so this makes sense
        regularization = torch.tensor([0.0], requires_grad=True)
        for i in range(len(tensor_labels)):
            regularization = regularization +  torch.log(tensor_labels[i]) + torch.log(1-tensor_labels[i])

        return -1*summ - regularization * REGULARIZATION_CONSTANT
    
    def run(self, num_steps):
        last_final_labels = self.convert_final_labels(self.tensor_labels)
        equal_count = 0

        for step in range(num_steps):
            if equal_count == 10:
                break
            loss = self.differentiable_log_likelihood(self.tensor_labels)
            if (torch.isnan(loss)):
                print("NAN")
                break
            print(loss)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            print(self.tensor_labels)
            final_labels = self.convert_final_labels(self.tensor_labels)
            print(final_labels)
            new_loss = self.g.total_log_likelihood_complete(self.theta, final_labels)
            new_ari = adjusted_rand_score(self.g.get_labels(), final_labels)
            print("New label likelihood: " + str(new_loss))
            print("New label ari: " + str(new_ari))

            if last_final_labels == final_labels:
                equal_count += 1
            else:
                last_final_labels = final_labels
                equal_count = 0

            self.label_aris.append(new_ari)
            self.label_LLs.append(loss)
            self.label_LLs_converted.append(new_loss)

        print()

        # true label likelihood
        print("True label likelihood: " + str(self.g.total_log_likelihood_complete(self.theta, self.g.get_labels())))
        print("New label likelihood final: " + str(self.label_LLs_converted[-1]))

true_p = .9
true_q = .1
gamma_nu, gamma_nr, gamma_eu, gamma_er = .3, .1, 1, 0.25
timesteps = 100
true_theta = [true_p, true_q, gamma_nu, gamma_nr, gamma_eu, gamma_er]

def generate_graph_26_starting_nodes(true_theta, timesteps):
    true_p, true_q, gamma_nu, gamma_nr, gamma_eu, gamma_er = true_theta
    H = xgi.Hypergraph([[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]])
    H.set_node_attributes({0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0,10:0,11:0,12:0,13:1,14:1,15:1,16:1,17:1,18:1,19:1,20:1,21:1,22:1,23:1,24:1,25:1}, name="label")
    g=GH(H, [0,1], true_p, true_q)
    g.add_hyperedge(timesteps, gamma_nu, gamma_nr, gamma_eu, gamma_er)

    return g

if __name__ == "__main__":
    true_p = .9
    true_q = .1
    gamma_nu, gamma_nr, gamma_eu, gamma_er = .3, .1, 1, 0.25
    timesteps = 100
    true_theta = [true_p, true_q, gamma_nu, gamma_nr, gamma_eu, gamma_er]

    g = generate_graph_26_starting_nodes(true_theta, timesteps)

    gd = GradientDescent(true_theta, g)

    gd.run(200)

    
    