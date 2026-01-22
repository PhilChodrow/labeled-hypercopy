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
        self.debug = False

        self.theta = theta
        self.g = g

        self.label_LLs_converted = []
        self.label_LLs = []
        self.label_aris = []

        # with open('gradient_descent_senate_bills.csv', 'a', newline="") as file:
        #     writer = csv.writer(file)
        #     writer.writerows([["making tensors"]])

        self.tensors = self.generate_tensors()

        # self.labels = np.random.uniform(0.4, 0.6, len(self.g.get_labels()))
        self.labels = [.5]*len(g.get_labels())
        self.tensor_labels = torch.full((len(g.get_labels()),), 0.5, requires_grad=True, dtype=torch.double)
        self.optimizer = torch.optim.Adam([self.tensor_labels], lr=learning_rate, betas=momentum)

        self.label_aris.append(adjusted_rand_score(self.g.get_labels(), self.convert_final_labels(self.tensor_labels)))
        self.label_LLs.append(self.differentiable_log_likelihood_3D_tensor(self.tensor_labels).item())
        if self.debug == True:
            self.label_LLs_converted.append(self.g.total_log_likelihood_complete(self.theta, self.convert_final_labels(self.tensor_labels)))

    def generate_canidate_f_indexes(self, e_index):
        canidate_f = []
        e = self.g.get_edges()[e_index]
        for f_index in range(e_index):
            if len(self.g.get_edges()[f_index].intersection(e)) != 0:
                canidate_f.append(f_index)
            
        return canidate_f
    
    def generalized_comb(self, n, k):
        return torch.exp(torch.lgamma(n+1) - (torch.lgamma(k+1)) - (torch.lgamma(n-k+1)))

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

            print(torch.log(e_sum))

        
        
        REGULARIZATION_CONSTANT = 1*len(self.g.get_edges())/5 # LL goes down based on the number of edges, so this makes sense
        regularization = torch.tensor([0.0], requires_grad=True)
        for i in range(len(tensor_labels)):
            regularization = regularization +  torch.log(tensor_labels[i]) + torch.log(1-tensor_labels[i])

        return -1*summ - regularization * REGULARIZATION_CONSTANT
    
    def ix_to_tensor(self, ix, m, n):
        if len(ix) != 0:
            wrapped_ix = [(e*m+f, v) for (e, f, v) in ix]
            return torch.sparse_coo_tensor(
                indices=torch.tensor(wrapped_ix).T,
                values=torch.ones(len(wrapped_ix)),
                size=(m*m, n), check_invariants=False, requires_grad=False,
                dtype=torch.float64
            )
        else:
            return torch.sparse_coo_tensor(size=(m*m, n), check_invariants=False, requires_grad=False, dtype=torch.float64)

    def generate_tensors(self):
        n = len(self.g.get_labels())
        m = len(self.g.get_edges())
        batch_size = int(m)

        # for batch_index in range(0,100):

        

        ix_cop = []
        ix_notcop = []
        ix_ext = []
        ix_posext = []
        ix_nov = []

    
        # with open('gradient_descent_senate_bills.csv', 'a', newline="") as file:
        #     writer = csv.writer(file)
        #     writer.writerows([[len(self.g.get_edges())]])
        for e_index, e in enumerate(self.g.get_edges()):
            
            # hardcoded in for testing
            if e_index >= batch_size: 
                break 
            
            if e_index % 1000 == 0:
                with open('gradient_descent_senate_bills.csv', 'a', newline="") as file:
                    writer = csv.writer(file)
                    writer.writerows([[e_index]])
                

            # neighbors = H.edges.neighbors(e)
            e = self.g.get_edges()[e_index]
            prev_nodes = list(range(self.g.last_added[e_index - 1] + 1))
            novel_nodes = set(e) - set(prev_nodes)
            for f_index in self.generate_canidate_f_indexes(e_index):
                f = self.g.get_edges()[f_index]
                
                copied_nodes = e.intersection(f)
                not_copied_nodes = set(f) - set(e)
                
                external_nodes = set(prev_nodes) - set(f)
                external_nodes_added = external_nodes.intersection(e)

                if len(copied_nodes) != 0:
                    for v in copied_nodes: 
                        ix_cop.append((int(e_index), int(f_index), int(v)))

                if len(not_copied_nodes) != 0:
                    for v in not_copied_nodes: 
                        ix_notcop.append((int(e_index), int(f_index), int(v)))

                if len(novel_nodes) != 0:
                    for v in novel_nodes: 
                        ix_nov.append((int(e_index), int(f_index), int(v)))

                # TODO: try to get rid of possible external nodes for compute ease
                # if len(external_nodes) != 0:
                #     for v in external_nodes: 
                #         ix_posext.append((int(e_index), int(f_index), int(v)))

                if len(external_nodes_added) != 0:
                    for v in external_nodes_added: 
                        ix_ext.append((int(e_index), int(f_index), int(v)))

        # with open('gradient_descent_senate_bills.csv', 'a', newline="") as file:
        #     writer = csv.writer(file)
        #     writer.writerows([["made ix"]])

        cop_tens = self.ix_to_tensor(ix_cop, m, n)
        with open('gradient_descent_senate_bills.csv', 'a', newline="") as file:
            writer = csv.writer(file)
            writer.writerows([['made tensor']])
        notcop_tens = self.ix_to_tensor(ix_notcop, m, n)
        with open('gradient_descent_senate_bills.csv', 'a', newline="") as file:
            writer = csv.writer(file)
            writer.writerows([['made tensor']])
        nov_tens = self.ix_to_tensor(ix_nov, m, n)
        with open('gradient_descent_senate_bills.csv', 'a', newline="") as file:
            writer = csv.writer(file)
            writer.writerows([['made tensor']])
        ext_tens = self.ix_to_tensor(ix_ext, m, n)
        with open('gradient_descent_senate_bills.csv', 'a', newline="") as file:
            writer = csv.writer(file)
            writer.writerows([['made tensor']])
        # TODO: testing
        # posext_tens = self.ix_to_tensor(ix_posext, m, n)
        posext_tens = self.ix_to_tensor([], m, n)


        return cop_tens, notcop_tens, nov_tens, ext_tens, posext_tens

    def differentiable_log_likelihood_3D_tensor(self, tensor_labels):
        # with open('gradient_descent_senate_bills.csv', 'a', newline="") as file:
        #     writer = csv.writer(file)
        #     writer.writerows([["running log likelihood"]])
        m = len(self.g.get_edges())
        p, q, gamma_nu, gamma_nr, gamma_eu, gamma_er = true_theta

        summ = torch.tensor([0.0], requires_grad=True)

        with open('gradient_descent_senate_bills.csv', 'a', newline="") as file:
            writer = csv.writer(file)
            writer.writerows([['calling LL']])

        one_minus_tensor_labels = 1-tensor_labels

        cop1 = self.tensors[0]@tensor_labels
        cop0 = self.tensors[0]@one_minus_tensor_labels

        with open('gradient_descent_senate_bills.csv', 'a', newline="") as file:
            writer = csv.writer(file)
            writer.writerows([['calling LL']])

        notcop1 = self.tensors[1]@tensor_labels
        notcop0 = self.tensors[1]@one_minus_tensor_labels

        nov1 = self.tensors[2]@tensor_labels
        nov0 = self.tensors[2]@one_minus_tensor_labels

        ext1 = self.tensors[3]@tensor_labels
        ext0 = self.tensors[3]@one_minus_tensor_labels

        # posext1 = self.tensors[4]@tensor_labels
        # posext0 = self.tensors[4]@(1-tensor_labels)

        with open('gradient_descent_senate_bills.csv', 'a', newline="") as file:
            writer = csv.writer(file)
            writer.writerows([['generated label vals']])

        # cop1_tens = self.tensors[0]@tensor_labels
        # cop0_tens = self.tensors[0]@(1-tensor_labels)

        # notcop1_tens = self.tensors[1]@tensor_labels
        # notcop0_tens = self.tensors[1]@(1-tensor_labels)

        # nov1_tens = self.tensors[2]@tensor_labels
        # nov0_tens = self.tensors[2]@(1-tensor_labels)

        # ext1_tens = self.tensors[3]@tensor_labels
        # ext0_tens = self.tensors[3]@(1-tensor_labels)

        # posext1_tens = self.tensors[4]@tensor_labels
        # posext0_tens = self.tensors[4]@(1-tensor_labels)

        # for e_index in range(1, len(self.g.get_edges())):
        #     canidate_f_indexes = self.generate_canidate_f_indexes(e_index)
            

        #     e_sum = torch.tensor([0.0], requires_grad=True)
        #     for f_index in canidate_f_indexes:
        #         cop1 = cop1_tens[e_index*m + f_index]
        #         cop0 = cop0_tens[e_index*m + f_index]
        #         notcop1 = notcop1_tens[e_index*m + f_index]
        #         notcop0 = notcop0_tens[e_index*m + f_index]
        #         nov1 = nov1_tens[e_index*m + f_index]
        #         nov0 = nov0_tens[e_index*m + f_index]
        #         ext1 = ext1_tens[e_index*m + f_index]
        #         ext0 = ext0_tens[e_index*m + f_index]
        #         posext1 = posext1_tens[e_index*m + f_index]
        #         posext0 = posext0_tens[e_index*m + f_index]
                
                
        #         prob_u_label_equals_1 = cop1 / (cop1 + cop0)
        #         prob_u_label_equals_0 = 1 - prob_u_label_equals_1

        #         prob_given_f_u_label_1 = (p**(cop1-1)) * ((1-p)**notcop1) * (q**cop0) * ((1-q)**notcop0)
        #         prob_given_f_u_label_1 = prob_given_f_u_label_1 * (gamma_eu**ext1) * math.exp(-gamma_eu) / math.gamma(ext1+1) / self.generalized_comb(posext1, ext1)
        #         prob_given_f_u_label_1 = prob_given_f_u_label_1 * (gamma_er**ext0) * math.exp(-gamma_er) / math.gamma(ext0+1) / self.generalized_comb(posext0, ext0)
        #         prob_given_f_u_label_1 = prob_given_f_u_label_1 * (gamma_nu**nov1) * math.exp(-gamma_nu) / math.gamma(nov1+1)
        #         prob_given_f_u_label_1 = prob_given_f_u_label_1 * (gamma_nr**nov0) * math.exp(-gamma_nr) / math.gamma(nov0+1)

        #         # calculate prob of e given f and u_label = 0
        #         prob_given_f_u_label_0 = (p**(cop0-1)) * ((1-p)**notcop0) * (q**cop1) * ((1-q)**notcop1)
        #         prob_given_f_u_label_0 = prob_given_f_u_label_0 * (gamma_eu**ext0) * math.exp(-gamma_eu) / math.gamma(ext0+1) / self.generalized_comb(posext0, ext0)
        #         prob_given_f_u_label_0 = prob_given_f_u_label_0 * (gamma_er**ext1) * math.exp(-gamma_er) / math.gamma(ext1+1) / self.generalized_comb(posext1, ext1)
        #         prob_given_f_u_label_0 = prob_given_f_u_label_0 * (gamma_nu**nov0) * math.exp(-gamma_nu) / math.gamma(nov0+1)
        #         prob_given_f_u_label_0 = prob_given_f_u_label_0 * (gamma_nr**nov1) * math.exp(-gamma_nr) / math.gamma(nov1+1)

        #         e_sum = e_sum + (prob_u_label_equals_1*prob_given_f_u_label_1 + prob_u_label_equals_0*prob_given_f_u_label_0)/e_index

            # summ = summ + torch.log(e_sum)

        # try vectorized
        prob_u_label_equals_1 = cop1 / (cop1 + cop0)
        prob_u_label_equals_0 = cop0 / (cop1 + cop0)
        prob_u_label_equals_1 = torch.nan_to_num(prob_u_label_equals_1, nan=0.0)
        prob_u_label_equals_0 = torch.nan_to_num(prob_u_label_equals_0, nan=0.0)

        # as approximate stand in for external nodes of each label
        count_1_label_nodes = torch.sum(tensor_labels).detach().clone()
        count_0_label_nodes = torch.sum(1-tensor_labels).detach().clone()

        

        prob_given_f_u_label_1 = (torch.pow(p, (cop1-1)) * (torch.pow((1-p), notcop1)) * (torch.pow(q,cop0)) * (torch.pow((1-q), notcop0)))
        prob_given_f_u_label_1 = prob_given_f_u_label_1 * torch.pow(gamma_eu,ext1) * math.exp(-gamma_eu) / torch.exp(torch.lgamma(ext1+1)) / self.generalized_comb(count_1_label_nodes, ext1)
        prob_given_f_u_label_1 = prob_given_f_u_label_1 * torch.pow(gamma_er,ext0) * math.exp(-gamma_er) / torch.exp(torch.lgamma(ext0+1)) / self.generalized_comb(count_0_label_nodes, ext0)
        prob_given_f_u_label_1 = prob_given_f_u_label_1 * torch.pow(gamma_nu,nov1) * math.exp(-gamma_nu) / torch.exp(torch.lgamma(nov1+1))
        prob_given_f_u_label_1 = prob_given_f_u_label_1 * torch.pow(gamma_nr,nov0) * math.exp(-gamma_nr) / torch.exp(torch.lgamma(nov0+1))

        # calculate prob of e given f and u_label = 0
        prob_given_f_u_label_0 = (torch.pow(p, (cop0-1)) * (torch.pow((1-p), notcop0)) * (torch.pow(q, cop1)) * (torch.pow((1-q), notcop1)))
        prob_given_f_u_label_0 = prob_given_f_u_label_0 * torch.pow(gamma_eu,ext0) * math.exp(-gamma_eu) / torch.exp(torch.lgamma(ext0+1)) / self.generalized_comb(count_0_label_nodes, ext0)
        prob_given_f_u_label_0 = prob_given_f_u_label_0 * torch.pow(gamma_er,ext1) * math.exp(-gamma_er) / torch.exp(torch.lgamma(ext1+1)) / self.generalized_comb(count_1_label_nodes, ext1)
        prob_given_f_u_label_0 = prob_given_f_u_label_0 * torch.pow(gamma_nu,nov0) * math.exp(-gamma_nu) / torch.exp(torch.lgamma(nov0+1))
        prob_given_f_u_label_0 = prob_given_f_u_label_0 * torch.pow(gamma_nr,nov1) * math.exp(-gamma_nr) / torch.exp(torch.lgamma(nov1+1))




        divide_by_e_index = []
        for i in range(len(self.g.get_edges())):
            divide_by_e_index.extend([float(i)] * len(self.g.get_edges()))


        divide_by_e_index = torch.tensor(divide_by_e_index, requires_grad=True)

            
        summ = ((prob_u_label_equals_1*prob_given_f_u_label_1 + prob_u_label_equals_0*prob_given_f_u_label_0)) /divide_by_e_index
        # summ = torch.nan_to_num(summ, nan=0.0, neginf=0.0)
        # need to log all of the individual e_sums

        # e_sums = [0.0] * len(self.g.get_edges())
        # e_sums = torch.tensor(e_sums)


        # replace with array math
        # for i in range(len(self.g.get_edges())):
        #     for j in range(len(self.g.get_edges())):
        #         e_sums[i] = e_sums[i] +  summ[i*len(self.g.get_edges()) + j]
        # summ = torch.sum(torch.log(e_sums[1:]))

        summ = summ.reshape(len(self.g.get_edges()), len(self.g.get_edges()))
        summ = torch.sum(summ, dim=1)

        # add epsilon to avoid infinity underflow
        epsilon = torch.tensor(10**-40, requires_grad=True)
        # epsilon = 10**-40
        # epsilon = 0
        summ = torch.sum(torch.log(summ[1:] + epsilon))
        # summ = torch.sum(torch.log(summ[1:]))

        

        


        # e_sums[0] = 1.0
        # all impossible likelihoods are nan / -inf, but should be zero
        
        


        # print(prob_given_f_u_label_0)
        # print(prob_given_f_u_label_1)
        # print(prob_u_label_equals_1)
        # print(prob_u_label_equals_0)
        # print(torch.log(e_sums))


        # with open('gradient_descent_senate_bills.csv', 'a', newline="") as file:
        #     writer = csv.writer(file)
        #     writer.writerows([["finished LL"]])

        
        REGULARIZATION_CONSTANT = math.sqrt(len(self.g.get_edges())) # LL goes down based on the number of edges, so this makes sense
        # regularization = torch.tensor([0.0], requires_grad=True)
        # for i in range(len(tensor_labels)):
        #     regularization = regularization + torch.log(tensor_labels[i]) + torch.log(1-tensor_labels[i])

        regularization = torch.sum(torch.log(tensor_labels) + torch.log(1-tensor_labels))

    
        # print("new LL: " + str((-1*summ - regularization * REGULARIZATION_CONSTANT)))

        # print("old LL: " + str(self.differentiable_log_likelihood(tensor_labels)))

        with open('gradient_descent_senate_bills_results.csv', 'a', newline="") as file:
            writer = csv.writer(file)
            writer.writerows([[-1*summ - regularization * REGULARIZATION_CONSTANT]])

        return -1*summ - regularization * REGULARIZATION_CONSTANT
    
    def generate_canidate_f_indexes_approx(self, e_index):
        canidate_f_indexes = []

        e = self.g.get_edges()[e_index]
        k = 0
        canidate_f_indexes = []
        for f_index in range(e_index):
            f = self.g.get_edges()[f_index]
            inter_size = len(e.intersection(f))

            if inter_size == k and k!=0:
                canidate_f_indexes.append(f_index)
            elif inter_size > k:
                k = inter_size
                canidate_f_indexes = [f_index]
        
        return canidate_f_indexes

        # for e_index in range(1, len(self.g.get_edges())):
        #     e = self.g.get_edges()[e_index]
        #     ks = []
        #     canidate_f_indexes = []
        #     for f_index in range(e_index):
        #         f = self.g.get_edges()[f_index]
        #         inter_size = len(e.intersection(f))

        #         if (inter_size != 0):
        #             canidate_f_indexes.append((f_index, inter_size))
            
        #     canidate_f_indexes = sorted(canidate_f_indexes, key=lambda x: x[1])
        #     for f_pair in canidate_f_indexes[-min(bound, len(canidate_f_indexes)):]:
        #         f_e_pairs.append(FEPair(self.g, e_index, f_pair[0], 1/len(canidate_f_indexes), self.labels, self.theta))

        return f_e_pairs

    def generate_tensors_approx(self):
        print("generate tensors")
        n = len(self.g.get_labels())
        m = len(self.g.get_edges())
        batch_size = int(m)

        ix_cop = []
        ix_notcop = []
        ix_ext = []
        ix_posext = []
        ix_nov = []

    
        for e_index, e in enumerate(self.g.get_edges()):
            
            # hardcoded in for testing
            if e_index >= batch_size: 
                break 
            
            if e_index % 1000 == 0:
                with open('gradient_descent_senate_bills.csv', 'a', newline="") as file:
                    writer = csv.writer(file)
                    writer.writerows([[e_index]])
                

            # neighbors = H.edges.neighbors(e)
            e = self.g.get_edges()[e_index]
            prev_nodes = list(range(self.g.last_added[e_index - 1] + 1))
            novel_nodes = set(e) - set(prev_nodes)
            for f_index in self.generate_canidate_f_indexes_approx(e_index):
                f = self.g.get_edges()[f_index]
                
                copied_nodes = e.intersection(f)
                not_copied_nodes = set(f) - set(e)
                
                external_nodes = set(prev_nodes) - set(f)
                external_nodes_added = external_nodes.intersection(e)

                if len(copied_nodes) != 0:
                    for v in copied_nodes: 
                        ix_cop.append((int(e_index), int(f_index), int(v)))

                if len(not_copied_nodes) != 0:
                    for v in not_copied_nodes: 
                        ix_notcop.append((int(e_index), int(f_index), int(v)))

                if len(novel_nodes) != 0:
                    for v in novel_nodes: 
                        ix_nov.append((int(e_index), int(f_index), int(v)))

                # TODO: try to get rid of possible external nodes for compute ease
                # if len(external_nodes) != 0:
                #     for v in external_nodes: 
                #         ix_posext.append((int(e_index), int(f_index), int(v)))

                if len(external_nodes_added) != 0:
                    for v in external_nodes_added: 
                        ix_ext.append((int(e_index), int(f_index), int(v)))

        # with open('gradient_descent_senate_bills.csv', 'a', newline="") as file:
        #     writer = csv.writer(file)
        #     writer.writerows([["made ix"]])

        cop_tens = self.ix_to_tensor(ix_cop, m, n)
        with open('gradient_descent_senate_bills.csv', 'a', newline="") as file:
            writer = csv.writer(file)
            writer.writerows([['made tensor']])
        notcop_tens = self.ix_to_tensor(ix_notcop, m, n)
        with open('gradient_descent_senate_bills.csv', 'a', newline="") as file:
            writer = csv.writer(file)
            writer.writerows([['made tensor']])
        nov_tens = self.ix_to_tensor(ix_nov, m, n)
        with open('gradient_descent_senate_bills.csv', 'a', newline="") as file:
            writer = csv.writer(file)
            writer.writerows([['made tensor']])
        ext_tens = self.ix_to_tensor(ix_ext, m, n)
        with open('gradient_descent_senate_bills.csv', 'a', newline="") as file:
            writer = csv.writer(file)
            writer.writerows([['made tensor']])
        # TODO: testing
        # posext_tens = self.ix_to_tensor(ix_posext, m, n)
        posext_tens = self.ix_to_tensor([], m, n)

        print("generated_tensors")
        return cop_tens, notcop_tens, nov_tens, ext_tens, posext_tens
        
    
    def run(self, num_steps):
        # last_final_labels = self.convert_final_labels(self.tensor_labels)
        equal_count = 0

        for step in range(num_steps):
            if equal_count == 20:
                break
            loss = self.differentiable_log_likelihood_3D_tensor(self.tensor_labels)
            # print(loss)
            if (torch.isnan(loss)):
                print("NAN")
                break
            loss.backward()

            # print("pre-tensor labels")
            # print(self.tensor_labels)
            self.optimizer.step()
            self.optimizer.zero_grad()

            #print("tensor labels")
            #print(self.tensor_labels)
            final_labels = self.convert_final_labels(self.tensor_labels)
            #print(final_labels)
            if self.debug == True:
                new_loss = self.g.total_log_likelihood_complete(self.theta, final_labels)
            new_ari = adjusted_rand_score(self.g.get_labels(), final_labels)
            # print("New label likelihood: " + str(new_loss))
            #print("New label ari: " + str(new_ari))

            if self.labels == final_labels:
                equal_count += 1
            else:
                self.labels = final_labels
                equal_count = 0

            self.label_aris.append(new_ari)
            self.label_LLs.append(loss.item())
            if self.debug == True:
                self.label_LLs_converted.append(new_loss)

        # true label likelihood
        # print("True label likelihood: " + str(self.g.total_log_likelihood_complete(self.theta, self.g.get_labels())))
        # print("New label likelihood final: " + str(self.label_LLs_converted[-1]))

        return self.convert_final_labels(self.tensor_labels)

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

# if __name__ == "__main__":
#     true_p = .9
#     true_q = .1
#     gamma_nu, gamma_nr, gamma_eu, gamma_er = .3, .1, 1, 0.25
#     timesteps = 100
#     true_theta = [true_p, true_q, gamma_nu, gamma_nr, gamma_eu, gamma_er]

#     g = generate_graph_26_starting_nodes(true_theta, timesteps)

#     gd = GradientDescent(true_theta, g)

#     gd.run(200)

    
    