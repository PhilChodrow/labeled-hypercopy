import xgi
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
import math
import scipy.special as ss; ss.binom

class GH:
    def __init__(self, H, labels, p, q):
        self.H = H
        self.labels = labels
        self.p = p
        self.q = q
        self.edge_members = H.edges.members()
        self.node_labels = H.nodes.attrs("label").aslist()
        self.nodes = list(H.nodes)
        self.last_added = [max(H.nodes)] * len(H.edges)
        self.total_num_1 = sum(self.node_labels)
        self.total_num_0 = len(self.node_labels) - self.total_num_1

        # helper for speed of gammaln
        self.stored_gammaln = [ss.gammaln(k) for k in range(200)]

    def get_labels(self):
        return(self.node_labels)
    
    def get_edges(self):
        return(self.edge_members)
    
    def set_values(self, value0, value1, u_label):
        if u_label == 0:
            return([value0, value1])
        else: 
            return([value1, value0])
        
    def return_key_values(self, e_index, u_index, e_prime_index, theta):
        p, q, gamma_nu, gamma_nr, gamma_eu, gamma_er = theta

        node_labels = self.node_labels

        e = self.edge_members[e_index]
        e_prime = self.edge_members[e_prime_index]
        u_label = node_labels[u_index]

        intersect = e.intersection(e_prime)

        if len(intersect) == 0:
            return(0)
        if u_index not in e_prime:
            return(0)

        node_labels = self.get_labels()
        e_labels = [node_labels[node] for node in e]
        e_prime_labels = [node_labels[node] for node in e_prime]
        int_labels = [node_labels[node] for node in intersect]

        # Get edge and intersection sizes
        e_prime_num_1 = sum(e_prime_labels)
        e_prime_num_0 = len(e_prime_labels) - e_prime_num_1
        e_num_1 = sum(e_labels)
        e_num_0 = len(e_labels) - e_num_1
        e_num_u, e_num_r = self.set_values(e_num_0, e_num_1, u_label)
        e_prime_num_u, e_prime_num_r = self.set_values(e_prime_num_0, e_prime_num_1, u_label)
        int_num_u, int_num_r = self.set_values(len(int_labels) - sum(int_labels), sum(int_labels), u_label)

        # Get the new nodes
        prev_edges = self.edge_members[0:e_prime_index]
        prev_nodes = list(range(self.last_added[e_prime_index - 1] + 1))

        novel_nodes = set(e_prime) - set(prev_nodes)
        novel_labels = [node_labels[node] for node in novel_nodes]

        novel_num_u, novel_num_r = self.set_values(len(novel_labels) - sum(novel_labels), sum(novel_labels), u_label)
      
        # Get the external nodes
        # total_num_1 = sum(node_labels)
        # total_num_0 = len(node_labels) - total_num_1
        all_ext_num_0 = self.total_num_0 - e_num_0
        all_ext_num_1 = self.total_num_1 - e_num_1

        # external_nodes = set(prev_nodes).intersection(e_prime) - set(e)
        # external_labels = [node_labels[node] for node in external_nodes]           
        ext_num_u = e_prime_num_u - int_num_u - novel_num_u
        ext_num_r = e_prime_num_r - int_num_r - novel_num_r


        all_ext_num_u, all_ext_num_r = self.set_values(all_ext_num_0, all_ext_num_1, u_label)
        # ext_num_u, ext_num_r = self.set_values(len(external_labels) - sum(external_labels), sum(external_labels), u_label)

        # Probability calculation
        prob_e = 1 / e_prime_index

        prob_u = 1 / len(e)

        P1 = p ** (int_num_u - 1)
        P2 = (1 - p) ** (e_num_u - int_num_u)
        P3 = q ** (int_num_r)
        P4 = (1 - q) ** (e_num_r - int_num_r)

        if ext_num_u == 0:
            prob_those_ext_u = 1
        else:
            prob_those_ext_u = 1 / ss.binom(all_ext_num_u, ext_num_u) # formerly: ext_num_u / all_ext_num_u
        if ext_num_r == 0:
            prob_those_ext_r = 1
        else: prob_those_ext_r = 1 / ss.binom(all_ext_num_r, ext_num_r) # formerly: ext_num_r / all_ext_num_r

        P5_numer = (math.e ** (-gamma_eu)) * (gamma_eu ** ext_num_u) * prob_those_ext_u * (math.e ** (-gamma_er)) * (gamma_er ** ext_num_r) * prob_those_ext_r
        P5_denom = math.factorial(ext_num_u) * math.factorial(ext_num_r)
        P5 = P5_numer / P5_denom
        P6_numer = (math.e ** (-gamma_nu)) * (gamma_nu ** novel_num_u) * (math.e ** (-gamma_nr)) * (gamma_nr ** novel_num_r)
        P6_denom = (math.factorial(novel_num_u) * math.factorial(novel_num_r))
        P6 = P6_numer / P6_denom
        prob_e_prime = P1 * P2 * P3 * P4  * P5 * P6
        
        prob = prob_e * prob_u * prob_e_prime

        return(prob)        
        
    def likelihood(self, e_index, u_index, e_prime_index, theta):
        p, q, gamma_nu, gamma_nr, gamma_eu, gamma_er = theta

        node_labels = self.node_labels

        e = self.edge_members[e_index]
        e_prime = self.edge_members[e_prime_index]
        u_label = node_labels[u_index]

        intersect = e.intersection(e_prime)

        if len(intersect) == 0:
            return(0)
        if u_index not in e_prime:
            return(0)

        node_labels = self.get_labels()
        e_labels = [node_labels[node] for node in e]
        e_prime_labels = [node_labels[node] for node in e_prime]
        int_labels = [node_labels[node] for node in intersect]

        # Get edge and intersection sizes
        e_prime_num_1 = sum(e_prime_labels)
        e_prime_num_0 = len(e_prime_labels) - e_prime_num_1
        e_num_1 = sum(e_labels)
        e_num_0 = len(e_labels) - e_num_1
        e_num_u, e_num_r = self.set_values(e_num_0, e_num_1, u_label)
        e_prime_num_u, e_prime_num_r = self.set_values(e_prime_num_0, e_prime_num_1, u_label)
        int_num_u, int_num_r = self.set_values(len(int_labels) - sum(int_labels), sum(int_labels), u_label)

        # Get the new nodes
        prev_edges = self.edge_members[0:e_prime_index]
        prev_nodes = list(range(self.last_added[e_prime_index - 1] + 1))

        novel_nodes = set(e_prime) - set(prev_nodes)
        novel_labels = [node_labels[node] for node in novel_nodes]

        novel_num_u, novel_num_r = self.set_values(len(novel_labels) - sum(novel_labels), sum(novel_labels), u_label)        
      
        # Get the external nodes
        # total_num_1 = sum(node_labels)
        # total_num_0 = len(node_labels) - total_num_1
        # CHANGE HERE: substracted novel nodes from total external nodes as novel nodes are not possible external nodes

        # previously all_ext_num_0 = self.total_num_0 - e_num_0
        # previously all_ext_num_1 = self.total_num_1 - e_num_1

        # new code
        all_ext_nodes = set(prev_nodes) - e
        all_ext_labels = [node_labels[node] for node in all_ext_nodes]

        all_ext_num_1 = sum(all_ext_labels)
        all_ext_num_0 = len(all_ext_labels) - sum(all_ext_labels) 

        # external_nodes = set(prev_nodes).intersection(e_prime) - set(e)
        # external_labels = [node_labels[node] for node in external_nodes]           
        ext_num_u = e_prime_num_u - int_num_u - novel_num_u
        ext_num_r = e_prime_num_r - int_num_r - novel_num_r

        all_ext_num_u, all_ext_num_r = self.set_values(all_ext_num_0, all_ext_num_1, u_label)
        # ext_num_u, ext_num_r = self.set_values(len(external_labels) - sum(external_labels), sum(external_labels), u_label)

        # Probability calculation
        prob_e = 1 / e_prime_index

        prob_u = 1 / len(e)

        P1 = p ** (int_num_u - 1)
        P2 = (1 - p) ** (e_num_u - int_num_u)
        P3 = q ** (int_num_r)
        P4 = (1 - q) ** (e_num_r - int_num_r)

        if ext_num_u == 0:
            prob_those_ext_u = 1
        else:
            prob_those_ext_u = 1 / ss.binom(all_ext_num_u, ext_num_u) # formerly: ext_num_u / all_ext_num_u
        if ext_num_r == 0:
            prob_those_ext_r = 1
        else: prob_those_ext_r = 1 / ss.binom(all_ext_num_r, ext_num_r) # formerly: ext_num_r / all_ext_num_r

        P5_numer = (math.e ** (-gamma_eu)) * (gamma_eu ** ext_num_u) * prob_those_ext_u * (math.e ** (-gamma_er)) * (gamma_er ** ext_num_r) * prob_those_ext_r
        P5_denom = math.factorial(ext_num_u) * math.factorial(ext_num_r)
        P5 = P5_numer / P5_denom
        P6_numer = (math.e ** (-gamma_nu)) * (gamma_nu ** novel_num_u) * (math.e ** (-gamma_nr)) * (gamma_nr ** novel_num_r)
        P6_denom = (math.factorial(novel_num_u) * math.factorial(novel_num_r))
        P6 = P6_numer / P6_denom
        prob_e_prime = P1 * P2 * P3 * P4  * P5 * P6
        
        prob = prob_e_prime * prob_e * prob_u

        return(prob)
    
    def likelihood_given_u_f(self, e_index, u_index, e_prime_index, theta):
        # NOTE: e = f, e_prime = e
        p, q, gamma_nu, gamma_nr, gamma_eu, gamma_er = theta

        node_labels = self.node_labels

        e = self.edge_members[e_index]
        e_prime = self.edge_members[e_prime_index]
        u_label = node_labels[u_index]

        intersect = e.intersection(e_prime)

        if len(intersect) == 0:
            return(0)
        if u_index not in e_prime:
            return(0)

        node_labels = self.get_labels()
        e_labels = [node_labels[node] for node in e]
        e_prime_labels = [node_labels[node] for node in e_prime]
        int_labels = [node_labels[node] for node in intersect]

        # Get edge and intersection sizes
        e_prime_num_1 = sum(e_prime_labels)
        e_prime_num_0 = len(e_prime_labels) - e_prime_num_1
        e_num_1 = sum(e_labels)
        e_num_0 = len(e_labels) - e_num_1
        e_num_u, e_num_r = self.set_values(e_num_0, e_num_1, u_label)
        e_prime_num_u, e_prime_num_r = self.set_values(e_prime_num_0, e_prime_num_1, u_label)
        int_num_u, int_num_r = self.set_values(len(int_labels) - sum(int_labels), sum(int_labels), u_label)

        # Get the new nodes
        prev_edges = self.edge_members[0:e_prime_index]
        prev_nodes = list(range(self.last_added[e_prime_index - 1] + 1))

        novel_nodes = set(e_prime) - set(prev_nodes)
        novel_labels = [node_labels[node] for node in novel_nodes]

        novel_num_u, novel_num_r = self.set_values(len(novel_labels) - sum(novel_labels), sum(novel_labels), u_label)        
      
        # Get the external nodes
        # total_num_1 = sum(node_labels)
        # total_num_0 = len(node_labels) - total_num_1
        # CHANGE HERE: substracted novel nodes from total external nodes as novel nodes are not possible external nodes

        # previously all_ext_num_0 = self.total_num_0 - e_num_0
        # previously all_ext_num_1 = self.total_num_1 - e_num_1

        # new code
        all_ext_nodes = set(prev_nodes) - e
        all_ext_labels = [node_labels[node] for node in all_ext_nodes]

        all_ext_num_1 = sum(all_ext_labels)
        all_ext_num_0 = len(all_ext_labels) - sum(all_ext_labels) 

        # external_nodes = set(prev_nodes).intersection(e_prime) - set(e)
        # external_labels = [node_labels[node] for node in external_nodes]           
        ext_num_u = e_prime_num_u - int_num_u - novel_num_u
        ext_num_r = e_prime_num_r - int_num_r - novel_num_r

        all_ext_num_u, all_ext_num_r = self.set_values(all_ext_num_0, all_ext_num_1, u_label)
        # ext_num_u, ext_num_r = self.set_values(len(external_labels) - sum(external_labels), sum(external_labels), u_label)

        P1 = p ** (int_num_u - 1)
        P2 = (1 - p) ** (e_num_u - int_num_u)
        P3 = q ** (int_num_r)
        P4 = (1 - q) ** (e_num_r - int_num_r)

        if ext_num_u == 0:
            prob_those_ext_u = 1
        else:
            prob_those_ext_u = 1 / ss.binom(all_ext_num_u, ext_num_u) # formerly: ext_num_u / all_ext_num_u
        if ext_num_r == 0:
            prob_those_ext_r = 1
        else: prob_those_ext_r = 1 / ss.binom(all_ext_num_r, ext_num_r) # formerly: ext_num_r / all_ext_num_r

        P5_numer = (math.e ** (-gamma_eu)) * (gamma_eu ** ext_num_u) * prob_those_ext_u * (math.e ** (-gamma_er)) * (gamma_er ** ext_num_r) * prob_those_ext_r
        P5_denom = math.factorial(ext_num_u) * math.factorial(ext_num_r)
        P5 = P5_numer / P5_denom
        P6_numer = (math.e ** (-gamma_nu)) * (gamma_nu ** novel_num_u) * (math.e ** (-gamma_nr)) * (gamma_nr ** novel_num_r)
        P6_denom = (math.factorial(novel_num_u) * math.factorial(novel_num_r))
        P6 = P6_numer / P6_denom
        prob_e_prime = P1 * P2 * P3 * P4  * P5 * P6
        
        prob = prob_e_prime

        return(prob)

    def add_hyperedge(self, num_edges = 1, gamma_nu = 1, gamma_nr = 1, gamma_eu = 1, gamma_er = 1):
        for i in range(num_edges):
            e_prime = [] # create an empty new edge called e_prime

            ## randomly select an existing hyperedge e to start with
            e_num = random.randint(0, len(self.edge_members) - 1)
            e = self.edge_members[e_num]
            e_size = len(e)

            ## randomly select a node u from e
            u_num = random.randint(0, e_size - 1)
            u = list(e)[u_num]
            u_label = self.node_labels[u]
            e_prime.append(u)

            ## get r label
            if self.labels[0] == u_label:
                r_label = self.labels[1]
            else:
                r_label = self.labels[0]

            ## add other nodes from e to e_prime
            # add node with same label as u with prob p
            # add node with different label from u with prob q
            for node in e:
                if self.node_labels[node] == u_label:
                    prob = self.p
                else:
                    prob = self.q
                if random.random() < prob:
                    e_prime.append(node)
            
            ## add exterior nodes
            num_ext_u = np.random.poisson(gamma_eu, 1)[0]
            num_ext_r = np.random.poisson(gamma_er, 1)[0]

            u_indices = list(filter(lambda x: self.node_labels[x] == u_label, self.nodes))
            r_indices = list(filter(lambda x: self.node_labels[x] != u_label, self.nodes))

            for i in range(0, num_ext_u):
                if len(set(u_indices) - set(e)) > 0:
                    exterior_node = random.sample(list(set(u_indices) - set(e)), 1)[0] # randomly sample a node from outside the existing hyperedge
                    e_prime.append(exterior_node)

            for i in range(0, num_ext_r):
                if len(set(r_indices) - set(e)) > 0:
                    exterior_node = random.sample(list(set(r_indices) - set(e)), 1)[0] # randomly sample a node from outside the existing hyperedge
                    e_prime.append(exterior_node)

            ## Add new nodes        
            num_new_u = np.random.poisson(gamma_nu, 1)[0]
            num_new_r = np.random.poisson(gamma_nr, 1)[0]

            last = self.last_added[-1]
            for i in range(0, num_new_u):
                new_node = len(self.nodes)
                self.nodes.append(new_node)
                self.node_labels.append(u_label)
                e_prime.append(new_node)
                last = new_node
                if u_label == 0:
                    self.total_num_0 += 1
                else:
                    self.total_num_1 += 1

            for i in range(0, num_new_r):
                new_node = len(self.nodes)
                self.nodes.append(new_node)
                self.node_labels.append(r_label)
                e_prime.append(new_node)
                last = new_node
                if u_label == 1:
                    self.total_num_0 += 1
                else:
                    self.total_num_1 += 1

            ## Add the edge to the hypergraph
            self.H.add_edge(e_prime)
            self.edge_members.append(set(e_prime))
            self.last_added.append(last)

        # big_H = xgi.Hypergraph(self.edge_members)
        # node_dict = dict(zip(self.nodes, self.node_labels))
        # big_H.set_node_attributes(node_dict, name = "label")
        # return(big_H)

    def log_likelihood_given_u_f(self, e_index, u_index, e_prime_index, theta, label):

        # log_likelihood of an edge e_prime given node u and edge e

       
        # can define node labels within the function with the line node_labels = self.get_labels(),
        # but the current config is betteer for testing
        p, q, gamma_nu, gamma_nr, gamma_eu, gamma_er = theta

        node_labels = label
        #NOTE: f = e, e = eprime in the following code

        e = self.edge_members[e_prime_index]
        f = self.edge_members[e_index]
        u_label = node_labels[u_index]

        intersect = e.intersection(f)

        if len(intersect) == 0:
            return np.log(0)
        elif u_index not in e:
            return np.log(0)

        intersect = set(intersect)

   
        e_labels = [node_labels[node] for node in e]

        f_labels = [node_labels[node] for node in f]
        int_labels = [node_labels[node] for node in intersect]

        # nodes added in steps 3, 4
        Sef1 = sum(int_labels)
        Sef0 = len(int_labels) - sum(int_labels)


        f1 = sum(f_labels)
        f0 = len(f_labels) - sum(f_labels)

        e1 = sum(e_labels)
        e0 = len(e_labels) - sum(e_labels)

        prev_edges = self.edge_members[0:e_prime_index]
        prev_nodes = list(range(self.last_added[e_prime_index - 1] + 1))


        # novel nodes (added via steps 7, 8)
        novel_nodes = set(e) - set(prev_nodes)
        novel_labels = [node_labels[node] for node in novel_nodes]

        Snte1 = sum(novel_labels)
        Snte0 = len(novel_labels) - sum(novel_labels)


        # external nodes in e (added via steps 5, 6)
        external_nodes = set(prev_nodes) - set(f)
        external_node_labels = [node_labels[node] for node in external_nodes]  

        
        external_nodes_in_e = external_nodes.intersection(e)
        external_nodes_in_e_labels = [node_labels[node] for node in external_nodes_in_e] 

        Stnfe1 = sum(external_nodes_in_e_labels)
        
        Stnfe0 = len(external_nodes_in_e_labels) - sum(external_nodes_in_e_labels)

        # nodes not added in step 5, 6, but could be choosen
        external_nodes_not_in_e = external_nodes - (e)
        external_nodes_not_in_e_labels = [node_labels[node] for node in external_nodes_not_in_e]  

        Stnfne1 = sum(external_nodes_not_in_e_labels)
        Stnfne0 = len(external_nodes_not_in_e_labels) - sum(external_nodes_not_in_e_labels)


        Stnf1 = sum(external_node_labels)
        Stnf0 = len(external_node_labels) - sum(external_node_labels)


        # Now, code prob in components and add
        prob_addition = 0
        if u_label == 1:
            prob_addition += Sef1*math.log(p/(1-p)) + f1*math.log(1-p) - math.log(p)
            prob_addition += Sef0*math.log(q/(1-q)) + f0*math.log(1-q)

            prob_addition += Stnfe1*math.log(gamma_eu) - gamma_eu + self.gammaln_fast(Stnfne1+1)
            prob_addition += Stnfe0*math.log(gamma_er) - gamma_er + self.gammaln_fast(Stnfne0+1)

            # prob_addition += Stnfe1*math.log(gamma_eu) - gamma_eu + ss.gammaln(Stnfne1+1)
            # prob_addition += Stnfe0*math.log(gamma_er) - gamma_er + ss.gammaln(Stnfne0+1)


            prob_addition += -1*self.gammaln_fast(Stnf1+1) + -1*self.gammaln_fast(Stnf0+1)
            # prob_addition += -1*ss.gammaln(Stnf1+1) + -1*ss.gammaln(Stnf0+1)

            # prob_addition += (Snte1*math.log(gamma_nu) - gamma_nu - math.log(math.factorial(Snte1)))
            # prob_addition += (Snte0*math.log(gamma_nr) - gamma_nr - math.log(math.factorial(Snte0)))

            prob_addition += (Snte1*math.log(gamma_nu) - gamma_nu - self.gammaln_fast(Snte1+1))
            prob_addition += (Snte0*math.log(gamma_nr) - gamma_nr - self.gammaln_fast(Snte0+1))

            
        else: # u_label = 0
            prob_addition += Sef0*math.log(p/(1-p)) + f0*math.log(1-p) - math.log(p)
            prob_addition += Sef1*math.log(q/(1-q)) + f1*math.log(1-q)

            # prob_addition += Stnfe0*math.log(gamma_eu) - gamma_eu + math.log(math.factorial(Stnfne0))
            # prob_addition += Stnfe1*math.log(gamma_er) - gamma_er + math.log(math.factorial(Stnfne1))

            prob_addition += Stnfe0*math.log(gamma_eu) - gamma_eu + self.gammaln_fast(Stnfne0+1)
            prob_addition += Stnfe1*math.log(gamma_er) - gamma_er + self.gammaln_fast(Stnfne1+1)

            # prob_addition += -1*math.log(math.factorial(Stnf1)) + -1*math.log(math.factorial(Stnf0))
            prob_addition += -1*self.gammaln_fast(Stnf1+1) + -1*self.gammaln_fast(Stnf0+1)

            prob_addition += (Snte0*math.log(gamma_nu) - gamma_nu - self.gammaln_fast(Snte0+1))
            prob_addition += (Snte1*math.log(gamma_nr) - gamma_nr - self.gammaln_fast(Snte1+1))

        return prob_addition
    
    def expected_log_likelihood(self, e_prime_index, theta, label):
        # Note: e_prime = e, e = f for the following code
        # log likelihood of e_prime expected
        p, q, gamma_nu, gamma_nr, gamma_eu, gamma_er = theta


        node_labels = label
        prev_edges = self.edge_members[0:e_prime_index]
        prev_nodes = list(range(self.last_added[e_prime_index - 1] + 1))
        #NOTE: f = e, e = eprime in the following code

        e = self.edge_members[e_prime_index]


        # find all possible f edges that can generate e (assume uniform random for this)
        canidate_f_indexes = []
        for i in range(len(prev_edges)):
            if len(set(prev_edges[i]).intersection(set(e))) != 0:
                canidate_f_indexes.append(i)

        # find all possible u within each f that can generate e... calc prob
        log_prob_sum = 0
        for f in canidate_f_indexes:
            log_prob_sum += self.expected_log_likelihood_given_f(e_prime_index, f, theta, label) / len(canidate_f_indexes)
       
        return log_prob_sum
    
    def expected_log_likelihood_given_f(self, e_prime_index, f, theta, label):
        node_labels = label
        prev_edges = self.edge_members[0:e_prime_index]

        p, q, gamma_nu, gamma_nr, gamma_eu, gamma_er = theta
        e = self.edge_members[e_prime_index] 


        log_prob_sum = 0
        canidates_u = []
        for potiential_u in e:
            if potiential_u in prev_edges[f]:
                canidates_u.append(potiential_u)

        if (len(canidates_u) == 0):
            return np.log(0)
        u_1_label_count = 0
        u_1_index = None
        u_0_label_count = 0
        u_0_index = None

        for u in canidates_u:
            if (node_labels[u] == 1):
                u_1_label_count += 1
                u_1_index = u
            else:
                u_0_label_count += 1
                u_0_index = u

        if u_1_index != None:
            log_prob_sum += self.log_likelihood_given_u_f(f, u_1_index, e_prime_index, theta, label) / len(canidates_u) * u_1_label_count
        
        if u_0_index != None:
            log_prob_sum += self.log_likelihood_given_u_f(f, u_0_index, e_prime_index, theta, label) / len(canidates_u) * u_0_label_count

        return log_prob_sum

   

    def greedy_expectation_step_given_f(self, v_index, f_index, e_index, theta, label):
        # Note: label is before the change, v_index is the index of the node that will have its label flipped
        node_labels = label
        prev_nodes = list(range(self.last_added[e_index - 1] + 1))
        p, q, gamma_nu, gamma_nr, gamma_eu, gamma_er = theta
        zero_to_one = False
        # determine if 0->1 or 1->0 change
        if node_labels[v_index] == 0:
            # 0->1 change
            zero_to_one = True
        
        # determine case node falls into and calculate appropriate expectation change
        e = self.edge_members[e_index]
        f = self.edge_members[f_index]
        intersect = e.intersection(f)

        if len(intersect) == 0:
            return np.log(0)

        intersect_labels = [node_labels[node] for node in intersect]

        # before changing the label!
        prob_u_1_label = sum(intersect_labels) / len(intersect_labels)
        prob_u_0_label = (len(intersect_labels) - sum(intersect_labels)) / len(intersect_labels)

        diff_in_expectation = 0
        # case 1+6
        # legit case to just calc from scratch here...
        if v_index in intersect:
            novel_nodes = set(e) - set(prev_nodes)
            novel_labels = [node_labels[node] for node in novel_nodes]

            e_labels = [node_labels[node] for node in e]
            f_labels = [node_labels[node] for node in f]
            int_labels = [node_labels[node] for node in intersect]

            Sef1 = sum(int_labels)
            Sef0 = len(int_labels) - sum(int_labels)


            f1 = sum(f_labels)
            f0 = len(f_labels) - sum(f_labels)

            e1 = sum(e_labels)
            e0 = len(e_labels) - sum(e_labels)

            Snte1 = sum(novel_labels)
            Snte0 = len(novel_labels) - sum(novel_labels)


            # external nodes in e (added via steps 5, 6)
            external_nodes = set(prev_nodes) - set(f)
            external_node_labels = [node_labels[node] for node in external_nodes]  

            
            external_nodes_in_e = external_nodes.intersection(e)
            external_nodes_in_e_labels = [node_labels[node] for node in external_nodes_in_e] 

            Stnfe1 = sum(external_nodes_in_e_labels)
            
            Stnfe0 = len(external_nodes_in_e_labels) - sum(external_nodes_in_e_labels)

            # nodes not added in step 5, 6, but could be choosen
            external_nodes_not_in_e = external_nodes - (e)
            external_nodes_not_in_e_labels = [node_labels[node] for node in external_nodes_not_in_e]  

            Stnfne1 = sum(external_nodes_not_in_e_labels)
            Stnfne0 = len(external_nodes_not_in_e_labels) - sum(external_nodes_not_in_e_labels)


            Stnf1 = sum(external_node_labels)
            Stnf0 = len(external_node_labels) - sum(external_node_labels) 

            # calc L1, L0
            L1 = 0
            L1 += Sef1*math.log(p/(1-p)) + f1*math.log(1-p) - math.log(p)
            L1 += Sef0*math.log(q/(1-q)) + f0*math.log(1-q)

            L1 += Stnfe1*math.log(gamma_eu) - gamma_eu + self.gammaln_fast(Stnfne1+1)
            L1 += Stnfe0*math.log(gamma_er) - gamma_er + self.gammaln_fast(Stnfne0+1)
            L1 += -1*self.gammaln_fast(Stnf1+1) + -1*self.gammaln_fast(Stnf0+1)

            L1 += (Snte1*math.log(gamma_nu) - gamma_nu - self.gammaln_fast(Snte1+1))
            L1 += (Snte0*math.log(gamma_nr) - gamma_nr - self.gammaln_fast(Snte0+1))

            L0 = 0
            L0 += Sef0*math.log(p/(1-p)) + f0*math.log(1-p) - math.log(p)
            L0 += Sef1*math.log(q/(1-q)) + f1*math.log(1-q)

            L0 += Stnfe0*math.log(gamma_eu) - gamma_eu + self.gammaln_fast(Stnfne0+1)
            L0 += Stnfe1*math.log(gamma_er) - gamma_er + self.gammaln_fast(Stnfne1+1)

            L0 += -1*self.gammaln_fast(Stnf1+1) + -1*self.gammaln_fast(Stnf0+1)

            L0 += (Snte0*math.log(gamma_nu) - gamma_nu - self.gammaln_fast(Snte0+1))
            L0 += (Snte1*math.log(gamma_nr) - gamma_nr - self.gammaln_fast(Snte1+1))
            if zero_to_one:
                NT1 = np.log(1-p) - np.log(1-q) + np.log(p/(1-p)) - np.log(q/(1-q))
                NT0 = np.log(1-q) - np.log(1-p) + np.log(q/(1-q)) - np.log(p/(1-p)) 
                prob_v_given_f = 1/len(intersect_labels)

                diff_in_expectation = prob_u_0_label*NT0 + prob_u_1_label*NT1 - prob_v_given_f*(L0+NT0) + prob_v_given_f*(L1+NT1)
            else:
                NT0 = np.log(1-p) - np.log(1-q) + np.log(p/(1-p)) - np.log(q/(1-q))
                NT1 = np.log(1-q) - np.log(1-p) + np.log(q/(1-q)) - np.log(p/(1-p)) 

                prob_v_given_f = 1/len(intersect_labels)
                diff_in_expectation = prob_u_0_label*NT0 + prob_u_1_label*NT1 + prob_v_given_f*(L0+NT0) - prob_v_given_f*(L1+NT1)

        
        # case 2
        if v_index not in e and v_index in f:
            NT0, NT1 = 0, 0
            if zero_to_one:
                NT0 = np.log(1-q) - np.log(1-p)
                NT1 = np.log(1-p) - np.log(1-q)
            else:
                NT1 = np.log(1-q) - np.log(1-p)
                NT0 = np.log(1-p) - np.log(1-q)
            diff_in_expectation = NT0 * prob_u_0_label + NT1 * prob_u_1_label

        # case 3
        if v_index not in e and v_index not in f and v_index in prev_nodes:
            # calc neccessary cardinalities
            external_nodes = set(prev_nodes) - set(f)
            external_node_labels = [node_labels[node] for node in external_nodes]  

            external_nodes_not_in_e = external_nodes - (e)
            external_nodes_not_in_e_labels = [node_labels[node] for node in external_nodes_not_in_e]  

            Stnfne1 = sum(external_nodes_not_in_e_labels)
            Stnfne0 = len(external_nodes_not_in_e_labels) - sum(external_nodes_not_in_e_labels)
 
            Stnf1 = sum(external_node_labels)
            Stnf0 = len(external_node_labels) - sum(external_node_labels)
            
            NT0, NT1 = 0, 0
            if zero_to_one:
                NT1 = np.log(Stnfne1 + 1) - np.log(Stnfne0) - np.log(Stnf1 + 1) + np.log(Stnf0)
                NT0 = -1*np.log(Stnfne0) + np.log(Stnfne1 + 1) - np.log(Stnf1 + 1) + np.log(Stnf0)
            else:
                NT0 = np.log(Stnfne0 + 1) - np.log(Stnfne1) - np.log(Stnf0 + 1) + np.log(Stnf1)
                NT1 = -1*np.log(Stnfne1) + np.log(Stnfne0 + 1) - np.log(Stnf0 + 1) + np.log(Stnf1)

            diff_in_expectation = NT0 * prob_u_0_label + NT1 * prob_u_1_label
        
        # case 4
        if v_index in e and v_index not in f and v_index in prev_nodes:
            external_nodes = set(prev_nodes) - set(f)
            external_node_labels = [node_labels[node] for node in external_nodes]  

            Stnf1 = sum(external_node_labels)
            Stnf0 = len(external_node_labels) - sum(external_node_labels)

            NT0, NT1 = 0, 0
            if zero_to_one:
                NT1 = np.log(gamma_eu) - np.log(gamma_er) - np.log(Stnf1 + 1) + np.log(Stnf0)
                NT0 = -1*np.log(gamma_eu) + np.log(gamma_er) - np.log(Stnf1 + 1) + np.log(Stnf0)
            else:
                NT0 = np.log(gamma_eu) - np.log(gamma_er) - np.log(Stnf0 + 1) + np.log(Stnf1)
                NT1 = -1*np.log(gamma_eu) + np.log(gamma_er) - np.log(Stnf0 + 1) + np.log(Stnf1)

            diff_in_expectation = NT0 * prob_u_0_label + NT1 * prob_u_1_label
        # case 5
        if v_index in e and v_index not in prev_nodes:
            novel_nodes = set(e) - set(prev_nodes)
            novel_labels = [node_labels[node] for node in novel_nodes]

            Snte1 = sum(novel_labels)
            Snte0 = len(novel_labels) - sum(novel_labels)

            if zero_to_one:
                NT1 = np.log(gamma_nu) - np.log(gamma_nr) - np.log(Snte1 + 1) + np.log(Snte0)
                NT0 = -1*np.log(gamma_nu) + np.log(gamma_nr) + np.log(Snte0) - np.log(Snte1 + 1)
            else:
                NT0 = np.log(gamma_nu) - np.log(gamma_nr) - np.log(Snte0 + 1) + np.log(Snte1)
                NT1 = -1*np.log(gamma_nu) + np.log(gamma_nr) + np.log(Snte1) - np.log(Snte0 + 1) 

            diff_in_expectation = NT0 * prob_u_0_label + NT1 * prob_u_1_label

        return diff_in_expectation
    
    def f_prob_array_given_e(self, e_index, theta, label):
        probs = np.full(e_index, -np.inf)
        
        for f_index in range(e_index):
            if (len(self.get_edges()[f_index].intersection(self.get_edges()[e_index])) != 0):
                # not entirely sure if this is legal
                probs[f_index] = self.expected_log_likelihood_given_f(e_index, f_index, theta, label)
                
            # print("f_index and likelihood: " + str(f_index) + ", " + str(probs[f_index]))
        if e_index != 0:
            probs = ss.softmax(probs)
            return probs

    def f_prob_array_given_e_array(self, e_index, theta, labels):
        # compute nodes copied from canidate f to e as array
        p, q, gamma_nu, gamma_nr, gamma_eu, gamma_er = theta
        e = self.get_edges()[e_index]
        canidate_f_indexes = []
        canidate_fs = []
        for f_index in range(len(self.get_edges()[:e_index])):
            if len(self.get_edges()[f_index].intersection(e)) != 0:
                canidate_f_indexes.append(f_index)
                canidate_fs.append(self.get_edges()[f_index])
        prev_nodes = list(range(self.last_added[e_index - 1] + 1))

        Sef = [f.intersection(e) for f in canidate_fs]
        Sef1 = []
        Sef0 = []
        for intersect in Sef:
            temp = [labels[node] for node in intersect]
            Sef1.append(sum(temp))
            Sef0.append(len(temp) - sum(temp))

        Sef1 = np.array(Sef1)
        Sef0 = np.array(Sef0)

        f0 = []
        f1 = []
        for f in canidate_fs:
            temp = [labels[node] for node in f]
            f1.append(sum(temp))
            f0.append(len(temp) - sum(temp))
        
        f0 = np.array(f0)
        f1 = np.array(f1)

        external_nodes = [set(prev_nodes) - set(f) for f in canidate_fs]
        external_nodes_in_e = [set(external).intersection(e) for external in external_nodes]

        Stnfe1 = []
        Stnfe0 = []
        for intersect in external_nodes_in_e:
            temp = [labels[node] for node in intersect]

            Stnfe1.append(sum(temp))
            Stnfe0.append(len(temp) - sum(temp))

        Stnfe1 = np.array(Stnfe1)
        Stnfe0 = np.array(Stnfe0)

        external_nodes_not_in_e = [external - e for external in external_nodes]

        Stnfne1 = [] 
        Stnfne0 = []
        for intersect in external_nodes_not_in_e:
            temp = [labels[node] for node in intersect]

            Stnfne1.append(sum(temp))
            Stnfne0.append(len(temp) - sum(temp))

        Stnfne1 = np.array(Stnfne1)
        Stnfne0 = np.array(Stnfne0)

        Stnf1 = []
        Stnf0 = []
        for intersect in external_nodes:
            temp = [labels[node] for node in intersect]

            Stnf1.append(sum(temp))
            Stnf0.append(len(temp) - sum(temp))
        
        Stnf1 = np.array(Stnf1)
        Stnf0 = np.array(Stnf0)

        novel_nodes = [set(e) - set(prev_nodes)]
        Snte1 = []
        Snte0 = []
        for intersect in novel_nodes:
            temp = [labels[node] for node in intersect]

            Snte1.append(sum(temp))
            Snte0.append(len(temp) - sum(temp))

        Snte1 = np.array(Snte1)
        Snte0 = np.array(Snte0)

        L1 = 0
        L1 += Sef1*math.log(p/(1-p)) + f1*math.log(1-p) - math.log(p)
        L1 += Sef0*math.log(q/(1-q)) + f0*math.log(1-q)

        L1 += Stnfe1*math.log(gamma_eu) - gamma_eu + self.gammaln_fast(Stnfne1+1)
        L1 += Stnfe0*math.log(gamma_er) - gamma_er + self.gammaln_fast(Stnfne0+1)
        L1 += -1*self.gammaln_fast(Stnf1+1) + -1*self.gammaln_fast(Stnf0+1)

        L1 += (Snte1*math.log(gamma_nu) - gamma_nu - self.gammaln_fast(Snte1+1))
        L1 += (Snte0*math.log(gamma_nr) - gamma_nr - self.gammaln_fast(Snte0+1))

        L0 = 0
        L0 += Sef0*math.log(p/(1-p)) + f0*math.log(1-p) - math.log(p)
        L0 += Sef1*math.log(q/(1-q)) + f1*math.log(1-q)

        L0 += Stnfe0*math.log(gamma_eu) - gamma_eu + self.gammaln_fast(Stnfne0+1)
        L0 += Stnfe1*math.log(gamma_er) - gamma_er + self.gammaln_fast(Stnfne1+1)
        L0 += -1*self.gammaln_fast(Stnf1+1) + -1*self.gammaln_fast(Stnf0+1)

        L0 += (Snte0*math.log(gamma_nu) - gamma_nu - self.gammaln_fast(Snte0+1))
        L0 += (Snte1*math.log(gamma_nr) - gamma_nr - self.gammaln_fast(Snte1+1))

    
        potential_us = [f.intersection(e) for f in canidate_fs]
        # cannot broadcast... weird
        u1 = []
        u0 = []
        for intersect in potential_us:
            temp = [labels[node] for node in intersect]

            u1.append(sum(temp))
            u0.append(len(temp) - sum(temp))

        u1 = np.array(u1)
        u0 = np.array(u0)


        #add these to LLs function
        LLs = (u1*L1 + u0*L0) / (u1+u0)

        prob_array = np.full(e_index, -np.inf)
        
        counter = 0
        for f_index in canidate_f_indexes:
            prob_array[f_index] = LLs[counter]
            counter += 1


        return ss.softmax(prob_array)

    def weighted_expected_log_likelihood(self, e_prime_index, theta, label):
        node_labels = label
        prev_edges = self.edge_members[0:e_prime_index]
        prev_nodes = list(range(self.last_added[e_prime_index - 1] + 1))
        #NOTE: f = e, e = eprime in the following code

        e = self.edge_members[e_prime_index]


        # find all possible f edges that can generate e (assume uniform random for this)
        canidate_f_indexes = []
        for i in range(len(prev_edges)):
            if len(set(prev_edges[i]).intersection(set(e))) != 0:
                canidate_f_indexes.append(i)

        f_probs = self.f_prob_array_given_e(e_prime_index, theta, label)

        log_prob_sum = 0
        for f in canidate_f_indexes:
            log_prob_sum += self.expected_log_likelihood_given_f(e_prime_index, f, theta, label) * f_probs[f]
       
        return log_prob_sum
    
    def total_log_likelihood(self, theta, label):
        # SUM OF EXPECTED NODES, NOT Total
        log_prob = 0
        # all edges but the first one
        for e_index in range(len(self.get_edges())-1):
            log_prob += self.expected_log_likelihood(e_index+1, theta, label)

        return log_prob
    
    def total_weighted_expected_log_likelihood(self, theta, label):
        total_prob = 0
        for e in range(0, len(self.get_edges())-1):
            total_prob += self.weighted_expected_log_likelihood(e+1, theta, label)

        return total_prob
    
    def gammaln_fast(self, ks):
        # log factorial fast...
        if isinstance(ks, np.ndarray):
            return ss.gammaln(ks)
                
        else:
            if ks >= 200:
                return ss.gammaln(ks)
            else:
                return self.stored_gammaln[ks]

    def generate_canidate_f(self, e_index):
        canidate_f = []
        e = self.get_edges()[e_index]
        for f_index in range(e_index):
            if len(self.get_edges()[f_index].intersection(e)) != 0:
                canidate_f.append(f_index)
        
        return canidate_f
    
    def p_e_given_f(self, theta, labels, e_index, f_index):
        e = self.get_edges()[e_index]
        f = self.get_edges()[f_index]

        node_labels = labels
        prev_nodes = list(range(self.last_added[e_index - 1] + 1))
        p, q, gamma_nu, gamma_nr, gamma_eu, gamma_er = theta

        # novel nodes counting
        novel_nodes = set(e) - set(prev_nodes)
        novel_labels = [node_labels[node] for node in novel_nodes]
        nov1 = sum(novel_labels)
        nov0 = len(novel_labels) - nov1

        # external nodes counting
        external_nodes = set(prev_nodes) - set(f)
        external_node_labels = [node_labels[node] for node in external_nodes]  

        external_nodes_added = external_nodes.intersection(e)
        external_nodes_added_labels = [node_labels[node] for node in external_nodes_added] 

        ext1 = sum(external_nodes_added_labels)
        ext0 = len(external_nodes_added_labels) - ext1

        posext1 = sum(external_node_labels)
        posext0 = len(external_node_labels) - posext1

        # copied and not copied nodes counting
        copied_nodes = e.intersection(f)
        copied_nodes_labels = [node_labels[node] for node in copied_nodes]

        cop1 = sum(copied_nodes_labels)
        cop0 = len(copied_nodes_labels) - cop1

        not_copied_nodes = set(f) - set(e)
        not_copied_nodes_labels = [node_labels[node] for node in not_copied_nodes]

        notcop1 = sum(not_copied_nodes_labels)
        notcop0 = len(not_copied_nodes_labels) - notcop1

        # find probs of u being label 1 and 0
        f_labels = [node_labels[node] for node in f]
        prob_u_label_equals_1 = sum(f_labels) / len(f_labels)
        prob_u_label_equals_0 = 1 - prob_u_label_equals_1

        # calculate prob of e given f and u_label = 1
        prob_given_f_u_label_1 = (p**(cop1-1)) * ((1-p)**notcop1) * (q**cop0) * ((1-q)**notcop0)
        prob_given_f_u_label_1 *= (gamma_eu**ext1) * math.exp(-gamma_eu) / math.factorial(ext1) / math.comb(posext1, ext1)
        prob_given_f_u_label_1 *= (gamma_er**ext0) * math.exp(-gamma_er) / math.factorial(ext0) / math.comb(posext0, ext0)
        prob_given_f_u_label_1 *= (gamma_nu**nov1) * math.exp(-gamma_nu) / math.factorial(nov1)
        prob_given_f_u_label_1 *= (gamma_nr**nov0) * math.exp(-gamma_nr) / math.factorial(nov0)

        # calculate prob of e given f and u_label = 0
        prob_given_f_u_label_0 = (p**(cop0-1)) * ((1-p)**notcop0) * (q**cop1) * ((1-q)**notcop1)
        prob_given_f_u_label_0 *= (gamma_eu**ext0) * math.exp(-gamma_eu) / math.factorial(ext0) / math.comb(posext0, ext0)
        prob_given_f_u_label_0 *= (gamma_er**ext1) * math.exp(-gamma_er) / math.factorial(ext1) / math.comb(posext1, ext1)
        prob_given_f_u_label_0 *= (gamma_nu**nov0) * math.exp(-gamma_nu) / math.factorial(nov0)
        prob_given_f_u_label_0 *= (gamma_nr**nov1) * math.exp(-gamma_nr) / math.factorial(nov1)

        # put together, return full expression
        return prob_u_label_equals_1*prob_given_f_u_label_1 + prob_u_label_equals_0*prob_given_f_u_label_0
    
    def total_log_likelihood_complete(self, theta, labels):
        # e_index 0 must be given...

        summation = 0
        for e_index in range(1, len(self.get_edges())):
            # add all edges together
            e_sum = 0
            canidate_f_indexes = self.generate_canidate_f(e_index)
            for f_index in canidate_f_indexes:
                e_sum += self.p_e_given_f(theta, labels, e_index, f_index)
            
            e_sum /= e_index # divide by possible f_choices
            summation += np.log(e_sum)

        return summation
    
    def generate_canidate_f_approx(self, e_index):
        e = self.get_edges()[e_index]
        k = 0
        canidate_f_indexes = []
        for f_index in range(e_index):
            f = self.get_edges()[f_index]
            inter_size = len(e.intersection(f))

            if inter_size == k:
                canidate_f_indexes.append(f_index)
            elif inter_size > k:
                k = inter_size
                canidate_f_indexes = [f_index]
        
        return canidate_f_indexes

    
    def total_log_likelihood_approx(self, theta, labels):
        # e_index 0 must be given...

        summation = 0
        for e_index in range(1, len(self.get_edges())):
            # add all edges together
            e_sum = 0
            canidate_f_indexes = self.generate_canidate_f_approx(e_index)
            for f_index in canidate_f_indexes:
                e_sum += self.p_e_given_f(theta, labels, e_index, f_index)
            
            e_sum /= len(canidate_f_indexes) # divide by possible f_choices
            summation += np.log(e_sum)

        return summation

    def total_log_likelihood_approx_batch(self, theta, labels):
        # e_index 0 must be given...

        summation = 0
        for e_index in np.random.choice(range(1,len(self.get_edges())), size=int((len(self.get_edges()))/2)):
            # add all edges together
            e_sum = 0
            canidate_f_indexes = self.generate_canidate_f_approx(e_index)
            for f_index in canidate_f_indexes:
                e_sum += self.p_e_given_f(theta, labels, e_index, f_index)
            
            e_sum /= len(canidate_f_indexes) # divide by possible f_choices
            summation += np.log(e_sum)

        return summation


        