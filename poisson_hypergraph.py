import xgi
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
import math

class GH:
    def __init__(self, H, labels, p, q):
        self.H = H
        self.labels = labels
        self.p = p
        self.q = q
        self.edge_members = H.edges.members()
        self.node_labels = H.nodes.attrs("label").aslist()
        self.nodes = list(H.nodes)

    def get_labels(self):
        return(self.node_labels)
    
    def get_edges(self):
        return(self.edge_members)
    
    def set_values(self, value0, value1, u_label):
        if u_label == 0:
            return([value0, value1])
        else: 
            return([value1, value0])
        
    def likelihood(self, e_index, u_index, e_prime_index, lambda_ = 1, u_prob = 1):
        node_labels = self.get_labels()

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
        int_labels = [node_labels[node] for node in intersect]

        # Get edge and intersection sizes
        e_num_u, e_num_r = self.set_values(len(e_labels) - sum(e_labels), sum(e_labels), u_label)
        int_num_u, int_num_r = self.set_values(len(int_labels) - sum(int_labels), sum(int_labels), u_label)

        # Get the external nodes
        prev_edges = self.edge_members[0:e_prime_index]
        prev_nodes = list(range(max(prev_edges[-1]) + 1))

        all_external_nodes = set(prev_nodes) - set(e)
        all_external_labels = [node_labels[node] for node in all_external_nodes]

        external_nodes = set([node for node in prev_nodes if node in e_prime]) - set(e)
        external_labels = [node_labels[node] for node in external_nodes]

        all_ext_num_u, all_ext_num_r = self.set_values(len(all_external_labels) - sum(all_external_labels), sum(all_external_labels), u_label)

        ext_num_u, ext_num_r = self.set_values(len(external_labels) - sum(external_labels), sum(external_labels), u_label)

        # Get the new nodes
        novel_nodes = set(e_prime) - set(prev_nodes)
        novel_labels = [node_labels[node] for node in novel_nodes]

        novel_num_u, novel_num_r = self.set_values(len(novel_labels) - sum(novel_labels), sum(novel_labels), u_label)

        # Probability calculation
        prob_e = 1 / e_prime_index

        prob_u = 1 / len(e)

        P1 = self.p ** (int_num_u - 1)
        P2 = (1 - self.p) ** (e_num_u - int_num_u)
        P3 = self.q ** (int_num_r)
        P4 = (1 - self.q) ** (e_num_r - int_num_r)

        if ext_num_u == 0:
            prob_those_ext_u = 1
        else:
            prob_those_ext_u = ext_num_u / all_ext_num_u
        if ext_num_r == 0:
            prob_those_ext_r = 1
        else: prob_those_ext_r = ext_num_r / all_ext_num_r

        P5_numer = (math.e ** (-lambda_ * u_prob)) * ((lambda_ * u_prob) ** ext_num_u) * prob_those_ext_u * (math.e ** (-lambda_ * (1 - u_prob))) * ((lambda_ * (1 - u_prob)) ** ext_num_r) * prob_those_ext_r
        P5_denom = math.factorial(ext_num_u) * math.factorial(ext_num_r)
        P5 = P5_numer / P5_denom
        P6_numer = (math.e ** (-lambda_ * u_prob)) * ((lambda_ * u_prob) ** novel_num_u) * (math.e ** (-lambda_ * (1 - u_prob))) * ((lambda_ * (1 - u_prob)) ** novel_num_r)
        P6_denom = (math.factorial(novel_num_u) * math.factorial(novel_num_r))
        P6 = P6_numer / P6_denom
        prob_e_prime = P1 * P2 * P3 * P4 * P5 * P6
        
        prob = prob_e * prob_u * prob_e_prime

        return(prob)

    def add_hyperedge(self, num_edges = 1, lambda_ = 1, u_prob = 1):
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
            num_ext = np.random.poisson(lambda_, 1)[0]


            u_indices = list(filter(lambda x: self.node_labels[x] == u_label, self.nodes))
            r_indices = list(filter(lambda x: self.node_labels[x] != u_label, self.nodes))


            for i in range(0, num_ext):
                is_u = np.random.choice([0, 1], p = [1 - u_prob, u_prob])
                if is_u == 0:
                    if len(set(r_indices) - set(e)) > 0:
                        exterior_node = random.sample(list(set(r_indices) - set(e)), 1)[0] # randomly sample a node from outside the existing hyperedge
                        e_prime.append(exterior_node)
                if is_u == 1:
                    if len(set(u_indices) - set(e)) > 0:
                        exterior_node = random.sample(list(set(u_indices) - set(e)), 1)[0] # randomly sample a node from outside the existing hyperedge
                        e_prime.append(exterior_node)

            ## Add new nodes        
            num_new = np.random.poisson(lambda_, 1)[0]

            for i in range(0, num_new):
                new_node = len(self.nodes)
                # self.H.add_node(new_node)
                self.nodes.append(new_node)
                is_u = np.random.choice([0, 1], p = [1 - u_prob, u_prob])
                if is_u == 1:
                    # self.H.set_node_attributes({new_node : u_label}, name = "label")
                    self.node_labels.append(u_label)
                if is_u == 0:
                    # self.H.set_node_attributes({new_node : r_label}, name = "label")
                    self.node_labels.append(r_label)
                e_prime.append(new_node)

            ## Add the edge to the hypergraph
            # self.H.add_edge(e_prime)
            self.edge_members.append(set(e_prime))
        
        # big_H = xgi.Hypergraph(self.edge_members)
        # node_dict = dict(zip(self.nodes, self.node_labels))
        # big_H.set_node_attributes(node_dict, name = "label")
        # return(big_H)