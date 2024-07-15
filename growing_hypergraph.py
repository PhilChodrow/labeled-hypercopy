import xgi
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd

class GH:
    def __init__(self, H, labels, p, q):
        self.H = H
        self.labels = labels
        self.p = p
        self.q = q

    def add_hyperedge(self, ext_method = "standard", new_method = "standard", num_new = 1, num_ext = 1, lambda_u = 1, lambda_r = 1, neighb_prob = 1):
        e_prime = [] # create an empty new edge called e_prime

        ## randomly select an existing hyperedge e to start with
        e_num = random.randint(0, self.H.num_edges - 1)
        e = self.H.edges.members()[e_num]
        e_size = self.H.edges.size.asdict().get(e_num)

        ## randomly select a node u from e
        u_num = random.randint(0, e_size - 1)
        u = list(e)[u_num]
        u_label = self.H.nodes.attrs.asdict().get(u).get("label")
        e_prime.append(u)

        ## add other nodes from e to e_prime
        # add node with same label as u with prob p
        # add node with different label from u with prob q
        for node in e:
            if self.H.nodes.attrs.asdict().get(node).get("label") == u_label:
                prob = self.p
            else:
                prob = self.q
            if random.random() < prob:
                e_prime.append(node)
        
        ## add exterior nodes
        if ext_method == "poisson":
            num_ext_u = np.random.poisson(lambda_u, 1)[0]
            num_ext_r = np.random.poisson(lambda_r, 1)[0]

            labels = self.H.nodes.attrs("label").aslist()
            u_indices = list(filter(lambda x: labels[x] == u_label, self.H.nodes))
            r_indices = list(filter(lambda x: labels[x] != u_label, self.H.nodes))

            for i in range(0, num_ext_u):
                if len(set(u_indices) - set(e)) > 0:
                    exterior_node = random.sample(list(set(u_indices) - set(e)), 1)[0] # randomly sample a node from outside the existing hyperedge
                    e_prime.append(exterior_node)
            
            for i in range(0, num_ext_r):
                if len(set(r_indices) - set(e)) > 0:
                    exterior_node = random.sample(list(set(r_indices) - set(e)), 1)[0] # randomly sample a node from outside the existing hyperedge
                    e_prime.append(exterior_node) 
        
        if ext_method == "poisson nonunif":
            num_ext_u = np.random.poisson(lambda_u, 1)[0]
            num_ext_r = np.random.poisson(lambda_r, 1)[0]

            neighbs = self.H.nodes.neighbors(u_num)

            labels = self.H.nodes.attrs("label").aslist()
            u_indices_not = list(filter(lambda x: labels[x] == u_label, self.H.nodes - neighbs))
            r_indices_not = list(filter(lambda x: labels[x] != u_label, self.H.nodes - neighbs))
            u_indices_neighbs = list(filter(lambda x: labels[x] == u_label, neighbs))
            r_indices_neighbs = list(filter(lambda x: labels[x] != u_label, neighbs))

            for i in range(0, num_ext_u):
                is_neighb = np.random.choice([0, 1], p = [1 - neighb_prob, neighb_prob])
                if is_neighb == 0:
                    if len(set(u_indices_not) - set(e)) > 0:
                        exterior_node = random.sample(list(set(u_indices_not) - set(e)), 1)[0] # randomly sample a node from outside the existing hyperedge
                        e_prime.append(exterior_node)
                if is_neighb == 1:
                    if len(set(u_indices_neighbs) - set(e)) > 0:
                        exterior_node = random.sample(list(set(u_indices_neighbs) - set(e)), 1)[0] # randomly sample a node from outside the existing hyperedge
                        e_prime.append(exterior_node)
            
            for i in range(0, num_ext_r):
                is_neighb = np.random.choice([0, 1], p = [1 - neighb_prob, neighb_prob])
                if is_neighb == 0:
                    if len(set(r_indices_not) - set(e)) > 0:
                        exterior_node = random.sample(list(set(r_indices_not) - set(e)), 1)[0] # randomly sample a node from outside the existing hyperedge
                        e_prime.append(exterior_node)
                if is_neighb == 1:
                    if len(set(r_indices_neighbs) - set(e)) > 0:
                        exterior_node = random.sample(list(set(r_indices_neighbs) - set(e)), 1)[0] # randomly sample a node from outside the existing hyperedge
                        e_prime.append(exterior_node)

        if ext_method == "standard":
            for i in range(0, num_ext):
                if len(list(self.H.nodes - e)) > 0:
                    exterior_node = random.sample(list(self.H.nodes - e), 1)[0] # randomly sample a node from outside the existing hyperedge
                    e_prime.append(exterior_node)
                # else:
                #     print("No remaining exterior nodes, one failed to add")

        ## add new nodes
        if new_method == "standard":
            for i in range(0, num_new):
                new_node = len(self.H.nodes)
                self.H.add_node(new_node)
                new_label = np.random.choice([self.labels[0], self.labels[1]])
                self.H.set_node_attributes({new_node : new_label}, name = "label")
                e_prime.append(new_node)
        if new_method == "majority label":
            for i in range(0, num_new):
                new_node = len(self.H.nodes)
                self.H.add_node(new_node)
                labels_0_count = 0
                labels_1_count = 0
                for j in e:
                    if self.H.nodes.attrs.asdict().get(j).get("label") == self.labels[0]:
                        labels_0_count += 1
                    else:
                        labels_1_count += 1
                if labels_0_count > labels_1_count:
                    new_label = self.labels[0]
                elif labels_0_count < labels_1_count:
                    new_label = self.labels[1]
                else:
                    new_label = np.random.choice([self.labels[0], self.labels[1]])
                self.H.set_node_attributes({new_node : new_label}, name = "label")
                e_prime.append(new_node)
        if new_method == "u label":
            for i in range(0, num_new):
                new_node = len(self.H.nodes)
                self.H.add_node(new_node)
                self.H.set_node_attributes({new_node : u_label}, name = "label")
                e_prime.append(new_node)
        if new_method == "bern edge prop":
            labels_0_count = 0
            labels_1_count = 0
            for j in e:
                if self.H.nodes.attrs.asdict().get(j).get("label") == self.labels[0]:
                    labels_0_count += 1
                else:
                    labels_1_count += 1
            prop = labels_1_count / (labels_0_count + labels_1_count)
            for i in range(0, num_new):
                new_node = len(self.H.nodes)
                self.H.add_node(new_node)
                new_label = np.random.choice([self.labels[0], self.labels[1]], p = [1 - prop, prop])
                self.H.set_node_attributes({new_node : new_label}, name = "label")
                e_prime.append(new_node)
        if new_method == "bern p":
            if self.labels[0] == u_label:
                    r_label = self.labels[1]
            else:
                r_label = self.labels[0]
            for i in range(0, num_new):
                new_node = len(self.H.nodes)
                self.H.add_node(new_node)
                new_label = np.random.choice([u_label, r_label], p = [self.p, 1 - self.p])
                self.H.set_node_attributes({new_node : new_label}, name = "label")
                e_prime.append(new_node)
        if new_method == "poisson":
            num_new_u = np.random.poisson(lambda_u, 1)[0]
            num_new_r = np.random.poisson(lambda_r, 1)[0]
            if self.labels[0] == u_label:
                r_label = self.labels[1]
            else:
                r_label = self.labels[0]

            for i in range(0, num_new_u):
                new_node = len(self.H.nodes)
                self.H.add_node(new_node)
                self.H.set_node_attributes({new_node : u_label}, name = "label")
                e_prime.append(new_node)
            for i in range(0, num_new_r):
                new_node = len(self.H.nodes)
                self.H.add_node(new_node)
                self.H.set_node_attributes({new_node : r_label}, name = "label")
                e_prime.append(new_node)
        self.H.add_edge(e_prime)