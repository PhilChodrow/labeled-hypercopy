try:
    from poisson_hypergraph import GH
except:
    from src.poisson_hypergraph import GH
import numpy as np
import math

class FEPair:
    def __init__(self, g, e_index, f_index, weight, node_labels, theta):
        self.g = g
        self.e_index = e_index
        self.f_index = f_index
        self.weight = weight
        self.node_labels = node_labels
        self.theta = theta
        
        # init node positions, node counts
        self.novel_nodes = None
        self.nov1 = None
        self.nov0 = None

        self.external_nodes = None
        self.posext1 = None
        self.posext0 = None

        self.external_nodes_added = None
        self.ext1 = None
        self.ext0 = None

        self.copied_nodes = None
        self.cop1 = None
        self.cop0 = None

        self.not_copied_nodes = None
        self.notcop1 = None
        self.notcop0 = None

        self.possible_u_nodes = None
        self.possible_u_nodes_count_1 = None
        self.possible_u_nodes_count_0 = None

        self.prob_given_u_1 = None
        self.prob_given_u_0 = None
        self.prob_u_label_equals_1 = None
        self.prob_u_label_equals_0 = None

        self.initialize_node_positions_counts_prob()

        # self.past_labels = None

        # TODO: convert the counts into an np array??
        



    # get positions of each node in the f, e pair... i.e., external, copied, etc.
    def initialize_node_positions_counts_prob(self):
        p, q, gamma_nu, gamma_nr, gamma_eu, gamma_er = self.theta
        
        # initialize positions
        e = self.g.get_edges()[self.e_index]
        f = self.g.get_edges()[self.f_index]

        prev_nodes = list(range(self.g.last_added[self.e_index - 1] + 1))

        # novel nodes counting
        self.novel_nodes = set(e) - set(prev_nodes)

        # external nodes counting
        self.external_nodes = set(prev_nodes) - set(f)
        self.external_nodes_added = self.external_nodes.intersection(e)

        # copied and not copied nodes counting
        self.copied_nodes = e.intersection(f)
        self.not_copied_nodes = set(f) - set(e)

        # TODO
        # can make this copied nodes later
        self.possible_u_nodes = f.intersection(e)

        # initialize counts
        # novel nodes counting
        novel_labels = [self.node_labels[node] for node in self.novel_nodes]
        self.nov1 = sum(novel_labels)
        self.nov0 = len(novel_labels) - self.nov1

        # external nodes counting
        external_node_labels = [self.node_labels[node] for node in self.external_nodes]  
        external_nodes_added_labels = [self.node_labels[node] for node in self.external_nodes_added] 

        self.ext1 = sum(external_nodes_added_labels)
        self.ext0 = len(external_nodes_added_labels) - self.ext1

        self.posext1 = sum(external_node_labels)
        self.posext0 = len(external_node_labels) - self.posext1

        # copied and not copied nodes counting
        copied_nodes_labels = [self.node_labels[node] for node in self.copied_nodes]

        self.cop1 = sum(copied_nodes_labels)
        self.cop0 = len(copied_nodes_labels) - self.cop1

        not_copied_nodes_labels = [self.node_labels[node] for node in self.not_copied_nodes]

        self.notcop1 = sum(not_copied_nodes_labels)
        self.notcop0 = len(not_copied_nodes_labels) - self.notcop1

        # find probs of u being label 1 and 0
        possible_u_labels = [self.node_labels[node] for node in self.possible_u_nodes]

        if (len(possible_u_labels) == 0):
            self.prob_u_label_equals_0 = 0
            self.prob_u_label_equals_1 = 0
        else:
            self.prob_u_label_equals_1 = sum(possible_u_labels) / len(possible_u_labels)
            self.prob_u_label_equals_0 = 1 - self.prob_u_label_equals_1

        prob_given_f_u_label_1 = (p**(self.cop1-1)) * ((1-p)**self.notcop1) * (q**self.cop0) * ((1-q)**self.notcop0)
        prob_given_f_u_label_1 *= (gamma_eu**self.ext1) * math.exp(-gamma_eu) / math.factorial(self.ext1) / math.comb(self.posext1, self.ext1)
        prob_given_f_u_label_1 *= (gamma_er**self.ext0) * math.exp(-gamma_er) / math.factorial(self.ext0) / math.comb(self.posext0, self.ext0)
        prob_given_f_u_label_1 *= (gamma_nu**self.nov1) * math.exp(-gamma_nu) / math.factorial(self.nov1)
        prob_given_f_u_label_1 *= (gamma_nr**self.nov0) * math.exp(-gamma_nr) / math.factorial(self.nov0)

        self.prob_given_u_1 = prob_given_f_u_label_1

        # calculate prob of e given f and u_label = 0
        prob_given_f_u_label_0 = (p**(self.cop0-1)) * ((1-p)**self.notcop0) * (q**self.cop1) * ((1-q)**self.notcop1)
        prob_given_f_u_label_0 *= (gamma_eu**self.ext0) * math.exp(-gamma_eu) / math.factorial(self.ext0) / math.comb(self.posext0, self.ext0)
        prob_given_f_u_label_0 *= (gamma_er**self.ext1) * math.exp(-gamma_er) / math.factorial(self.ext1) / math.comb(self.posext1, self.ext1)
        prob_given_f_u_label_0 *= (gamma_nu**self.nov0) * math.exp(-gamma_nu) / math.factorial(self.nov0)
        prob_given_f_u_label_0 *= (gamma_nr**self.nov1) * math.exp(-gamma_nr) / math.factorial(self.nov1)

        self.prob_given_u_0 = prob_given_f_u_label_0

    
    # update and return likelihood when the labels are changed
    def calculate_prob(self, theta, node_labels):
        p, q, gamma_nu, gamma_nr, gamma_eu, gamma_er = theta

        # novel nodes counting
        novel_labels = [node_labels[node] for node in self.novel_nodes]
        nov1 = sum(novel_labels)
        nov0 = len(novel_labels) - nov1

        # external nodes counting
        external_node_labels = [node_labels[node] for node in self.external_nodes]  
        external_nodes_added_labels = [node_labels[node] for node in self.external_nodes_added] 

        ext1 = sum(external_nodes_added_labels)
        ext0 = len(external_nodes_added_labels) - ext1

        posext1 = sum(external_node_labels)
        posext0 = len(external_node_labels) - posext1

        # copied and not copied nodes counting
        copied_nodes_labels = [node_labels[node] for node in self.copied_nodes]

        cop1 = sum(copied_nodes_labels)
        cop0 = len(copied_nodes_labels) - cop1

        not_copied_nodes_labels = [node_labels[node] for node in self.not_copied_nodes]

        notcop1 = sum(not_copied_nodes_labels)
        notcop0 = len(not_copied_nodes_labels) - notcop1

        # find probs of u being label 1 and 0
        possible_u_labels = [node_labels[node] for node in self.possible_u_nodes]

        prob_u_label_equals_1 = 0
        prob_u_label_equals_0 = 0
        if len(possible_u_labels) != 0:
            prob_u_label_equals_1 = sum(possible_u_labels) / len(possible_u_labels)
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

    def greedy_calculate_prob(self, label_changed_index, label_new_value):
        # NOTE: assumes valid change and does not check... for example, if labels is already what it is attempted to change
        # should only calculate the prob if there was a change, separate function required to actually change the values
        p, q, gamma_nu, gamma_nr, gamma_eu, gamma_er = self.theta

        # check what set the label is in...
        if label_changed_index in self.novel_nodes:
            # print("novel")
            if label_new_value == 1:
                temp_prob_given_u_1 = self.prob_given_u_1 * gamma_nu / gamma_nr * self.nov0 / (self.nov1+1)
                temp_prob_given_u_0 = self.prob_given_u_0 * gamma_nr / gamma_nu * self.nov0 / (self.nov1+1)

                return temp_prob_given_u_1*self.prob_u_label_equals_1 + temp_prob_given_u_0*self.prob_u_label_equals_0
            else:
                temp_prob_given_u_0 = self.prob_given_u_0 * gamma_nu / gamma_nr * self.nov1 / (self.nov0+1)
                temp_prob_given_u_1 = self.prob_given_u_1 * gamma_nr / gamma_nu * self.nov1 / (self.nov0+1)

                return temp_prob_given_u_1*self.prob_u_label_equals_1 + temp_prob_given_u_0*self.prob_u_label_equals_0

        elif label_changed_index in self.external_nodes_added:
            # print("external added")
            if label_new_value == 1:
                temp_prob_given_u_1 = self.prob_given_u_1 * gamma_eu / gamma_er * self.posext0 / (self.posext1+1)
                temp_prob_given_u_0 = self.prob_given_u_0 * gamma_er / gamma_eu * self.posext0 / (self.posext1+1)

                return temp_prob_given_u_1*self.prob_u_label_equals_1 + temp_prob_given_u_0*self.prob_u_label_equals_0

            else:
                temp_prob_given_u_0 = self.prob_given_u_0 * gamma_eu / gamma_er * self.posext1 / (self.posext0+1)
                temp_prob_given_u_1 = self.prob_given_u_1 * gamma_er / gamma_eu * self.posext1 / (self.posext0+1)

                return temp_prob_given_u_1*self.prob_u_label_equals_1 + temp_prob_given_u_0*self.prob_u_label_equals_0

        elif label_changed_index in self.external_nodes:
            # print("external not added")
            if label_new_value == 1:
                temp_prob_given_u_1 = self.prob_given_u_1 * (self.posext1 - self.ext1 + 1) / (self.posext0 - self.ext0) * self.posext0 / (self.posext1 + 1)
                temp_prob_given_u_0 = self.prob_given_u_0 * (self.posext1 - self.ext1 + 1) / (self.posext0 - self.ext0) * self.posext0 / (self.posext1 + 1)

                if (self.posext0-self.ext0 == 0):
                    temp_prob_given_u_1 = self.prob_given_u_1 * (self.posext1 - self.ext1 + 1) * self.posext0 / (self.posext1 + 1)
                    temp_prob_given_u_0 = self.prob_given_u_0 * (self.posext1 - self.ext1 + 1) * self.posext0 / (self.posext1 + 1)

                return temp_prob_given_u_1*self.prob_u_label_equals_1 + temp_prob_given_u_0*self.prob_u_label_equals_0

            else:
                temp_prob_given_u_0 = self.prob_given_u_0 * (self.posext0 - self.ext0 + 1) / (self.posext1 - self.ext1) * self.posext1 / (self.posext0 + 1)
                temp_prob_given_u_1 = self.prob_given_u_1 * (self.posext0 - self.ext0 + 1) / (self.posext1 - self.ext1) * self.posext1 / (self.posext0 + 1)

                if (self.posext1-self.ext1 == 0):
                    temp_prob_given_u_0 = self.prob_given_u_0 * (self.posext0 - self.ext0 + 1) * self.posext1 / (self.posext0 + 1)
                    temp_prob_given_u_1 = self.prob_given_u_1 * (self.posext0 - self.ext0 + 1) * self.posext1 / (self.posext0 + 1)
                
                return temp_prob_given_u_1*self.prob_u_label_equals_1 + temp_prob_given_u_0*self.prob_u_label_equals_0
        
        elif label_changed_index in self.copied_nodes:
            # print("copied")
            # these are possible u nodes
            if label_new_value == 1:
                temp_prob_given_u_1 = self.prob_given_u_1 * p / q
                temp_prob_given_u_0 = self.prob_given_u_0 * q / p

                temp_prob_u_label_equals_1 = self.prob_u_label_equals_1 + 1/(self.cop1 + self.cop0)
                temp_prob_u_label_equals_0 = self.prob_u_label_equals_0 - 1/(self.cop1 + self.cop0)

                return temp_prob_given_u_1*temp_prob_u_label_equals_1 + temp_prob_given_u_0*temp_prob_u_label_equals_0         

            else:
                temp_prob_given_u_0 = self.prob_given_u_0 * p / q
                temp_prob_given_u_1 = self.prob_given_u_1 * q / p

                temp_prob_u_label_equals_1 = self.prob_u_label_equals_1 - 1/(self.cop1 + self.cop0)
                temp_prob_u_label_equals_0 = self.prob_u_label_equals_0 + 1/(self.cop1 + self.cop0)

                return temp_prob_given_u_1*temp_prob_u_label_equals_1 + temp_prob_given_u_0*temp_prob_u_label_equals_0

        elif label_changed_index in self.not_copied_nodes:
            # print("not copied")
            if label_new_value == 1:
                temp_prob_given_u_1 = self.prob_given_u_1 * (1-p) / (1-q)
                temp_prob_given_u_0 = self.prob_given_u_0 * (1-q) / (1-p)

                return temp_prob_given_u_1*self.prob_u_label_equals_1 + temp_prob_given_u_0*self.prob_u_label_equals_0

            else:
                temp_prob_given_u_0 = self.prob_given_u_0 * (1-p) / (1-q)
                temp_prob_given_u_1 = self.prob_given_u_1 * (1-q) / (1-p)

                return temp_prob_given_u_1*self.prob_u_label_equals_1 + temp_prob_given_u_0*self.prob_u_label_equals_0
        else: 
            # print("not considered in the edge")
            return self.prob_given_u_1*self.prob_u_label_equals_1 + self.prob_given_u_0*self.prob_u_label_equals_0
    
    def change_counts_given_label_changed(self, label_changed_index, label_new_value):
        # todo add changes to likelihood terms as well
        # perform corresponding greedy change

        # TODO: save last considered move to avoid computing twice...
        p, q, gamma_nu, gamma_nr, gamma_eu, gamma_er = self.theta

        # check what set the label is in...
        if label_changed_index in self.novel_nodes:
            # print("novel")
            if label_new_value == 1:
                self.prob_given_u_1 *= gamma_nu / gamma_nr * self.nov0 / (self.nov1+1)
                self.prob_given_u_0 *= gamma_nr / gamma_nu * self.nov0 / (self.nov1+1)

                self.nov1 += 1
                self.nov0 -= 1
            else:
                self.prob_given_u_0 *= gamma_nu / gamma_nr * self.nov1 / (self.nov0+1)
                self.prob_given_u_1 *= gamma_nr / gamma_nu * self.nov1 / (self.nov0+1)

                self.nov0 += 1
                self.nov1 -= 1

        elif label_changed_index in self.external_nodes_added:
            # print("external added")
            if label_new_value == 1:
                self.prob_given_u_1 *= gamma_eu / gamma_er * self.posext0 / (self.posext1+1)
                self.prob_given_u_0 *= gamma_er / gamma_eu * self.posext0 / (self.posext1+1)

                self.posext1 += 1
                self.posext0 -= 1
                self.ext1 += 1
                self.ext0 -= 1

            else:
                self.prob_given_u_0 *= gamma_eu / gamma_er * self.posext1 / (self.posext0+1)
                self.prob_given_u_1 *= gamma_er / gamma_eu * self.posext1 / (self.posext0+1)

                self.posext0 += 1
                self.posext1 -= 1
                self.ext0 += 1
                self.ext1 -= 1

        elif label_changed_index in self.external_nodes:
            # print("external not added")
            if label_new_value == 1:
                self.prob_given_u_1 = self.prob_given_u_1 * (self.posext1 - self.ext1 + 1) / (self.posext0 - self.ext0) * self.posext0 / (self.posext1 + 1)
                self.prob_given_u_0 = self.prob_given_u_0 * (self.posext1 - self.ext1 + 1) / (self.posext0 - self.ext0) * self.posext0 / (self.posext1 + 1)

                if (self.posext0-self.ext0 == 0):
                    self.prob_given_u_1 = self.prob_given_u_1 * (self.posext1 - self.ext1 + 1) * self.posext0 / (self.posext1 + 1)
                    self.prob_given_u_0 = self.prob_given_u_0 * (self.posext1 - self.ext1 + 1) * self.posext0 / (self.posext1 + 1)

                self.posext1 += 1
                self.posext0 -= 1

            else:
                self.prob_given_u_0 *= (self.posext0 - self.ext0 + 1) / (self.posext1 - self.ext1) * self.posext1 / (self.posext0 + 1)
                self.prob_given_u_1 *= (self.posext0 - self.ext0 + 1) / (self.posext1 - self.ext1) * self.posext1 / (self.posext0 + 1)

                if (self.posext1-self.ext1 == 0):
                    self.prob_given_u_0 = self.prob_given_u_0 * (self.posext0 - self.ext0 + 1) * self.posext1 / (self.posext0 + 1)
                    self.prob_given_u_1 = self.prob_given_u_1 * (self.posext0 - self.ext0 + 1) * self.posext1 / (self.posext0 + 1)
                
                self.posext0 += 1
                self.posext1 -= 1
        
        elif label_changed_index in self.copied_nodes:
            # print("copied")
            # these are possible u nodes
            if label_new_value == 1:
                self.prob_given_u_1 *= p / q
                self.prob_given_u_0 *= q / p

                self.prob_u_label_equals_1 += 1/(self.cop1 + self.cop0)
                self.prob_u_label_equals_0 -= 1/(self.cop1 + self.cop0)

                self.cop1 += 1
                self.cop0 -= 1

            else:
                self.prob_given_u_0 *= p / q
                self.prob_given_u_1 *= q / p

                self.prob_u_label_equals_1 -= 1/(self.cop1 + self.cop0)
                self.prob_u_label_equals_0 += 1/(self.cop1 + self.cop0)

                self.cop0 += 1
                self.cop1 -= 1

        elif label_changed_index in self.not_copied_nodes:
            # print("not copied")
            if label_new_value == 1:
                self.prob_given_u_1 *= (1-p) / (1-q)
                self.prob_given_u_0 *= (1-q) / (1-p)

                self.notcop1 += 1
                self.notcop0 -= 1

            else:
                self.prob_given_u_0 *= (1-p) / (1-q)
                self.prob_given_u_1 *= (1-q) / (1-p)

                self.notcop0 += 1
                self.notcop1 -= 1
        else: 
            # print("not considered in the edge")
            # return self.prob_given_u_1*self.prob_u_label_equals_1 + self.prob_given_u_0*self.prob_u_label_equals_0
            pass
        
        # error checking
        if self.nov1 < 0 or self.nov0 < 0 or self.ext1 < 0 or self.ext0 < 0 or self.posext1 < 0 or self.posext0 < 0:
            print("bad")



