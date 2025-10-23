import numpy as np
import pandas as pd
import sys 

class ToySimulator:
    
    def __init__(self, edge_list, theta, force_label = None):
        
        self.theta = theta
        self.edge_list = edge_list
        
        self.eta_plus = theta[0]
        self.eta_minus = theta[1]
        self.lambda_plus = theta[2]
        self.lambda_minus = theta[3]
        self.gamma_plus = theta[4]
        self.gamma_minus = theta[5]
        
        self.force_label = force_label
        
    def simulate(self, n_samples):
        
        e_ix = np.random.randint(0, len(self.edge_list), n_samples)
        e = self.edge_list[e_ix[0]]
        
        q = e[0] / (e[0] + e[1])
        
        e_ = [0, 0]
        
        if self.force_label is not None:
            z_u = self.force_label 
        
        else:
            z_u = np.random.choice([0, 1], p = [q, 1-q])
        
        e_[  z_u] += 1 + np.random.binomial(e[z_u] - 1, self.eta_plus)
        e_[1-z_u] +=     np.random.binomial(e[1-z_u],   self.eta_minus)
        
        e_[  z_u] += np.random.poisson(self.lambda_plus  + self.gamma_plus)
        e_[1-z_u] += np.random.poisson(self.lambda_minus + self.gamma_minus)
        
        self.edge_list += [e_]    
    
    def get_joint_counts(self): 
        
        max_1 = max([e[0] for e in self.edge_list])
        max_2 = max([e[1] for e in self.edge_list])
        max_all = max(max_1, max_2)
        
        joint_counts = np.zeros((max_all+1, max_all+1))
        for e in self.edge_list:
            joint_counts[e[0], e[1]] += 1
        
        return joint_counts

if __name__ == "__main__":
    
    job_num = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    
    N_REPS = 1
    N_STEPS = int(1e7)
    k_max = 70
    
    EDGE_LIST = [[5, 5]]
    
    # loop over values of a parameter a
    
    MU_EXACT = []
    MU_APPROX = []
    V_EXACT = []
    V_APPROX = []

    ETA_PLUS = []

    A = np.linspace(0.0, 1.0, 11)

    THETA = [.5+A/2, .5-A/2, A, 1-A, 0.05, 0.05]
    
    DF = pd.DataFrame()
    
    for i in range(len(A)):
    
        theta = [THETA[0][i], THETA[1][i], THETA[2][i], THETA[3][i], THETA[4], THETA[5]]
            
        for ell in range(N_REPS):
            
            #------
            # actual simulation
            #------
            
            big_C = np.zeros((k_max, k_max))
            
            TS = ToySimulator(edge_list = [[5,5]], theta = theta, force_label = None)
            for _ in range(N_STEPS):
                TS.simulate(1)
            C = TS.get_joint_counts()
            
            if C.shape[0] < k_max:
                C = np.pad(C, ((0, k_max - C.shape[0]), (0, k_max - C.shape[1])), mode='constant')
            
            big_C += C
            
            
            #------
            # approximations with no interaction 
            #------
            big_C0 = np.zeros((k_max, k_max))
            TS = ToySimulator(edge_list = [[5,5]], theta = theta, force_label = 0)
            for _ in range(N_STEPS):
                TS.simulate(1)
            C0 = TS.get_joint_counts()
            
            if C0.shape[0] < k_max:
                C0 = np.pad(C0, ((0, k_max - C0.shape[0]), (0, k_max - C0.shape[1])), mode='constant')
            
            big_C0 += C0
            
            
            big_C += big_C.T # symmetrize, now the mixture question is moot
            big_C = big_C / big_C.sum()

            big_C0 += big_C0.T # symmetrize, now the mixture question is moot
            big_C0 = big_C0 / big_C0.sum()

            p_K_0_exact = big_C.sum(axis=1)

            
            # calculate moments
            
            mu_exact = np.sum(p_K_0_exact * np.arange(len(p_K_0_exact)))
            v_exact = np.sum(p_K_0_exact * np.arange(len(p_K_0_exact))**2) - mu_exact**2    
            mu_approx = np.sum(big_C0.sum(axis=1) * np.arange(len(big_C0.sum(axis=1))))
            v_approx = np.sum(big_C0.sum(axis=1) * np.arange(len(big_C0.sum(axis=1)))**2) - mu_approx**2

            d = {
                "a" : A[i],
                "eta_plus" : theta[0],
                "eta_minus" : theta[1],
                "lambda_plus" : theta[2],
                "lambda_minus" : theta[3],
                "gamma_plus" : theta[4],
                "gamma_minus" : theta[5],
                "rep" : ell,
                "job" : job_num,
                "mu_exact" : mu_exact,
                "v_exact" : v_exact,
                "mu_approx" : mu_approx,
                "v_approx" : v_approx,
            }
            
            df = pd.DataFrame([d])
            DF = pd.concat([DF, df], ignore_index=True)
            
    # save the results
    DF.to_csv(f"throughput/edge-sizes/toy_edge_size_simulator_results_{job_num}.csv", index=False)
        