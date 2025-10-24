import numpy as np
from itertools import product
from scipy.special import binom
from math import factorial

def linear_operator(k_max, theta):
    
    eta_plus     = theta[0]
    eta_minus    = theta[1]
    lambda_plus  = theta[2]
    lambda_minus = theta[3]
    gamma_plus   = theta[4]
    gamma_minus  = theta[5]
    
    A = np.zeros((k_max+1, k_max+1, k_max+1, k_max+1))
        
    # number of nodes of labels 0 and 1 in originating edge 
    for k0, k1 in product(range(k_max + 1), range(k_max + 1)):
        
        if k0 + k1 == 0:
            continue 
        
        for z_u in [0, 1]:
        
            eta_0, eta_1 = theta[z_u], theta[1 - z_u]
            lambda_0, lambda_1 = theta[2 + z_u], theta[2 + 1 - z_u]
            gamma_0, gamma_1 = theta[4 + z_u], theta[4 + 1 - z_u]
            
            # probability that selected node u has specified label z_u
            q = [k0 / (k0 + k1), k1 / (k0 + k1)][z_u]
            
            # binomial node copying
            for l0, l1 in product(range(k0+1), range(k1+1)):

                if z_u == 0: 
                    pl0 = binomial_pmf(l0, eta_0, k0, condition_1 = True)
                    pl1 = binomial_pmf(l1, eta_1, k1, condition_1 = False)
                    
                else:
                    pl0 = binomial_pmf(l0, eta_0, k0, condition_1 = False)
                    pl1 = binomial_pmf(l1, eta_1, k1, condition_1 = True)
                
                # poisson node additions: extant nodes plus new nodes
                for h0, h1 in product(range(k_max - l0 + 1), range(k_max - l1 + 1)):
                    
                    ph0 = poisson_pmf(h0, lambda_0 + gamma_0)
                    ph1 = poisson_pmf(h1, lambda_1 + gamma_1)
                
                    A[k0, k1, l0 + h0, l1 + h1] += q*pl0*pl1*ph0*ph1
                        
    return A

def binomial_pmf(n_success, prob, n_trials, condition_1 = False):
    if not condition_1: 
        return binom(n_trials, n_success) * (prob**n_success)*((1 - prob)**(n_trials - n_success))
    
    else:
        if n_trials >= 1:
            return binom(n_trials - 1, n_success-1)*(prob**(n_success - 1))*((1 - prob)**(n_trials - n_success))
        else:
            return 0

def poisson_pmf(k, lam):
    return (lam**k)*np.exp(-lam) / factorial(k)
          
def matrix_of_linear_map(k_max, theta):
    
    I = np.eye((k_max + 1)**2)
    M = np.zeros(((k_max + 1)**2, (k_max + 1)**2))
    
    A = linear_operator(k_max, theta)
    
    for i in range((k_max + 1)**2):
        
        u = I[:, i] # basis vector
        T = u.reshape((k_max + 1, k_max + 1)) # reshape to tensor
        
        # apply linear operator to T
        # check on this for troubleshooting
        S = np.einsum("ijkl, ij -> kl", A, T)
        
        v = S.reshape((k_max+1)**2) # reshape back to vector
        M[:, i] = v
        
    return M