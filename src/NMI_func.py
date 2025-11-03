# implementation of NMI from https://network-science-notes.github.io/chapters/13-modularity-maximization.html#validation-of-community-detection-algorithms
import numpy as np

def labels_to_array(z, G):
    if isinstance(z, dict):
        z = np.array([z[node] for node in G.nodes])
    elif isinstance(z, list):
        z = np.array(z)
    return z

def H(z, g):
    z = labels_to_array(z, g)
    n = len(z)
    p = np.array([np.mean(z==i) for i in np.unique(z)])
    return -np.sum(p*np.log(p))

def I(z, z_, g):
    z = labels_to_array(z, g)
    z_ = labels_to_array(z_, g)

    n = len(z)

    summation = 0
    for i in np.unique(z):
        for j in np.unique(z_):
            p = np.mean((z==i) & (z_==j))
            if p==0:
                continue
            summation += p*np.log(p/(np.mean(z==i)*np.mean(z_==j)))

    return summation

def NMI(z, z_, g):
    return 2*I(z, z_, g)/(H(z, g) + H(z_, g))