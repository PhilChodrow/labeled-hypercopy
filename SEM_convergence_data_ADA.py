import xgi
import sem_expo
import numpy as np
import csv


def main():
    func = sem_expo.sem_functions()
    s_intial = np.array([1, 2, 1, 2, 0.5, 0.5, 0.5, 0.5])
    g1_true_theta = [0.9, 0.1, 0.75, 0.25, 0.75, 0.25]
    g2_true_theta = [0.6, 0.4, 0.75, 0.25, 0.75, 0.25]
    g3_true_theta = [0.1, 0.9, 0.9, 0.1, 0.9, 0.1]
    g4_true_theta = [0.6, 0.4, 0.9, 0.1, 0.9, 0.1]

    print("Graph 1")
    GH1 = func.generate_hypergraph(g1_true_theta, 500)
    estimates1, likelihoods1 = func.SEM(GH1, s_intial, 1600, 0.01, 0.001)
    with open('graph1_sem_ests.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(estimates1)
    np.savetxt('graph1_sem_liks.csv', likelihoods1, delimiter=',') 

    print("Graph 2")
    GH2 = func.generate_hypergraph(g2_true_theta, 500)
    estimates2, likelihoods2 = func.SEM(GH2, s_intial, 1600, 0.01, 0.001)
    with open('graph2_sem_ests.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(estimates2)
    np.savetxt('graph2_sem_liks.csv', likelihoods2, delimiter=',') 

    print("Graph 3")
    GH3 = func.generate_hypergraph(g3_true_theta, 500)
    estimates3, likelihoods3 = func.SEM(GH3, s_intial, 1600, 0.01, 0.001)
    with open('graph3_sem_ests.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(estimates3)
    np.savetxt('graph3_sem_liks.csv', likelihoods3, delimiter=',') 

    print("Graph 4")
    GH4 = func.generate_hypergraph(g4_true_theta, 500)
    estimates4, likelihoods4 = func.SEM(GH4, s_intial, 1600, 0.01, 0.001)
    with open('graph4_sem_ests.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(estimates4)
    np.savetxt('graph4_sem_liks.csv', likelihoods4, delimiter=',') 
    

if __name__ == "__main__":
    main()