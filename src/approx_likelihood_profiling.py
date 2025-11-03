import cProfile
import pstats
from poisson_hypergraph import GH
import xgi
from simulated_annealing import SimulatedAnnealingApprox

true_p = .9
true_q = .1
gamma_nu, gamma_nr, gamma_eu, gamma_er = .75, .25, 1, 0.25
timesteps = 20
true_theta = [true_p, true_q, gamma_nu, gamma_nr, gamma_eu, gamma_er]

def generate_graph_26_starting_nodes(true_theta, timesteps):
    true_p, true_q, gamma_nu, gamma_nr, gamma_eu, gamma_er = true_theta
    H = xgi.Hypergraph([[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]])
    H.set_node_attributes({0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0,10:0,11:0,12:0,13:1,14:1,15:1,16:1,17:1,18:1,19:1,20:1,21:1,22:1,23:1,24:1,25:1}, name="label")
    g=GH(H, [0,1], true_p, true_q)
    g.add_hyperedge(timesteps, gamma_nu, gamma_nr, gamma_eu, gamma_er)

    return g
g = generate_graph_26_starting_nodes(true_theta, 20)

SA = SimulatedAnnealingApprox(g, true_theta)

def test():
    for _ in range(10000):
        SA.step()

    

if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    test()  # Call the function or code you want to profile
    profiler.disable()

    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative').print_stats(30)