import sys 
sys.path.append("src")
sys.path.append("scripts")


import numpy as np 
from itertools import product
from matplotlib import pyplot as plt
from scipy.special import binom
from toy_edge_size_simulator import ToySimulator
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from math import factorial
from linear_map import matrix_of_linear_map




if __name__ == "__main__":
    
    np.random.seed(123)
    
    k_max = 12
    theta = [0.8, 0.2, 0.5, 0.2, 0.4, 0.2]
    
    # FIRST: spectral analysis of the linear operator governing the system dynamics
    
    M = matrix_of_linear_map(k_max, theta)
    
    # column sums should be equal to 1
    
    
    fig, ax = plt.subplots(1, 1, figsize = (6, 6))
    
    ax.imshow(M, cmap = "viridis", interpolation = "nearest", origin = "lower")
    ax.set_title("Matrix of Linear Map")
    
    # time for eigenstuff
    vals, E = np.linalg.eig(M)    

    # retrieve the top eigenvectors 
    idx = vals.argsort()[::-1]
    vals = vals[idx]
    E = E[:, idx]
    
    eigvecs = []
    for i in range(3):
        vi = E[:, i].real
        vi = vi / vi.sum()
        vi = vi / np.abs(vi).sum()
        eigvecs.append(vi.reshape((k_max + 1, k_max + 1)))
    
    
    # SECOND: simulation with the same parameters
    
    n_steps = int(1e5)
    project_every = 1
    
    C = np.zeros((k_max + 1, k_max + 1))
    TS = ToySimulator(edge_list = [[5,5]], theta = theta, force_label = None)

    projections = {i: [] for i in range(3)}
    projection_timesteps = []
    
    # print(Cz)
    # print((eigvecs[0]**2).sum())
    
    for j in range(1, n_steps): 
        TS.simulate(n_samples = 1)
        e = TS.edge_list[-1]
        if e[0] <= k_max and e[1] <= k_max:
            C[e[0], e[1]] += 1
            
        if j % project_every == 0:
            for i in range(3): 
                
                C_ = C / np.sqrt((C**2).sum())
                E_ = eigvecs[i]
                E_norm = E_ / np.sqrt((E_**2).sum())
                         
                projections[i].append((E_norm*C_).sum())
            projection_timesteps.append(j)
            
        
    fig = plt.figure(layout="constrained", figsize = (9.5, 6))
    
    widths = [1, 1, 1, 0.1]
    gs = GridSpec(2, 4, figure=fig, width_ratios=widths)
    
    axes = []
    for i in range(3): 
        axes.append(fig.add_subplot(gs[0, i]))
    
    axes.append(fig.add_subplot(gs[1, 0:2]))
    axes.append(fig.add_subplot(gs[1, 2]))
    
    
    # num_eigvecs = 3
    
    v_min = -0.05
    vmax  = 0.05
    
    # fig, ax = plt.subplots(1, 4, figsize = (18, 6))
    
    colors = ["steelblue", "darkgrey", "lightgrey"]
    
    for i in range(3):
        vi = np.real(E[:, i])
        val = np.real(vals[i])
        vi = vi / vi.sum()
        vi = vi / np.abs(vi).sum()
        P = vi.reshape((k_max + 1, k_max + 1))
        
        im = axes[i].imshow(P, cmap = "BrBG", interpolation = "none", vmin = v_min, vmax = vmax, origin = "lower")
        axes[i].set_title(rf"$v_{i+1}$ ($\lambda_{i+1} = {val:.2f}$)")
        axes[i].set_xlabel(r"$k_0$")
        if i == 0: 
            axes[i].set_ylabel(r"$k_1$")
            
        axes[i].set_xticks(range(0, k_max + 1, 3))
        axes[i].set_yticks(range(0, k_max + 1, 3))
        
        
        axes[3].plot(projection_timesteps, projections[i], label = fr"$v_{i+1}$", color = colors[i])
        axes[3].legend()
        axes[3].semilogx()
        axes[3].set_title("Cosine similarity of  sample edge size\ndistribution with top 3 eigenvectors")
    
    
    
    
    axes[3].plot([1, n_steps], [0, 0], "k--", alpha = 0.5)
    axes[3].set_xlabel("Timestep")
    axes[3].set_ylabel("Cosine similarity")
    
    
    #     fig.colorbar(im, ax = ax[i])
    
    final = axes[4].imshow(C / C.sum(), cmap = "Greys", interpolation = "none", vmin = 0, origin = "lower")
    axes[4].set_title(f"Final simulation state")
    
    axes[4].set_xlabel(r"$k_0$")
    axes[4].set_ylabel(r"$k_1$")
    
    axes[4].set_xticks(range(0, k_max + 1, 3))
    axes[4].set_yticks(range(0, k_max + 1, 3))
    
    cbar_ax = fig.add_subplot(gs[0, 3])
    fig.colorbar(im, cax=cbar_ax)
    
    cbar_ax = fig.add_subplot(gs[1, 3])
    fig.colorbar(final, cax=cbar_ax)
    
    # plt.tight_layout()

    
    # divider = make_axes_locatable(axes[4])
    # cax = divider.append_axes('right', size='5%', pad=0.05)
    # fig.colorbar(final, cax=cax, orientation='vertical');

    
    plt.savefig("fig/linear_map_eigenvectors.png", dpi = 300)