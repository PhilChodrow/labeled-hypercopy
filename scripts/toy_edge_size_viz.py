import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import os 

all_dfs = os.listdir("throughput/edge-sizes/")
print(all_dfs)

df = pd.DataFrame()
for f in all_dfs:
    if f.endswith(".csv"):
        df = pd.concat([df, pd.read_csv(f"throughput/edge-sizes/{f}")], ignore_index=True)

print(df.shape)

a = df.groupby(["eta_plus", "eta_minus", "lambda_plus", "lambda_minus", "gamma_plus", "gamma_minus"])[["mu_exact", "v_exact", "mu_approx", "v_approx"]].mean().reset_index()


# analytic estimates
EK_0 = (a["lambda_minus"] + a["gamma_minus"]) / (1 - a["eta_minus"])

EK_1 = (1 + a["lambda_plus"] + a["gamma_plus"] - a["eta_plus"]) / (1 - a["eta_plus"])

mu_analytic = (EK_0 + EK_1) / 2

VK_0 = (a["eta_minus"]*EK_0 + a["lambda_minus"] + a["gamma_minus"]) / (1 - a["eta_minus"]**2)
VK_1 = (a["eta_plus"] * (1 - a["eta_plus"])*EK_1 - a["eta_plus"]*(1 - a["eta_plus"]) + a["lambda_plus"] + a["gamma_plus"]) / (1 - a["eta_plus"]**2) 

v_analytic = (VK_0 + VK_1) / 2 + (EK_0 - EK_1)**2 / 4


fig, ax = plt.subplots(1, 2, figsize=(8, 4))    


ax[0].scatter(a["eta_plus"], a["mu_exact"], label="simulation", zorder = 10, color = "cornflowerblue")
ax[0].plot(a["eta_plus"], a["mu_approx"], label="independent approximation", linestyle='--', color = "cornflowerblue")
ax[0].plot(a["eta_plus"], mu_analytic, label="analytic", color = "cornflowerblue")


ax[1].scatter(a["eta_plus"], a["v_exact"], label="simulation", zorder = 10,  color = "firebrick")
ax[1].plot(a["eta_plus"], a["v_approx"], label="independent approximation", linestyle ='--', color = "firebrick")
ax[1].plot(a["eta_plus"], v_analytic, label="analytic", color = "firebrick")


for i in range(2):
    ax[i].semilogy()
    ax[i].grid()
    
ax[0].legend()

ax[0].set(xlabel=r'$\eta_+$', ylabel=r'$\mu$')
ax[1].set(xlabel=r'$\eta_+$', ylabel=r'$V$')

ax[0].set(title="Mean edge size")
ax[1].set(title="Variance of edge size")

plt.tight_layout()
plt.savefig("fig/edge_size_analysis.png")
