import numpy as np
import pandas as pd


# Define parameters
lmbda1 = 0  # intensity
parlambda1 = [35.32, 2.32 * 2 * np.pi, -0.2]  # parameters of intensity function
params1 = [18.3806, 1.1052]  # parameters of the lognormal distribution
T1 = 10  # time
N1 = 1  # trajectories
N2 = 100  # Number of Monte Carlo simulations

# Read data from file
data = pd.read_table("ncl.dat", delim_whitespace=True, header=None)

# Process data
t = np.repeat(data.iloc[:, 1], 2)
t1 = np.concatenate(([0], t, [T1]))
PCS = np.cumsum(data.iloc[:, 2]) / 1e+09
PCS1 = np.repeat(PCS, 2)
PCS1 = np.concatenate(([0, 0], PCS1))
z = np.column_stack((t1, PCS1))

# Monte Carlo simulations
mc_simulations = np.zeros((N2, len(t1), 2))
for i in range(N2):
    y1 = sim_nhpp_alp(lmbda1, parlambda1, params1, T1, 1)
    y1[:, 1] /= 1e+09
    mc_simulations[i, :, 0] = t1
    mc_simulations[i, :, 1] = y1[:, 1]

# Calculate mean and quantiles
mc_mean = np.mean(mc_simulations[:, :, 1], axis=0)
mc_quantiles = np.quantile(mc_simulations[:, :, 1], [0.05, 0.95], axis=0)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(t1, z[:, 1], 'g-', label='Historical Data')
plt.plot(t1, mc_mean, 'r--', label='Monte Carlo Mean')
plt.plot(t1, mc_quantiles[0, :], 'b--', label='5th Percentile')
plt.plot(t1, mc_quantiles[1, :], 'b--', label='95th Percentile')
plt.xlabel('Years', fontsize=14)
plt.ylabel('Aggregate Loss Process (USD billion)', fontsize=14)
plt.title('Monte Carlo Simulations', fontsize=16)
plt.legend()
plt.show()
