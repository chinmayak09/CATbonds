## 3D plot of the CAT bond price, for the bond paying only coupons, for the Burr claim amounts and a non-homogeneous Poisson process governing the flow of losses.


import numpy as np
from scipy.stats import expon, gamma, norm, weibull_min, pareto
from scipy.stats import poisson

def burrrnd(alpha, lambd, tau, n=1, m=1):
    u = np.zeros((n, m))
    for i in range(m):
        u[:, i] = (lambd * (np.random.uniform(0, 1, n)**(-1/alpha) - 1))**(1/tau)
    return u

def mixexprnd(p=0.5, beta1=1, beta2=2, n=1, m=1):
    y = expon.rvs(scale=1/beta2, size=n*m)
    aux = np.random.uniform(0, 1, n*m) <= p
    y[aux] = expon.rvs(scale=1/beta1, size=np.sum(aux))
    y = y.reshape((n, m))
    return y

def simHPP(lambd, T, N):
    if lambd <= 0 or not np.isscalar(lambd):
        raise ValueError("simHPP: Lambda must be a positive scalar.")
    if T <= 0 or not np.isscalar(T):
        raise ValueError("simHPP: T must be a positive scalar.")
    if N <= 0 or not np.isscalar(N):
        raise ValueError("simHPP: N must be a positive scalar.")
    
    EN = poisson.rvs(mu=lambd * T, size=N)
    rpp = 2 * max(EN) + 2
    ym = np.full((rpp, N), T)
    y = np.zeros((rpp, N, 2))
    y[:, :, 0] = ym
    y[:, :, 1] = np.tile(EN, (rpp, 1))
    
    for i in range(N):
        if EN[i] > 0:
            ttmp = np.sort(T * np.random.uniform(size=EN[i]))
            y[1:(2 * EN[i] + 1), i, 0] = np.concatenate(([0], np.repeat(ttmp, 2)))
        y[1:(2 * EN[i] + 2), i, 1] = np.concatenate(([0], np.repeat(np.arange(EN[i]), 2), [EN[i]]))
    
    return y

def simNHPP(lambd, parlambd, T, N):
    a, b = parlambd[0], parlambd[1]
    if lambd == 0:
        d = parlambd[2]
        JM = simHPP(a + b, T, N)
    elif lambd == 1:
        JM = simHPP(a + b * T, T, N)
    elif lambd == 2:
        d = parlambd[2]
        JM = simHPP(a + b * T, T, N)
    
    rjm, cpp = JM.shape[0], JM.shape[1]
    yy = np.zeros((rjm, cpp, 2))
    yy[:, :, 0] = np.full((rjm, cpp), T)
    
    maxEN = 0
    for i in range(N):
        pom = JM[:, i, 0][JM[:, i, 0] < T]
        pom = pom[1::2]
        R = np.random.uniform(size=len(pom))
        
        if lambd == 0:
            lambdat = (a + b * np.sin(2 * np.pi * (pom + d))) / (a + b)
        elif lambd == 1:
            lambdat = (a + b * pom) / (a + b * T)
        elif lambd == 2:
            lambdat = (a + b * np.sin(2 * np.pi * (pom + d))**2) / (a + b)
        
        pom = pom[R < lambdat]
        EN = len(pom)
        maxEN = max(maxEN, EN)
        
        yy[1:(2 * EN + 1), i, 0] = np.concatenate(([0], np.repeat(pom, 2)))
        yy[2:(2 * EN), i, 1] = np.floor(np.arange(1, 2 * EN) / 2)
        yy[(2 * EN + 1):, i, 1] = EN
    
    return yy[:2 * maxEN + 2]

def paretornd(alpha, lambd, n=1, m=1):
    u = np.zeros((n, m))
    for i in range(m):
        u[:, i] = lambd * (np.random.uniform(0, 1, n)**(-1/alpha) - 1)
    return u

def simNHPPALP(lambd, parlambd, distrib, params, T, N):
    if lambd not in [0, 1, 2]:
        raise ValueError("simNHPPALP: Lambda must be either 0, 1 or 2.")
    if T <= 0 or not np.isscalar(T):
        raise ValueError("simNHPPALP: T must be a positive scalar.")
    if N <= 0 or not np.isscalar(N):
        raise ValueError("simNHPPALP: N must be a positive scalar.")
    
    if lambd in [0, 2] and len(parlambd) != 3:
        raise ValueError("simNHPPALP: for lambda 0 or 2, parlambd must be a 3 x 1 vector.")
    if lambd == 1 and len(parlambd) != 2:
        raise ValueError("simNHPPALP: for lambda 1, parlambd must be a 2 x 1 vector.")
    
    if distrib in ["Burr", "mixofexps"] and len(params) != 3:
        raise ValueError("simNHPPALP: for Burr and mixofexps distributions, params must be a 3 x 1 vector.")
    if distrib in ["gamma", "lognormal", "Pareto", "Weibull"] and len(params) != 2:
        raise ValueError("simNHPPALP: for gamma, lognormal, Pareto and Weibull distributions, params must be a 2 x 1 vector.")
    if distrib == "exponential" and len(params) != 1:
        raise ValueError("simNHPPALP: for exponential distribution, params must be a scalar.")
    
    if distrib not in ["exponential", "gamma", "mixofexps", "Weibull", "lognormal", "Pareto", "Burr"]:
        raise ValueError("simNHPPALP: distrib should be: exponential, gamma, mixofexps, Weibull, lognormal, Pareto or Burr")
    
    poisproc = simNHPP(lambd, parlambd, T, N)
    rpp, cpp = poisproc.shape[0], poisproc.shape[1]
    losses = np.zeros((rpp, cpp))
    
    def get_aux(poisproc, i, T):
        if N == 1:
            return min(np.where(poisproc[:, 0] == T)[0])
        else:
            return min(np.where(poisproc[:, i, 0] == T)[0])
    
    if distrib == "Burr":
        for i in range(N):
            aux = get_aux(poisproc, i, T)
            if aux > 2:
                laux = np.cumsum(burrrnd(params[0], params[1], params[2], aux//2 - 1, 1))
                losses[3:aux, i] = laux[np.ceil((np.arange(aux - 2) / 2)).astype(int) - 1]
                if aux < rpp:
                    losses[aux:, i] = laux[-1]
            else:
                losses[:, i] = 0
    
    elif distrib == "exponential":
        for i in range(N):
            aux = get_aux(poisproc, i, T)
            if aux > 2:
                laux = np.cumsum(expon.rvs(scale=params[0], size=aux//2 - 1))
                losses[3:aux, i] = laux[np.ceil((np.arange(aux - 2) / 2)).astype(int) - 1]
                if aux < rpp:
                    losses[aux:, i] = laux[-1]
            else:
                losses[:, i] = 0
    
    elif distrib == "gamma":
        for i in range(N):
            aux = get_aux(poisproc, i, T)
            if aux > 2:
                laux = np.cumsum(gamma.rvs(a=params[0], scale=1/params[1], size=aux//2 - 1))
                losses[3:aux, i] = laux[np.ceil((np.arange(aux - 2) / 2)).astype(int) - 1]
                if aux < rpp:
                    losses[aux:, i] = laux[-1]
            else:
                losses[:, i] = 0
    
    elif distrib == "lognormal":
        for i in range(N):
            aux = get_aux(poisproc, i, T)
            if aux > 2:
                laux = np.cumsum(norm.rvs(loc=params[0], scale=params[1], size=aux//2 - 1))
                losses[3:aux, i] = laux[np.ceil((np.arange(aux - 2) / 2)).astype(int) - 1]
                if aux < rpp:
                    losses[aux:, i] = laux[-1]
            else:
                losses[:, i] = 0
    
    elif distrib == "mixofexps":
        for i in range(N):
            aux = get_aux(poisproc, i, T)
            if aux > 2:
                laux = np.cumsum(mixexprnd(params[0], params[1], params[2], aux//2 - 1, 1))
                losses[3:aux, i] = laux[np.ceil((np.arange(aux - 2) / 2)).astype(int) - 1]
                if aux < rpp:
                    losses[aux:, i] = laux[-1]
            else:
                losses[:, i] = 0
    
    elif distrib == "Pareto":
        for i in range(N):
            aux = get_aux(poisproc, i, T)
            if aux > 2:
                laux = np.cumsum(paretornd(params[0], params[1], aux//2 - 1, 1))
                losses[3:aux, i] = laux[np.ceil((np.arange(aux - 2) / 2)).astype(int) - 1]
                if aux < rpp:
                    losses[aux:, i] = laux[-1]
            else:
                losses[:, i] = 0
    
    elif distrib == "Weibull":
        for i in range(N):
            aux = get_aux(poisproc, i, T)
            if aux > 2:
                laux = np.cumsum(weibull_min.rvs(c=params[0], scale=params[1], size=aux//2 - 1))
                losses[3:aux, i] = laux[np.ceil((np.arange(aux - 2) / 2)).astype(int) - 1]
                if aux < rpp:
                    losses[aux:, i] = laux[-1]
            else:
                losses[:, i] = 0
    
    out = np.dstack((poisproc, losses))
    return out


## Main
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Function placeholder
def bond_zero_coupon(Z, D, T, r, lambda_type, parlambda, distr, params, Tmax, N):
    # This function needs to be implemented or replaced with the appropriate computation.
    # For now, we return a dummy array with the right shape.
    return np.random.rand(len(D) * len(T), 3)

# Parameters
parlambda = [35.32, 2.32 * 2 * np.pi, -0.2]  # NHPP1
distr = "Burr"  # 'lognormal'
params = [0.4801, 3.9495 * 1e16, 2.1524]  # Burr

# Load data
data = pd.read_table("ncl.dat", header=None)
A = np.mean(data.iloc[:, 2]) * (34.2 / 4)

# Define grid parameters
na = 41  # default 41
D = np.linspace(A, 12 * A, na)
B = 0.25
nb = 41  # default 41
T = np.linspace(B, 8 * B, nb)
Tmax = np.max(T)
lambda_type = 0
N = 1000  # default 1000
r = np.log(1.025)
Z = 1.06

# Compute bond zero coupon values
d1 = bond_zero_coupon(Z, D, T, r, lambda_type, parlambda, distr, params, Tmax, N)
y = d1[:, 0]
x = d1[:, 1] / 1e9
z = d1[:, 2]

# Create a DataFrame for plotting
data_plot = pd.DataFrame({'x': x, 'y': y, 'z': z})

# Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(data_plot['x'], data_plot['y'], data_plot['z'], cmap='viridis')

# Labels and formatting
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('Zero-Coupon CAT Bond Price')

plt.show()
