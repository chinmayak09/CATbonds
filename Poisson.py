import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Function to simulate a homogeneous Poisson process
def sim_hpp(lmbda, T, N):
    EN = np.random.poisson(lmbda * T, N)
    y = np.tile(T, (2 * np.max(EN) + 2, N))
    yy = np.dstack((y, np.repeat(EN, 2 * np.max(EN) + 2).reshape(2 * np.max(EN) + 2, N)))
    for i in range(N):
        if EN[i] > 0:
            yy[1:(2 * EN[i] + 1), i, 0] = np.concatenate(([0], np.sort(T * np.random.rand(EN[i])).repeat(2)))
        else:
            yy[0, i, 0] = 0
        yy[1:(2 * EN[i] + 2), i, 1] = np.concatenate(([0], np.arange(1, EN[i] + 1).repeat(2), [EN[i]]))
    return yy

# Function to simulate a non-homogeneous Poisson process
def sim_nhpp(lmbda, parlambda, T, N):
    a, b = parlambda[0], parlambda[1]
    if lmbda == 0:
        c = parlambda[2]
        JM = sim_hpp(a + b, T, N)
    elif lmbda == 1:
        JM = sim_hpp(a + b * T, T, N)
    elif lmbda == 3:
        JM = sim_hpp(a + b * T, T, N)
    
    rjm, cpp = JM.shape[:2]
    yy = np.dstack((np.tile(T, (rjm, N)), np.zeros((rjm, N))))
    maxEN = 0
    for i in range(N):
        pom = JM[:, i, 0][JM[:, i, 0] < T]
        pom = pom[::2]
        R = np.random.rand(len(pom))
        if lmbda == 0:
            lambdat = (a + b * np.sin(2 * np.pi * (pom + c))) / (a + b)
        elif lmbda == 1:
            lambdat = (a + b * pom) / (a + b * T)
        elif lmbda == 3:
            lambdat = (a + b * np.sin(2 * np.pi * (pom + c))**2) / (a + b)
        pom = pom[R < lambdat]
        EN = len(pom)
        maxEN = max(maxEN, EN)
        yy[1:(2 * EN + 1), i, 0] = np.concatenate(([0], pom.repeat(2)))
        yy[1:(2 * EN), i, 1] = np.arange(EN)
        yy[(2 * EN + 1):rjm, i, 1] = EN
    yy = yy[:2 * maxEN + 2, :, :]
    return yy

# Function to simulate aggregate loss process driven by NHPP for lognormal distribution
def sim_nhpp_alp(lmbda, parlambda, params, T, N):
    if N == 1:
        poisproc = sim_nhpp(lmbda, parlambda, T, N)[:, :, :, None]
    else:
        poisproc = sim_nhpp(lmbda, parlambda, T, N)
    
    rpp, cpp = poisproc.shape[:2]
    losses = np.zeros((rpp, cpp))
    for i in range(N):
        aux = np.where(poisproc[:, i, 0] == T)[0][0]
        if aux > 1:
            laux = np.cumsum(np.random.lognormal(params[0], params[1], aux // 2 - 1))
            losses[2:aux, i] = np.repeat(laux, 2)
        else:
            if aux < rpp - 1:
                losses[aux + 1:, i] = laux[-1]
        
    if N == 1:
        y = np.dstack((poisproc[:, :, 0], losses))
    else:
        y = np.dstack((poisproc[:, :, 0], losses))
    return y

# Set seed for reproducibility
np.random.seed(2)

# Define parameters
lmbda1 = 0  # intensity
parlambda1 = [35.32, 2.32 * 2 * np.pi, -0.2]  # parameters of intensity function
params1 = [18.3806, 1.1052]  # parameters of the lognormal distribution
T1 = 10  # time
N1 = 1  # trajectories
N2 = 100

# Simulate aggregate loss process
y1 = sim_nhpp_alp(lmbda1, parlambda1, params1, T1, N1)
y1[:, 1] /= 1e+09

# Read data from file
data = pd.read_table("ncl.dat", delim_whitespace=True, header=None)

# Process data
t = np.repeat(data.iloc[:, 1], 2)
t1 = np.concatenate(([0], t, [T1]))
PCS = np.cumsum(data.iloc[:, 2]) / 1e+09
PCS1 = np.repeat(PCS, 2)
PCS1 = np.concatenate(([0, 0], PCS1))
z = np.column_stack((t1, PCS1))

# Calculate mean of aggregate loss process
t2 = np.linspace(0, T1, 1001)
RP = np.exp(params1[0] + params1[1]**2/2) * (parlambda1[0] * t2 - parlambda1[1]/(2*np.pi) * (np.cos(2 * np.pi * (t2 + parlambda1[2])) - np.cos(2 * np.pi * parlambda1[2])))
me = np.column_stack((t2, RP/1e+09))

# Compute quantile lines
def quantile_lines(data, step, perc):
    N = data.shape[1]
    R = data.shape[0]
    begin = data[0, 0, 0]
    end = data[R - 1, 0, 0]
    num_points = int((end - begin) / step) + 1
    vec_step = np.linspace(begin, end, num_points)
    y = np.zeros((num_points, len(perc)))
    for i in range(num_points):
        vec_val = []
        for j in range(N):
            aux1 = data[:, j, 0]
            aux2 = data[:, j, 1]
            pos = np.searchsorted(aux1, vec_step[i], side='right')
            if pos < R:
                vec_val.append(aux2[pos - 1] + (vec_step[i] - aux1[pos - 1]) * (aux2[pos] - aux2[pos - 1]) / (aux1[pos] - aux1[pos - 1]))
            else:
                vec_val.append(aux2[pos - 1])
        y[i, :] = np.quantile(vec_val, perc)
    return np.column_stack((vec_step, y))

# Compute quantile lines
step1 = 0.05  # time
perc1 = [0.05, 0.95]  # quantiles
qq1 = quantile_lines(sim_nhpp, step1, perc1)
