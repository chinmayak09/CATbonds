import numpy as np

# Placeholder for simNHPPALP
def simNHPPALP(lambda_type, parlambda, distr, params, Tmax, N):
    return np.random.rand(100, N, 2)  # Dummy data for testing

def bond_zero_coupon(Z, D, T, r, lambda_type, parlambda, distr, params, Tmax, N):
    if lambda_type not in [0, 1, 2]:
        raise ValueError("BondZeroCoupon: Lambda must be either 0, 1, or 2.")
    if not np.isscalar(Z):
        raise ValueError("BondZeroCoupon: payment at maturity Z needs to be a scalar")
    if not np.isscalar(r):
        raise ValueError("BondZeroCoupon: discount rate needs to be a scalar")
    if D.ndim != 1:
        raise ValueError("BondZeroCoupon: threshold level D needs to be a vector")
    if T.ndim != 1:
        raise ValueError("BondZeroCoupon: time to expiry T needs to be a vector")
    
    x = simNHPPALP(lambda_type, parlambda, distr, params, Tmax, N)
    Tl = len(T)
    Dl = len(D)
    y = np.zeros((Tl * Dl, 3))
    
    wyn = 0
    for i in range(Tl):
        for j in range(Dl):
            for k in range(N):
                traj = np.column_stack((x[:, k, 0], x[:, k, 1]))
                traj_filtered = traj[traj[:, 0] <= T[i]]
                if len(traj_filtered) > 0 and traj_filtered[-1, 1] <= D[j]:
                    wyn += 1
            y[i * Dl + j, 0] = T[i]
            y[i * Dl + j, 1] = D[j]
            y[i * Dl + j, 2] = Z * np.exp(-r * T[i]) * wyn / N
            wyn = 0
    
    return y


Z = 1.06
D = np.linspace(1, 12, 41)  # Example threshold levels
T = np.linspace(0.25, 2, 41)  # Example time to expiry
r = np.log(1.025)
lambda_type = 0
parlambda = [35.32, 2.32 * 2 * np.pi, -0.2]  # Example NHPP1 parameters
distr = "Burr"
params = [0.4801, 3.9495 * 1e16, 2.1524]  # Example Burr parameters
Tmax = max(T)
N = 1000

# Calculate bond zero coupon
result = bond_zero_coupon(Z, D, T, r, lambda_type, parlambda, distr, params, Tmax, N)
print(result)


import numpy as np

def burrrnd(alpha=1, lambda_=1, tau=2, n=1, m=1):
    """
    Generates an M-by-N array of random numbers from the Burr distribution with parameters ALPHA, LAMBDA, and TAU.

    Parameters:
    alpha (float): Shape parameter of the Burr distribution (default=1).
    lambda_ (float): Scale parameter of the Burr distribution (default=1).
    tau (float): Another shape parameter of the Burr distribution (default=2).
    n (int): Number of rows in the output array (default=1).
    m (int): Number of columns in the output array (default=1).

    Returns:
    np.ndarray: An M-by-N array of random numbers from the Burr distribution.
    """
    u = np.zeros((n, m))
    for i in range(m):
        u[:, i] = (lambda_ * (np.random.uniform(0, 1, n) ** (-1 / alpha) - 1)) ** (1 / tau)
    return u

def mixexprnd(p=0.5, beta1=1, beta2=2, n=1, m=1):
    """
    Generates an M-by-N array of random numbers from the mixed exponential distribution with parameters P, BETA1, and BETA2.

    Parameters:
    p (float): Probability of choosing the first exponential distribution (default=0.5).
    beta1 (float): Rate parameter of the first exponential distribution (default=1).
    beta2 (float): Rate parameter of the second exponential distribution (default=2).
    n (int): Number of rows in the output array (default=1).
    m (int): Number of columns in the output array (default=1).

    Returns:
    np.ndarray: An M-by-N array of random numbers from the mixed exponential distribution.
    """
    y = np.random.exponential(scale=beta2, size=n * m)
    aux = np.random.uniform(0, 1, n * m) <= p
    
    if np.any(aux):
        y[aux] = np.random.exponential(scale=beta1, size=np.sum(aux))
    
    y = y.reshape(n, m)
    return y

# Example usage with default parameters
burr_sample = burrrnd()
mixexp_sample = mixexprnd()

print("Burr sample:", burr_sample)
print("Mixed exponential sample:", mixexp_sample)



import numpy as np

def sim_hpp(lambda_, T, N):
    """
    Generates N trajectories of the homogeneous Poisson process with intensity LAMBDA over the time horizon T.
    
    Parameters:
    lambda_ (float): Intensity of the Poisson process.
    T (float): Time horizon.
    N (int): Number of trajectories.
    
    Returns:
    np.ndarray: A 3D array representing the trajectories of the homogeneous Poisson process.
    """
    if lambda_ <= 0 or not isinstance(lambda_, (int, float)):
        raise ValueError("sim_hpp: Lambda must be a positive scalar.")
    if T <= 0 or not isinstance(T, (int, float)):
        raise ValueError("sim_hpp: T must be a positive scalar.")
    if N <= 0 or not isinstance(N, int):
        raise ValueError("sim_hpp: N must be a positive scalar.")
    
    EN = np.random.poisson(lambda_ * T, N)
    max_EN = 2 * np.max(EN) + 2
    y = np.zeros((max_EN, N, 2))
    y[:, :, 0] = T
    y[:, :, 1] = np.tile(EN, (max_EN, 1))

    for i in range(N):
        if EN[i] > 0:
            ttmp = np.sort(T * np.random.uniform(0, 1, EN[i]))
            y[1:(2 * EN[i] + 1), i, 0] = np.concatenate(([0], ttmp.repeat(2)))
        y[1:(2 * EN[i] + 2), i, 1] = np.concatenate(([0], np.floor(np.arange(1, 2 * EN[i]) / 2), [EN[i]]))

    return y

def sim_nhpp(lambda_, parlambda, T, N):
    """
    Generates N trajectories of the non-homogeneous Poisson process with intensity specified by LAMBDA.
    
    Parameters:
    lambda_ (int): Type of intensity function (0 - sine function, 1 - linear function, 2 - sine square function).
    parlambda (list): Parameters of the intensity function.
    T (float): Time horizon.
    N (int): Number of trajectories.
    
    Returns:
    np.ndarray: A 3D array representing the trajectories of the non-homogeneous Poisson process.
    """
    a = parlambda[0]
    b = parlambda[1]
    if lambda_ == 0:
        d = parlambda[2]
        JM = sim_hpp(a + b, T, N)
    elif lambda_ == 1:
        JM = sim_hpp(a + b * T, T, N)
    elif lambda_ == 2:
        d = parlambda[2]
        JM = sim_hpp(a + b * T, T, N)
    else:
        raise ValueError("sim_nhpp: Lambda must be 0, 1, or 2.")
    
    rjm = JM.shape[0]
    yy = np.zeros((rjm, N, 2))
    yy[:, :, 0] = T

    maxEN = 0
    for i in range(N):
        pom = JM[:, i, 0][JM[:, i, 0] < T]
        pom = pom[1::2]  # Take every second element starting from index 1
        R = np.random.uniform(0, 1, len(pom))
        if lambda_ == 0:
            lambdat = (a + b * np.sin(2 * np.pi * (pom + d))) / (a + b)
        elif lambda_ == 1:
            lambdat = (a + b * pom) / (a + b * T)
        elif lambda_ == 2:
            lambdat = (a + b * np.sin(2 * np.pi * (pom + d)) ** 2) / (a + b)
        
        pom = pom[R < lambdat]
        EN = len(pom)
        maxEN = max(maxEN, EN)
        yy[1:(2 * EN + 1), i, 0] = np.concatenate(([0], np.repeat(pom, 2)))
        yy[1:(2 * EN), i, 1] = np.floor(np.arange(1, 2 * EN) / 2)
        yy[(2 * EN + 1):rjm, i, 1] = EN

    return yy[:(2 * maxEN + 2), :, :]


lambda_ = 0
parlambda = [35.32, 2.32 * 2 * np.pi, -0.2]
T = 10
N = 5

homogeneous_sample = sim_hpp(lambda_=2, T=10, N=5)
non_homogeneous_sample = sim_nhpp(lambda_=lambda_, parlambda=parlambda, T=T, N=N)

print("Homogeneous Poisson Process Sample:\n", homogeneous_sample)
print("Non-Homogeneous Poisson Process Sample:\n", non_homogeneous_sample)


def pareto_rnd(alpha=1, lambda_=1, n=1, m=1):
    """
    Returns an M-by-N array of random numbers chosen from the Pareto distribution with parameters ALPHA, LAMBDA.
    
    Parameters:
    alpha (float): Shape parameter of the Pareto distribution.
    lambda_ (float): Scale parameter of the Pareto distribution.
    n (int): Number of rows.
    m (int): Number of columns.
    
    Returns:
    np.ndarray: M-by-N array of Pareto-distributed random numbers.
    """
    u = np.zeros((n, m))
    for i in range(m):
        u[:, i] = lambda_ * (np.random.uniform(0, 1, n) ** (-1 / alpha) - 1)
    return u

def sim_nhppalp(lambda_, parlambda, distrib, params, T, N):
    """
    Generates an aggregate loss process driven by the non-homogeneous Poisson process.
    
    Parameters:
    lambda_ (int): Type of intensity function (0 - sine, 1 - linear, 2 - sine square).
    parlambda (list): Parameters of the intensity function.
    distrib (str): Claim size distribution.
    params (list): Parameters of the claim size distribution.
    T (float): Time horizon.
    N (int): Number of trajectories.
    
    Returns:
    np.ndarray: 2*max+2 x N x 2 array representing the generated process.
    """
    if lambda_ not in [0, 1, 2]:
        raise ValueError("simNHPPALP: Lambda must be either 0, 1, or 2.")
    if T <= 0 or not isinstance(T, (int, float)):
        raise ValueError("simNHPPALP: T must be a positive scalar.")
    if N <= 0 or not isinstance(N, int):
        raise ValueError("simNHPPALP: N must be a positive scalar.")
    if (len(parlambda) != 3 and lambda_ != 1) or (len(parlambda) != 2 and lambda_ == 1):
        raise ValueError("simNHPPALP: Incorrect length of parlambda for the given lambda.")
    if distrib in ["Burr", "mixofexps"] and len(params) != 3:
        raise ValueError("simNHPPALP: For Burr and mixofexps distributions, params must be a 3 x 1 vector.")
    if distrib in ["gamma", "lognormal", "Pareto", "Weibull"] and len(params) != 2:
        raise ValueError("simNHPPALP: For gamma, lognormal, Pareto and Weibull distributions, params must be a 2 x 1 vector.")
    if distrib == "exponential" and len(params) != 1:
        raise ValueError("simNHPPALP: For exponential distribution, params must be a scalar.")
    if distrib not in ["exponential", "gamma", "mixofexps", "Weibull", "lognormal", "Pareto", "Burr"]:
        raise ValueError("simNHPPALP: distrib should be: exponential, gamma, mixofexps, Weibull, lognormal, Pareto, or Burr")

  

    poisproc = sim_nhpp(lambda_, parlambda, T, N)
    rpp, cpp, _ = poisproc.shape
    losses = np.zeros((rpp, cpp))

    if distrib == "Burr":
        
        val = burrrnd()
        pass
    elif distrib == "exponential":
        dist_func = lambda size: np.random.exponential(scale=params[0], size=size)
    elif distrib == "gamma":
        dist_func = lambda size: np.random.gamma(shape=params[0], scale=1/params[1], size=size)
    elif distrib == "lognormal":
        dist_func = lambda size: np.random.lognormal(mean=params[0], sigma=params[1], size=size)
    elif distrib == "mixofexps":
        # Define mixexprnd function here
        pass
    elif distrib == "Pareto":
        dist_func = lambda size: pareto_rnd(alpha=params[0], lambda_=params[1], n=size, m=1).flatten()
    elif distrib == "Weibull":
        dist_func = lambda size: np.random.weibull(a=params[1], size=size) * (params[0] ** (1 / params[1]))

    for i in range(N):
        if N == 1:
            aux = np.min(np.where(poisproc[:, 0] == T))
        else:
            aux = np.min(np.where(poisproc[:, i, 0] == T))
        if aux > 2:
            laux = np.cumsum(dist_func(aux//2 - 1))
            losses[2:aux, i] = laux[np.ceil(np.arange(1, aux - 1) / 2).astype(int)]
            if aux < rpp:
                losses[aux:rpp, i] = laux[-1]
        else:
            losses[:, i] = 0

    y = np.zeros_like(poisproc)
    y[:, :, 0] = poisproc[:, :, 0]
    y[:, :, 1] = losses

    return y

lambda_ = 0
parlambda = [35.32, 2.32 * 2 * np.pi, -0.2]
distrib = "Pareto"
params = [1.5, 1.0]
T = 10
N = 5

result = sim_nhppalp(lambda_, parlambda, distrib, params, T, N)
print(result)


## Main
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define parameters and distributions
parlambda = [35.32, 2.32 * 2 * np.pi, -0.2]  # NHPP1
distr1 = "Burr"
distr2 = "lognormal"
params1 = [0.4801, 3.9495 * 1e+16, 2.1524]  # Burr
params2 = [18.3806, 1.1052]  # Lognormal

# Read data file
data = pd.read_csv("ncl.dat", delim_whitespace=True, header=None)
A = np.mean(data.iloc[:, 2]) * (34.2 / 4)

# Define sequences and other parameters
na = 41  # default 41
D = np.linspace(A, 12 * A, na)
B = 0.25
nb = 41  # default 41
T = np.linspace(B, 8 * B, nb)
Tmax = np.max(T)
lambda_ = 0
N = 1000  # default 1000
r = np.log(1.025)
Z = 1.06

# Define the BondZeroCoupon function (this needs to be implemented based on the domain logic)
def bond_zero_coupon(Z, D, T, r, lambda_, parlambda, distrib, params, Tmax, N):
    
    return np.random.rand(len(D), 3)

# Generate data using the BondZeroCoupon function
d1 = bond_zero_coupon(Z, D, T, r, lambda_, parlambda, distr1, params1, Tmax, N)
d2 = bond_zero_coupon(Z, D, T, r, lambda_, parlambda, distr2, params2, Tmax, N)

# Prepare data for plotting
y = d1[:, 0]
x = d1[:, 1] / 1e+09
z = d1[:, 2] - d2[:, 2]
data = pd.DataFrame({'x': x, 'y': y, 'z': z})

# Plot using matplotlib
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Wireframe plot
ax.plot_wireframe(data['x'], data['y'], data['z'])

# Set plot labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Wireframe plot')

# Customize the plot appearance
ax.view_init(elev=30, azim=-60)
plt.show()

