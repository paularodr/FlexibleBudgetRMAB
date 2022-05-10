import numpy as np
import minmax_methods as minmax
from environments import dropOutState

T = 2 #flexible time horizon
H = 10 #total time horizon
N = 5
B = 1

# Define K matrix
K = np.zeros((T,H+1))
for t in range(T):
    K[t,t] = 1
K[:,H] = [-1]*T

# inputs
tau = 0.1
sigma = 0.1
print(f'tau*sigma*T<1? {tau*sigma*T<1}')
x = np.zeros(H+1) # initial lagrange multipliers
y = np.ones(T) # initial budget variables
start_state = [2]*N

# set dropout state environment
env = dropOutState(N, B, start_state,P_noise=False)
P = env.T_one
S = env.S
R = env.R
C = [0,1]

gamma =0.95
n_iter=200
tolerance=1e-06
sample_size=50

(l,l_diff),(b,b_diff) = minmax.chambolle_pock(tau, sigma, K, x, y, start_state, P, S, R, C, B,  n_iter, tolerance, sample_size, gamma)

import matplotlib.pyplot as plt
plt.plot([x[0] for x in b_diff])
plt.show()

