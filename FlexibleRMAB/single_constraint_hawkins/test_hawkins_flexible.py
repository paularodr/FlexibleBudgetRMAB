import numpy as np
import hawkins_methods
from environments import dropOutState

H = 2
T = 2

gamma = 1.0
N = 4
B = 1.0 #one step budget
start_state = np.array([2]*N)


env = dropOutState(N, B, start_state, P_noise=False)
P = env.T_one
R = env.R
C = env.C

Q_vals = np.zeros((N, P.shape[2], P.shape[1], H))

current_state = env.current_state.reshape(-1).astype(int)

L_vals, lambda_vals, b_vals, m = hawkins_methods.hawkins_flexible(H, T, P, R, C, B, current_state, gamma=gamma)

m.objVal


for p in range(N):
    for i in range(3):
        for j in range(2):
            for h in range(H):
                print(p,i,j)
                print(- lamtime[h]*C[j] + gamma*(L[p,:,h+1]+R[p,:]).dot(P[p,i,j]))


for v in m.getVars():
    print(v.varName,v.x)


for i in range(N):
    for a in range(P.shape[2]):
        for s in range(P.shape[1]):
            for h in range(H):
                Q_vals[i,a,s,h] = R[i,s] - C[a]*lambda_vals[h] + gamma*L_vals[i,:,h].dot(P[i,s,a])