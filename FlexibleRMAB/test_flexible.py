import numpy as np 
import lp_methods

N = 5 # number of arms
S = 3 # number of states
A = 2 # number of actions
B = 2 # indexes should be the same regardless of budget, but make sure it is non-negative

np.random.seed(0)

P0 = np.array([
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
        ])

P1 = np.array([
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
        ])

P2 = np.array([
                [0.0, 0.5, 0.5],
                [0.0, 0.0, 1.0],
        ])



T_i = np.array([P0, P1, P2])
T = np.array([T_i for _ in range(N)])

# make the reward function something simple, only based on current state
R = np.array([[0,0,1] for i in range(N)])

# simple costs...
C = [0, 1]

# randomly sample start states
#start_state = np.random.choice([0,1], size=N, replace=True)
start_state = np.array([1 for _ in range(N)])


a_index = 1 # for binary-action case, a_index is always 1

V_adjusted, indexes = lp_methods.lp_to_compute_index(T, R, C, B, start_state, a_index, lambda_lim=None, gamma=0.95)

print(indexes)
