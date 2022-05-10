import numpy as np 
import lp_methods

N = 5 # number of arms
S = 3 # number of states
A = 2 # number of actions
B = 2 # indexes should be the same regardless of budget, but make sure it is non-negative

np.random.seed(0)

T = np.random.rand(N,S,A,S)

# normalize so we get valid transition probabilities
P0 = np.array([
                [0.95, 0.05, 0.0],
                [0.75, 0.25, 0.0],
        ])

P0 = np.array([
                [0.95, 0.05, 0.0],
                [0.75, 0.25, 0.0],
        ])

P0 = np.array([
                [0.95, 0.05, 0.0],
                [0.75, 0.25, 0.0],
        ])

T = T /  T.sum(axis=3, keepdims=True)

# make the reward function something simple, only based on current state
R = np.array([[0,1] for i in range(N)])

# simple costs...
C = [0, 1]

# randomly sample start states
start_state = np.random.choice([0,1], size=N, replace=True)



a_index = 1 # for binary-action case, a_index is always 1

V_adjusted, indexes = lp_methods.lp_to_compute_index(T, R, C, B, start_state, a_index, lambda_lim=None, gamma=0.95)

print(indexes)
