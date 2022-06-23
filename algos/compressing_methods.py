import itertools
import numpy as np
from . import hawkins_actions

def compress(P, C, size):
    # P is a ndarray of size Nx2xAx2 (binary actions)
    # return:
    # - transition probability matriz for steps of size 'size'
    # - cost of actions
    # - list of multi-actions

    multi_actions = list(itertools.product([0, 1], repeat=size))
    Psize = []
    for i in range(P.shape[0]):
        P_i = P[i]
        P_i = np.swapaxes(P_i, 0, 1)
        Psize_i = []
        for a in multi_actions:
            x = P_i[a[0]]
            for d in range(1,size):
                x = np.matmul(x,P_i[a[d]])
            Psize_i.append(x)
        Psize_i = np.swapaxes(Psize_i, 0, 1)
        Psize.append(Psize_i)
    Psize = np.array(Psize)

    costs = [np.sum([C[i] for i in x]) for x in multi_actions]
    return Psize, costs, multi_actions



def hawkins_window(horizon, window, N, P, R, C, B, current_state, gamma, finite_horizon):
    # B: per step budget
    # P: one step transition probabilities
    # return: array for size (size)xN with binary actions to perform on each arm at each step of the window
    
    # Calculate P_size: transition probabilities for steps of size 'size'
    if window >1:
        Psize, costs, multi_actions = compress(P, C, window)
    else:
        Psize = P
        costs = C
        multi_actions = [0,1]

    # multi-actions to take in window of size 'size' using Hawkins
    actions_window, L, Q, lambda_val, Q_vals_per_state = hawkins_actions.get_hawkins_actions(horizon, N, Psize, R, costs, B, current_state, gamma, finite_horizon)

    # from multi-action to bianry actions at each step
    if window >1:
        actions_binary = [[]]*window
        for action_i in actions_window:
            for t in range(window):
                actions_binary[t] = actions_binary[t] + [multi_actions[action_i][t]]
        actions_binary = np.array(actions_binary)
    else:
        actions_binary = [actions_window]
    return actions_binary