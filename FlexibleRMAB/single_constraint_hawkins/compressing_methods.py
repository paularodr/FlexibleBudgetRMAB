import itertools
import numpy as np
import hawkins_actions

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
            x = np.matmul(P_i[a[0]],P_i[a[1]])
            Psize_i.append(x)
        Psize_i = np.swapaxes(Psize_i, 0, 1)
        Psize.append(Psize_i)
    Psize = np.array(Psize)

    costs = [np.sum([C[i] for i in x]) for x in multi_actions]
    return Psize, costs, multi_actions



def hawkins_window(size, N, P, R, C, B, current_state, gamma=0.95):
    # B: per step budget
    # P: one step transition probabilities
    # return: array for size (size)xN with binary actions to perform on each arm at each step of the window
    
    # Calculate P_size: transition probabilities for steps of size 'size'
    if size >1:
        Psize, costs, multi_actions = compress(P, C, size)
    else:
        Psize = P
        costs = C
        multi_actions = [0,1]

    # multi-actions to take in window of size 'size' using Hawkins
    actions_window, L, Q, lambda_val, Q_vals_per_state = hawkins_actions.get_hawkins_actions(N, Psize, R, costs, B, current_state, gamma)

    # from multi-action to bianry actions at each step
    if size >1:
        actions_binary = [[]]*size
        for action_i in actions_window:
            for t in range(size):
                actions_binary[t] = actions_binary[t] + [multi_actions[action_i][t]]
        actions_binary = np.array(actions_binary)
    else:
        actions_binary = [actions_window]
    return actions_binary