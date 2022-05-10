import numpy as np 
import hawkins_methods


def get_hawkins_actions(T, R, C, B, current_state, gamma):
    actions = np.zeros(N)

    # N x A x S matrix
    Q_vals = np.zeros((N, T.shape[2], T.shape[1]))

    current_state = current_state.reshape(-1).astype(int)

    L_vals, lambda_val = hawkins_methods.hawkins(T, R, C, B, current_state, gamma=gamma)


    for i in range(N):
        for a in range(T.shape[2]):
            for s in range(T.shape[1]):
                Q_vals[i,a,s] = R[i,s] - C[a]*lambda_val + gamma*L_vals[i].dot(T[i,s,a])


    Q_vals_per_state = np.zeros((N, T.shape[2]))
    for i in range(N):
        s = current_state[i]
        Q_vals_per_state[i] = Q_vals[i,:,s]


    decision_matrix = hawkins_methods.action_knapsack(Q_vals_per_state, C, B)

    actions = np.argmax(decision_matrix, axis=1)

    if not (decision_matrix.sum(axis=1) <= 1).all(): raise ValueError("More than one action per person")

    payment = 0
    for i in range(len(actions)):
        payment += C[actions[i]]
    if not payment <= B: 
        print("budget")
        print(B)
        print("Cost")
        print(C)
        print("ACTIONS")
        print(actions)
        raise ValueError("Over budget")


    return actions


N = 5
S = 2
A = 4 #amount of action types

T = np.random.rand(N,S,A,S)
T = T/T.sum(axis=-1, keepdims=True)

# C = np.array([np.arange(A) for _ in range(N)])
C = np.arange(A)

R = np.array([np.arange(S) for _ in range(N)])

gamma = 0.95

B = 3.0

current_state = np.random.choice(np.arange(S), N, replace=True)

actions = get_hawkins_actions(T, R, C, B, current_state, gamma)

print('state\t',current_state)
print('actions\t',actions)
