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
S = 3
A = 4 #amount of action types
B = 4.0

#cost
C = [0,1,1,2]

#reward
R = np.array([[0,0,1] for i in range(N)])
#R = np.array([np.arange(S) for _ in range(N)])

gamma = 0.95

# one-step transition probabilities
P0 = np.array([
    [1,0,0],
    [0.8,0.2,0],
    [0,0.8,0.2]
    ])

P1 = np.array([
    [0.8,0.2,0],
    [0,0,1],
    [0,0,1]
    ])

# two-step transition probabilities
T0 = np.dot(P0,P0)
T1 = np.dot(P1,P0)
T2 = np.dot(P0,P1)
T3 = np.dot(P1,P1)

T_i = np.array([T0, T1, T2, T3])
T_i = np.swapaxes(T_i, 0, 1)
T = np.array([T_i for _ in range(N)])


#current_state = np.random.choice(np.arange(S), N, replace=True)
#current_state = np.array([2]*N)
current_state = np.array([2,2,1,1,1])
actions = get_hawkins_actions(T, R, C, B, current_state, gamma)

print('state\t',current_state)
print('actions\t',actions)

#Check: Seems like Q values for action 0 are higher or equal than for other actions 