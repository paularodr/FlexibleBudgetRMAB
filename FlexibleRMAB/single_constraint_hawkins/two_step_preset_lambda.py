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


    return actions, L_vals, Q_vals, lambda_val, Q_vals_per_state





def get_hawkins_actions_preset_lambda(T, R, C, B, current_state, lambda_val, gamma):
    actions = np.zeros(N)

    # N x A x S matrix
    Q_vals = np.zeros((N, T.shape[2], T.shape[1]))

    current_state = current_state.reshape(-1).astype(int)

    L_vals, lambda_val = hawkins_methods.hawkins_set_lambda(T, R, C, B, current_state, lambda_val=lambda_val, gamma=gamma)


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


    return actions, L_vals, Q_vals, lambda_val, Q_vals_per_state



N = 5
S = 3
A = 4 #amount of action types
B = 4.0

#cost
C = [0,1,1,2] # (0,0), (1,0), (0,1), (1,1)

#reward
R = np.array([[0,0,1] for i in range(N)])
#R = np.array([np.arange(S) for _ in range(N)])

gamma = 0.95

# one-step transition probabilities
#not act transition probability. SxS
P0 = np.array([
    [1.0, 0.0, 0.0],
    [0.9, 0.1, 0.0],
    [0.0 ,0.8, 0.2]
    ])

#act transition probability. SxS
P1 = np.array([
    [1, 0.0, 0.0],
    [0.0, 0.0, 1.0],
    [0.0, 0.0, 1.0]
    ])

# two-step transition probabilities
T0 = np.matmul(P0,P0) # action 0: (0,0)
T1 = np.matmul(P1,P0) # action 1: (1,0)
T2 = np.matmul(P0,P1) # action 2: (0,1)
T3 = np.matmul(P1,P1) # action 3: (1,1)

T_i = np.array([T0, T1, T2, T3])
T_i = np.swapaxes(T_i, 0, 1)
T = np.array([T_i for _ in range(N)])


#current_state = np.random.choice(np.arange(S), N, replace=True)
#current_state = np.array([2]*N)
# i dont think its a tie breaking issue. for current state [2,0,1,1,1], hawkins actions are [0 0 1 1 1] (total cost 3 < 4).
# intuition: seems like the issue comes from the fact that not acting on an arm in state 2 means passing through state 1 before going to 0, which still accumulates reward whith a low cost
current_state = np.array([2,2,2,1,1])
actions, L, Q, lambda_val, Q_vals_per_state = get_hawkins_actions(T, R, C, B, current_state, gamma)

print('state\t',current_state)
print('actions\t',actions)

#Check: Seems like Q values for action 0 are higher or equal than for other actions
print('hawkins solve')
print(actions)
print(L)
print(Q)
print(lambda_val)
print('Q per state')
print(Q_vals_per_state)


epsilon=1e-1
new_lambda = lambda_val-epsilon

actions, L_vals, Q_vals, new_lambda, Q_vals_per_state = get_hawkins_actions_preset_lambda(T, R, C, B, current_state, lambda_val=new_lambda, gamma=gamma)

print('set lambda val solve')
print('state\t',current_state)
print('actions\t',actions)
print(L_vals)
print(Q_vals)
print(Q_vals_per_state)
print(new_lambda)


# plot state value functions for different values of lambda
lambda_vals = np.arange(0,20,0.5)