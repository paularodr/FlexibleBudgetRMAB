import numpy as np 
import hawkins_methods


def get_hawkins_actions(N, T, R, C, B, current_state, gamma):
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

def get_hawkins_actions(N, T, R, C, B, current_state, gamma):
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



def get_hawkins_actions_preset_lambda(N, T, R, C, B, current_state, lambda_val, gamma):
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

def actions_set_lambda(env, gamma, steps, epsilon=1e-1):
    if steps == 2:
        actions_hawkins, L, Q, lambda_val, Q_vals_per_state = get_hawkins_actions(env.N, env.T_two, env.R, env.C, 2*env.B, env.current_state, gamma)
        new_lambda = lambda_val-epsilon
        actions_lambda, L_vals, Q_vals, new_lambda, Q_vals_per_state = get_hawkins_actions_preset_lambda(env.N, env.T_two, env.R, env.C, 2*env.B, env.current_state, lambda_val=new_lambda, gamma=gamma)
    if steps == 1:
        actions_hawkins, L, Q, lambda_val, Q_vals_per_state = get_hawkins_actions(env.N, env.T_one, env.R, [0,1], env.B, env.current_state, gamma)
        new_lambda = lambda_val-epsilon
        actions_lambda, L_vals, Q_vals, new_lambda, Q_vals_per_state = get_hawkins_actions_preset_lambda(env.N, env.T_one, env.R, [0,1], env.B, env.current_state, lambda_val=new_lambda, gamma=gamma)
    return actions_lambda

def actions_set_lambda_window(env, gamma, B, epsilon=1e-1):
    actions_hawkins, L, Q, lambda_val, Q_vals_per_state = get_hawkins_actions(env.N, env.T_one, env.R, [0,1], B, env.current_state, gamma)
    new_lambda = lambda_val-epsilon
    actions_lambda, L_vals, Q_vals, new_lambda, Q_vals_per_state = get_hawkins_actions_preset_lambda(env.N, env.T_one, env.R, [0,1], B, env.current_state, lambda_val=new_lambda, gamma=gamma)
    return actions_lambda