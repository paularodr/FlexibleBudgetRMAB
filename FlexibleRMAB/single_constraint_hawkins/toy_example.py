import pickle
import numpy as np
from tqdm import tqdm 
import hawkins_actions
import hawkins_methods
from environments import dropOutState

gamma = 0.95
N = 2
B = 1.0 #one step budget
start_state = np.array([1]*N)

def actions_set_lambda(env, gamma, steps, epsilon=1e-1):
    if steps == 2:
        actions_hawkins, L, Q, lambda_val, Q_vals_per_state = hawkins_actions.get_hawkins_actions(env.N, env.T_two, env.R, env.C, 2*env.B, env.current_state, gamma)
        new_lambda = lambda_val-epsilon
        actions_lambda, L_vals, Q_vals, new_lambda, Q_vals_per_state = hawkins_actions.get_hawkins_actions_preset_lambda(env.N, env.T_two, env.R, env.C, 2*env.B, env.current_state, lambda_val=new_lambda, gamma=gamma)
    if steps == 1:
        actions_hawkins, L, Q, lambda_val, Q_vals_per_state = hawkins_actions.get_hawkins_actions(env.N, env.T_one, env.R, [0,1], env.B, env.current_state, gamma)
        new_lambda = lambda_val-epsilon
        actions_lambda, L_vals, Q_vals, new_lambda, Q_vals_per_state = hawkins_actions.get_hawkins_actions_preset_lambda(env.N, env.T_one, env.R, [0,1], env.B, env.current_state, lambda_val=new_lambda, gamma=gamma)
    return actions_lambda

def actions_set_lambda_window(env, gamma, B, epsilon=1e-1):
    actions_hawkins, L, Q, lambda_val, Q_vals_per_state = hawkins_actions.get_hawkins_actions(env.N, env.T_one, env.R, [0,1], B, env.current_state, gamma)
    new_lambda = lambda_val-epsilon
    actions_lambda, L_vals, Q_vals, new_lambda, Q_vals_per_state = hawkins_actions.get_hawkins_actions_preset_lambda(env.N, env.T_one, env.R, [0,1], B, env.current_state, lambda_val=new_lambda, gamma=gamma)
    return actions_lambda

def append_results(algo, actions, state, reward):
    results[algo]['actions'] = results[algo]['actions'] + [actions]
    results[algo]['states'] = results[algo]['states'] + [state]
    results[algo]['rewards'] = results[algo]['rewards'] + [reward]
    return results

algos = ['fixed_window','closing_window', 'singlestep', 'hawkins_flexible']
results = {}    
envs = {}
for algo in algos:
    results[algo] = {'actions':[],'states':[],'rewards':[]}
    envs[algo] = dropOutState(N, B, start_state)

# SINGLE STEP WITH REWARD SHAPING
algo = 'singlestep'
random_states = []
for k in range(2):
    actions = actions_set_lambda(envs[algo], gamma, steps=1, epsilon=1e-1)
    random_states.append(np.random.get_state())
    np.random.set_state(random_states[k])
    current_state, reward = envs[algo].onestep(actions)
    results = append_results(algo, actions, current_state, reward)

# FIXED WINDOW (TAKE 2 STEPS)
algo = 'fixed_window'    
actions_twostep = actions_set_lambda(envs[algo], gamma, steps=2, epsilon=1e-1)
actions_first = np.array([(1 if x==1 or x==3 else 0) for x in actions_twostep])
actions_second = np.array([(1 if x==2 or x==3 else 0) for x in actions_twostep])

np.random.set_state(random_states[0])
current_state, reward = envs[algo].onestep(actions_first) #take first step
results = append_results(algo,actions_first, current_state, reward)

np.random.set_state(random_states[1])
current_state, reward = envs[algo].onestep(actions_second) #take second step
results = append_results(algo,actions_second, current_state, reward)

# CLOSING WINDOW
algo = 'closing_window'
actions_twostep = actions_set_lambda(envs[algo], gamma, steps=2, epsilon=1e-1)
actions_first = np.array([(1 if x==1 or x==3 else 0) for x in actions_twostep])
actions_second = np.array([(1 if x==2 or x==3 else 0) for x in actions_twostep])

## First step of closing window
np.random.set_state(random_states[0])
current_state, reward = envs[algo].onestep(actions_first) #take first step
results = append_results(algo,actions_first, current_state, reward)
reserve = 2*B - actions_first.sum()

## Second step of closing window
if reserve>0:
    actions_second_window = actions_set_lambda_window(envs[algo], gamma, B=reserve, epsilon=1e-1)
else:
    actions_second_window = np.array([0]*N)

np.random.set_state(random_states[1])
current_state, reward = envs[algo].onestep(actions_second_window) #take second step
results = append_results(algo,actions_second_window, current_state, reward)

# maxmin hawkings with b_t variables (max over fixed b_t)
algo = 'hawkins_flexible'
obj_vals = []
bvals_list = [[1,1],[2,0],[0,2]] 
for bvals in bvals_list:
    L_vals, lambda_vals, b_vals, m = hawkins_methods.hawkins_fixed_bt(2, 2, envs[algo].T_one, envs[algo].R, envs[algo].C, B, bvals, envs[algo].current_state)
    obj_vals.append(m.objVal)
b_argmax = bvals_list[np.argmax(obj_vals)]

L_vals, lambda_vals, b_vals, m = hawkins_methods.hawkins_fixed_bt(2, 2, envs[algo].T_one, envs[algo].R, envs[algo].C, B, b_argmax, envs[algo].current_state)

H = 2
P = envs[algo].T_one
R = envs[algo].R
C = envs[algo].C
current_state = envs[algo].current_state

Q_vals = np.zeros((N, P.shape[2], P.shape[1]))

current_state = current_state.reshape(-1).astype(int)

for i in range(N):
    for a in range(P.shape[2]):
        for s in range(P.shape[1]):
            Q_vals[i,a,s] = R[i,s] - C[a]*lambda_vals[0] + L_vals[i,:,0].dot(P[i,s,a])



Q_vals_per_state = np.zeros((N, P.shape[2]))
for i in range(N):
    s = current_state[i]
    Q_vals_per_state[i] = Q_vals[i,:,s]

decision_matrix = hawkins_methods.action_knapsack(Q_vals_per_state, C, B)

actions = np.argmax(decision_matrix, axis=1)

