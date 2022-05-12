import pickle
import time
import numpy as np
import hawkins_actions
import compressing_methods
import minmax_methods as minmax
from environments import dropOutState
from tqdm import tqdm 
import argparse
import os

parser = argparse.ArgumentParser(description='Experiments')
parser.add_argument('seed', metavar='seed', type=int,
                    help='Set random seed')

args = parser.parse_args()

def append_results(algo, actions, state, reward):
    actions = list(actions)
    state = list(state)
    results[algo]['actions'] = results[algo]['actions'] + [actions]
    results[algo]['states'] = results[algo]['states'] + [state]
    results[algo]['rewards'] = results[algo]['rewards'] + [reward]
    return results

algos = ['hawkins_single','hawkins_fixed','hawkins_closing','chambolle-pock']

# Parameters

seed = args.seed
T = 2 #flexible time horizon
H = 50 #total time horizon
N = 5 #arms
S = 3 #states
B = 1.0 #one step budget
C = [0,1]
start_state = np.array([2]*N)

# Define K matrix
def createK(T,H):
    K = np.zeros((T,H+1))
    for t in range(T):
        K[t,t] = 1
    K[:,H] = [-1]*T
    return K

# chambolle-pock algorithms
gamma = 0.95
tau = 0.1
sigma = 0.1
x = np.zeros(H+1) # initial lagrange multipliers
y = np.ones(T) # initial budget variables
n_iter=200
tolerance=1e-06
sample_size=50

HORIZON = int(H/T) #double steps
# dictionaries to save results
results = {}
envs = {}
np.random.seed(seed)
state = np.random.get_state()
for algo in algos:
    results[algo] = {'actions':[],'states':[],'rewards':[], 'runtime':0}
    np.random.set_state(state)
    envs[algo] = dropOutState(N, B, start_state,P_noise=True)

for t in range(HORIZON):
    print(f'timestep: {t}/{HORIZON}')
    # Hawkins single
    algo = 'hawkins_single'
    random_states = []
    P = envs[algo].T_one
    R = envs[algo].R
    start = time.time()
    for k in range(T):
            output = hawkins_actions.get_hawkins_actions(N, P, R, C, B, envs[algo].current_state, gamma)
            #output = get_hawkins_actions(N, P, R, C, B, start_state, gamma)
            # output[0]
            # output = (actions, L_vals, Q_vals, lambda_val, Q_vals_per_state)
            actions = output[0]
            random_states.append(np.random.get_state())
            np.random.set_state(random_states[k])
            current_state, reward = envs[algo].onestep(actions)
            results = append_results(algo, actions, current_state, reward)
    runtime = (time.time()-start)
    results[algo]['runtime'] += runtime

    # Hawkins fixed
    algo = 'hawkins_fixed'
    start = time.time()
    actions = compressing_methods.hawkins_window(T, N, P, R, C, T*B, envs[algo].current_state, gamma)
    states, rewards = envs[algo].multiple_steps(T, actions, random_states)
    for i in range(T):
        results = append_results(algo, actions[i], states[i], rewards[i])
    runtime = (time.time()-start)
    results[algo]['runtime'] += runtime


    # Hawkins closing
    algo = 'hawkins_closing'
    used = 0
    start = time.time()
    for i in range(T):
        size_close = T - i
        budget = B*T - used # if resources not shared in future budget = B*size_close
        if budget > 0:
            actions = compressing_methods.hawkins_window(size_close, N, P, R, C, budget, envs[algo].current_state, gamma)
        else:
            actions = [np.array([0]*N)]
        used += actions[0].sum()
        np.random.set_state(random_states[i])
        current_state, reward = envs[algo].onestep(actions[0])
        results = append_results(algo, actions[0], current_state, reward)
    runtime = (time.time()-start)
    results[algo]['runtime'] += runtime


    # Minmax Chmbolle-Pock
    # algo = 'chambolle-pock'
    # used = 0
    # start = time.time()
    # for i in range(T):
    #     size_close = T - i
    #     budget = B*T - used
    #     time_horizon = H - t -i
    #     K = createK(size_close,time_horizon)
    #     if budget > 0:
    #         x = np.zeros(time_horizon+1)
    #         y = np.ones(size_close)
    #         # CHANGE H AND T GIVEN TO CHAMBOLLE POCK
    #         actions, l, budgets, Q_vals = minmax.chambolle_pock_actions(tau, sigma, K, x, y, envs[algo].current_state, P, S, R, C, B, budget,  n_iter, tolerance, sample_size)
    #     else:
    #         actions = np.array([0]*N)
    #     used += actions.sum()
    #     np.random.set_state(random_states[i])
    #     current_state, reward = envs[algo].onestep(actions)
    #     results = append_results(algo, actions, current_state, reward)
    # runtime = (time.time()-start)
    # results[algo]['runtime'] += runtime


# results from list to numpy array
for k in ['actions','states','rewards']:
    for d in algos:
        results[d][k] = np.array(results[d][k])

#save results
experiment = f'T_{T}_H_{H}_N_{N}_B_{int(B)}'
dir_path = f'experiments/dropOutState/closing_window/{experiment}'
with open(f'{dir_path}/{experiment}_seed_{seed}.pkl', 'wb') as f:
    pickle.dump(results, f)