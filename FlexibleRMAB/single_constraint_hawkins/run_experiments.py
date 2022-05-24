import pickle
import time
import numpy as np
import hawkins_actions
import compressing_methods
import minmax_methods as minmax
from environments import dropOutState, riskProneArms, birthDeathProcess
import argparse
import tracemalloc

parser = argparse.ArgumentParser(description='Experiments')
parser.add_argument('--seed', metavar='seed', type=int,
                    help='Set random seed')
parser.add_argument('--domain', metavar='dom', type=str, choices = ['dropOutState', 'riskProneArms', 'birthDeathProcess'],
                    help='Set domain')
parser.add_argument('--T', metavar='T', type=int,
                    help='Flexible time horizon')
parser.add_argument('--H', metavar='H', type=int,
                    help='Total time horizon')
parser.add_argument('--N', metavar='N', type=int,
                    help='Number of arms', default=10)
parser.add_argument('--S', metavar='S', type=int,
                    help='Number of states', default=2)
args = parser.parse_args()
 

def append_results(algo, actions, state, reward):
    actions = list(actions)
    state = list(state)
    results[algo]['actions'] = results[algo]['actions'] + [actions]
    results[algo]['states'] = results[algo]['states'] + [state]
    results[algo]['rewards'] = results[algo]['rewards'] + [reward]
    return results

algos = ['hawkins_single','hawkins_fixed','hawkins_closing','chambolle-pock-10','chambolle-pock-50','chambolle-pock-100', 'chambolle-pock-200']

# Parameters

seed = args.seed
domain = args.domain
T = args.T #flexible time horizon
H = args.H #total time horizon
N = args.N #arms
M = 1 #fraction of risk prone arms
S = args.S #states
B = 1.0 #one step budget
C = [0,1]

# chambolle-pock algorithms
gamma = 1
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
    results[algo] = {'actions':[],'states':[],'rewards':[], 'runtime':0, 'memory':0}
    np.random.set_state(state)
    if domain == 'dropOutState':
        envs[algo] = dropOutState(N, B,P_noise=True)
    if domain == 'riskProneArms':
        envs[algo] = riskProneArms(N,B,M)
    if domain == 'birthDeathProcess':
        envs[algo] = birthDeathProcess(N,B,S)


for t in range(HORIZON):
    print(f'timestep: {t}/{HORIZON}')
    # Hawkins single
    algo = 'hawkins_single'
    random_states = []
    P = envs[algo].T
    R = envs[algo].R
    start = time.time()
    tracemalloc.start()
    for k in range(T):
            output = hawkins_actions.get_hawkins_actions(H-t-k, N, P, R, C, B, envs[algo].current_state, gamma)
            actions = output[0]
            random_states.append(np.random.get_state())
            np.random.set_state(random_states[k])
            current_state, reward = envs[algo].onestep(actions)
            results = append_results(algo, actions, current_state, reward)
    runtime = (time.time()-start)
    results[algo]['runtime'] += runtime
    results[algo]['memory'] += tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    # Hawkins fixed
    algo = 'hawkins_fixed'
    start = time.time()
    tracemalloc.start()
    actions = compressing_methods.hawkins_window(T, N, P, R, C, T*B, envs[algo].current_state, gamma)
    states, rewards = envs[algo].multiple_steps(T, actions, random_states)
    for i in range(T):
        results = append_results(algo, actions[i], states[i], rewards[i])
    runtime = (time.time()-start)
    results[algo]['runtime'] += runtime
    results[algo]['memory'] += tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()


    # Hawkins closing
    algo = 'hawkins_closing'
    used = 0
    start = time.time()
    tracemalloc.start()
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
    results[algo]['memory'] += tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    #Minmax Chmbolle-Pock
    algo = 'chambolle-pock-50'
    used = 0
    for i in range(T):
        size_close = T - i
        budget = B*T - used
        time_horizon = H - t -i
        K = minmax.createK(size_close,time_horizon)
        if budget > 0:
            x = np.zeros(time_horizon+1)
            y = np.ones(size_close)
            # CHANGE H AND T GIVEN TO CHAMBOLLE POCK
            actions, l, budgets, Q_vals = minmax.chambolle_pock_actions(tau, sigma, K, x, y, envs[algo].current_state, P, R, C, B, budget,  50, tolerance, sample_size)
        else:
            actions = np.array([0]*N)
        used += actions.sum()
        np.random.set_state(random_states[i])
        current_state, reward = envs[algo].onestep(actions)
        results = append_results(algo, actions, current_state, reward)
    runtime = (time.time()-start)
    results[algo]['runtime'] += runtime

    #Minmax Chmbolle-Pock
    algo = 'chambolle-pock-100'
    used = 0
    for i in range(T):
        size_close = T - i
        budget = B*T - used
        time_horizon = H - t -i
        K = minmax.createK(size_close,time_horizon)
        if budget > 0:
            x = np.zeros(time_horizon+1)
            y = np.ones(size_close)
            actions, l, budgets, Q_vals = minmax.chambolle_pock_actions(tau, sigma, K, x, y, envs[algo].current_state, P, R, C, B, budget,  100, tolerance, sample_size)
        else:
            actions = np.array([0]*N)
        used += actions.sum()
        np.random.set_state(random_states[i])
        current_state, reward = envs[algo].onestep(actions)
        results = append_results(algo, actions, current_state, reward)
    runtime = (time.time()-start)
    results[algo]['runtime'] += runtime

    #Minmax Chmbolle-Pock
    algo = 'chambolle-pock-200'
    used = 0
    for i in range(T):
        size_close = T - i
        budget = B*T - used
        time_horizon = H - t -i
        K = minmax.createK(size_close,time_horizon)
        if budget > 0:
            x = np.zeros(time_horizon+1)
            y = np.ones(size_close)
            actions, l, budgets, Q_vals = minmax.chambolle_pock_actions(tau, sigma, K, x, y, envs[algo].current_state, P, R, C, B, budget,  200, tolerance, sample_size)
        else:
            actions = np.array([0]*N)
        used += actions.sum()
        np.random.set_state(random_states[i])
        current_state, reward = envs[algo].onestep(actions)
        results = append_results(algo, actions, current_state, reward)
    runtime = (time.time()-start)
    results[algo]['runtime'] += runtime


# results from list to numpy array
for k in ['actions','states','rewards']:
    for d in algos:
        results[d][k] = np.array(results[d][k])

for algo in algos:
    x=results[algo]['rewards'].sum()
    print(f'{algo}: {x}')

#save results
experiment = f'T_{T}_H_{H}_N_{N}_S_{S}_B_{int(B)}'
dir_path = f'experiments/{domain}/{experiment}'
with open(f'{dir_path}/{experiment}_seed_{seed}.pkl', 'wb') as f:
    pickle.dump(results, f)