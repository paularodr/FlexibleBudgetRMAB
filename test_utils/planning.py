import numpy as np
import time
import math
import tracemalloc
from algos import compressing_methods, minmax_methods, hawkins_actions

def append_results(results, algo, actions, state, reward):
    actions = list(actions)
    state = list(state)
    results[algo]['actions'] = results[algo]['actions'] + [actions]
    results[algo]['states'] = results[algo]['states'] + [state]
    results[algo]['rewards'] = results[algo]['rewards'] + [reward]
    return results

def plan_chambolle_pock(T,N,B,H,C,t,tau,sigma,envs,algo,niter,tolerance, sample_size, random_states,results):
    P = envs[algo].T
    R = envs[algo].R
    used = 0
    start = time.time()
    for i in range(T):
        size_close = T - i
        budget = B*T - used
        time_horizon = H - t -i
        K = minmax_methods.createK(size_close,time_horizon)
        if budget > 0:
            x = np.zeros(time_horizon+1)
            y = np.ones(size_close)
            actions, l, budgets, Q_vals = minmax_methods.chambolle_pock_actions(tau, sigma, K, x, y, envs[algo].current_state, P, R, C, B, budget,  niter, tolerance, sample_size)
        else:
            actions = np.array([0]*N)
        used += actions.sum()
        np.random.set_state(random_states[i])
        current_state, reward = envs[algo].onestep(actions)
        results = append_results(results, algo, actions, current_state, reward)
    runtime = (time.time()-start)
    results[algo]['runtime'] += runtime
    return results, envs

def plan_hawkins_closing(step,H,T, B, N, C, envs, algo, gamma, random_states,results,finite_horizon):
    P = envs[algo].T
    R = envs[algo].R
    used=0
    start = time.time()
    tracemalloc.start()
    for i in range(T):
        size_close = T - i
        horizon = math.floor((H-step*T-i)/size_close)
        budget = B*T - used 
        if budget > 0:
            actions = compressing_methods.hawkins_window(horizon,size_close, N, P, R, C, budget, envs[algo].current_state, gamma,finite_horizon)
        else:
            actions = [np.array([0]*N)]
        used += actions[0].sum()
        np.random.set_state(random_states[i])
        current_state, reward = envs[algo].onestep(actions[0])
        results = append_results(results, algo, actions[0], current_state, reward)
    runtime = (time.time()-start)
    results[algo]['runtime'] += runtime
    results[algo]['memory'] += tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return results, envs

def plan_hawkins_fixed(step,H,T,B,N, C, envs, algo, gamma, random_states,results,finite_horizon):
    horizon = math.floor(H/T) - step

    P = envs[algo].T
    R = envs[algo].R
    start = time.time()
    tracemalloc.start()
    actions = compressing_methods.hawkins_window(horizon, T, N, P, R, C, T*B, envs[algo].current_state, gamma,finite_horizon)
    states, rewards = envs[algo].multiple_steps(T, actions, random_states)
    for i in range(T):
        results = append_results(results, algo, actions[i], states[i], rewards[i])
    runtime = (time.time()-start)
    results[algo]['runtime'] += runtime
    results[algo]['memory'] += tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return results, envs

def plan_hawkins_single(H,T,N,C,B,envs,algo,gamma,results,finite_horizon):
    random_states = []
    P = envs[algo].T
    R = envs[algo].R
    start = time.time()
    tracemalloc.start()
    for k in range(T):
            output = hawkins_actions.get_hawkins_actions(H, N, P, R, C, B, envs[algo].current_state, gamma,finite_horizon)
            actions = output[0]
            random_states.append(np.random.get_state())
            np.random.set_state(random_states[k])
            current_state, reward = envs[algo].onestep(actions)
            results = append_results(results, algo, actions, current_state, reward)
    runtime = (time.time()-start)
    results[algo]['runtime'] += runtime
    results[algo]['memory'] += tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return results, envs, random_states
