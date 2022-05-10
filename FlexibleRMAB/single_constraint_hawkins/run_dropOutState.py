import pickle
import numpy as np
from tqdm import tqdm 
import hawkins_actions
from environments import dropOutState

gamma = 0.95
N = 5
B = 1.0 #one step budget
start_state = np.array([2]*N)


def append_results(algo, actions, state, reward):
    results[algo]['actions'] = results[algo]['actions'] + [actions]
    results[algo]['states'] = results[algo]['states'] + [state]
    results[algo]['rewards'] = results[algo]['rewards'] + [reward]
    return results

EPOCHS = 100
HORIZON = 2 #double steps
algos = ['fixed_window','closing_window', 'singlestep', 'singlestep_rs']

for epoch in tqdm(range(EPOCHS)):
    results = {}    
    envs = {}
    for algo in algos:
        results[algo] = {'actions':[],'states':[],'rewards':[]}
        envs[algo] = dropOutState(N, B, start_state)


    for t in range(HORIZON):

        # SINGLE STEP (TWO RUNS)
        algo = 'singlestep'
        random_states = []
        for k in range(2):
            actions = hawkins_actions.actions_set_lambda(envs[algo], gamma, steps=1, epsilon=1e-1)
            random_states.append(np.random.get_state())
            np.random.set_state(random_states[k])
            current_state, reward = envs[algo].onestep(actions)
            results = append_results(algo, actions, current_state, reward)

        # FIXED WINDOW (TAKE 2 STEPS)
        algo = 'fixed_window'    
        actions_twostep = hawkins_actions.actions_set_lambda(envs[algo], gamma, steps=2, epsilon=1e-1)
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
        actions_twostep = hawkins_actions.actions_set_lambda(envs[algo], gamma, steps=2, epsilon=1e-1)
        actions_first = np.array([(1 if x==1 or x==3 else 0) for x in actions_twostep])
        actions_second = np.array([(1 if x==2 or x==3 else 0) for x in actions_twostep])

        ## First step of closing window
        np.random.set_state(random_states[0])
        current_state, reward = envs[algo].onestep(actions_first) #take first step
        results = append_results(algo,actions_first, current_state, reward)
        reserve = 2*B - actions_first.sum()

        ## Second step of closing window
        if reserve>0:
            actions_second_window = hawkins_actions.actions_set_lambda_window(envs[algo], gamma, B=reserve, epsilon=1e-1)
        else:
            actions_second_window = np.array([0]*N)

        np.random.set_state(random_states[1])
        current_state, reward = envs[algo].onestep(actions_second_window) #take second step
        results = append_results(algo,actions_second_window, current_state, reward)
 
    # results from list to numpy array
    for k in ['actions','states','rewards']:
        for d in algos:
            results[d][k] = np.array(results[d][k])

    #save results
    with open(f'experiments/dropOutState_v4/epoch_{epoch}.pkl', 'wb') as f:
        pickle.dump(results, f)