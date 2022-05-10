import pickle
import numpy as np
from tqdm import tqdm 
import hawkins_actions
from environments import dropOutState

gamma = 0.95
N = 10
B = 4.0 #two step budget
start_state = np.array([2]*N)

def actions_set_lambda(env, gamma, steps, epsilon=1e-1):
    if steps == 2:
        actions_hawkins, L, Q, lambda_val, Q_vals_per_state = hawkins_actions.get_hawkins_actions(env.N, env.T_two, env.R, env.C, env.B, env.current_state, gamma)
        new_lambda = lambda_val-epsilon
        actions_lambda, L_vals, Q_vals, new_lambda, Q_vals_per_state = hawkins_actions.get_hawkins_actions_preset_lambda(env.N, env.T_two, env.R, env.C, env.B, env.current_state, lambda_val=new_lambda, gamma=gamma)
    if steps == 1:
        actions_hawkins, L, Q, lambda_val, Q_vals_per_state = hawkins_actions.get_hawkins_actions(env.N, env.T_one, env.R, [0,1], env.B/2, env.current_state, gamma)
        new_lambda = lambda_val-epsilon
        actions_lambda, L_vals, Q_vals, new_lambda, Q_vals_per_state = hawkins_actions.get_hawkins_actions_preset_lambda(env.N, env.T_one, env.R, [0,1], env.B/2, env.current_state, lambda_val=new_lambda, gamma=gamma)
    return actions_lambda

def actions_set_lambda_window(env, gamma, B, epsilon=1e-1):
    actions_hawkins, L, Q, lambda_val, Q_vals_per_state = hawkins_actions.get_hawkins_actions(env.N, env.T_one, env.R, [0,1], B, env.current_state, gamma)
    new_lambda = lambda_val-epsilon
    actions_lambda, L_vals, Q_vals, new_lambda, Q_vals_per_state = hawkins_actions.get_hawkins_actions_preset_lambda(env.N, env.T_one, env.R, [0,1], B, env.current_state, lambda_val=new_lambda, gamma=gamma)
    return actions_lambda


EPOCHS = 200
HORIZON = 25 #double steps

for epoch in tqdm(range(EPOCHS)):
    onestep_env = dropOutState(N, B, start_state)
    twostep_env = dropOutState(N, B, start_state)
    twostepwd_env = dropOutState(N, B, start_state)

    twostep = {'actions':[],'states':[],'rewards':[]}
    onestep = {'actions':[],'states':[],'rewards':[]}
    twostepwd = {'actions':[],'states':[],'rewards':[]}

    for t in range(HORIZON):

        # ONE-STEP TWO TIMES
        random_states = []
        for k in range(2):
            actions_onestep = actions_set_lambda(onestep_env, gamma, steps=1, epsilon=1e-1)
            random_states.append(np.random.get_state())
            np.random.set_state(random_states[k])
            current_state, reward = onestep_env.onestep(actions_onestep)
            onestep['actions'] = onestep['actions'] + [actions_onestep]
            onestep['states'] = onestep['states'] + [current_state]
            onestep['rewards'] = onestep['rewards'] + [reward]

        #TWO-STEP        
        actions_twostep = actions_set_lambda(twostep_env, gamma, steps=2, epsilon=1e-1)
        actions_first = np.array([(1 if x==1 or x==3 else 0) for x in actions_twostep])
        actions_second = np.array([(1 if x==2 or x==3 else 0) for x in actions_twostep])
        
        np.random.set_state(random_states[0])
        current_state, reward = twostep_env.onestep(actions_first) #take first step
        twostep['actions'] = twostep['actions'] + [actions_first]
        twostep['states'] = twostep['states'] + [current_state]
        twostep['rewards'] = twostep['rewards'] + [reward]

        np.random.set_state(random_states[1])
        current_state, reward = twostep_env.onestep(actions_second) #take second step
        twostep['actions'] = twostep['actions'] + [actions_second]
        twostep['states'] = twostep['states'] + [current_state]
        twostep['rewards'] = twostep['rewards'] + [reward]

        #TWO-STEP WINDOW      
        actions_twostep = actions_set_lambda(twostepwd_env, gamma, steps=2, epsilon=1e-1)
        actions_first = np.array([(1 if x==1 or x==3 else 0) for x in actions_twostep])
        actions_second = np.array([(1 if x==2 or x==3 else 0) for x in actions_twostep])
          
        np.random.set_state(random_states[0])
        current_state, reward = twostepwd_env.onestep(actions_first) #take first step
        twostepwd['actions'] = twostepwd['actions'] + [actions_first]
        twostepwd['states'] = twostepwd['states'] + [current_state]
        twostepwd['rewards'] = twostepwd['rewards'] + [reward]

        reserve = B - actions_first.sum()
        if reserve>0:
            actions_second_window = actions_set_lambda_window(twostepwd_env, gamma, B=reserve, epsilon=1e-1)
        else:
            actions_second_window = np.array([0]*N)

        np.random.set_state(random_states[1])
        current_state, reward = twostepwd_env.onestep(actions_second_window) #take second step
        twostepwd['actions'] = twostepwd['actions'] + [actions_second_window]
        twostepwd['states'] = twostepwd['states'] + [current_state]
        twostepwd['rewards'] = twostepwd['rewards'] + [reward]

    for k in twostep.keys():
        twostep[k] = np.array(twostep[k])
        onestep[k] = np.array(onestep[k])
        twostepwd[k] = np.array(twostepwd[k])

    print(f'One step reward: {onestep["rewards"].sum()}')
    print(f'Two step reward: {twostep["rewards"].sum()}')
    print(f'Two step window reward: {twostepwd["rewards"].sum()}')

    results = {'onestep':onestep, 'twostep':twostep, 'twostepwd':twostepwd}
    with open(f'experiments/dropOutState_v3/epoch_{epoch}.pkl', 'wb') as f:
        pickle.dump(results, f)