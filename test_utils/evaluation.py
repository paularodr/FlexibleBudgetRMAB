import numpy as np
import random

def nextState(actions, state, T):
    S = T.shape[1]

    # perform action on each arm and update state
    newState = state.copy()
    for p in range(len(actions)):
            prob = T[p][state[p]][actions[p]]
            newState[p] = np.random.choice(np.arange(S),p=prob)
    return newState

def getReward(state, R):
    rws = [R[i][state[i]] for i in range(len(state))]
    return sum(rws)

def usedBudget(costs,actions, K):
    used = []
    for i in range(1,K+1):
        used.append(costs[np.where(actions==i)[0]][:,i].sum())
    return used

def averageCumulative(reward, start_state, R):
    acr = [sum(reward[:t+1])/(t+1) for t in range(len(reward))]
    acr = [getReward(start_state, R)] + acr
    return acr

def ACR(rewards, start_state, R):
    return np.apply_along_axis(lambda x: averageCumulative(x,start_state, R),1,rewards)

