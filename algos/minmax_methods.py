import test_utils.evaluation as evaluation
import numpy as np
import test_utils.evaluation as evaluation
from gurobipy import *
from tqdm import tqdm
from pathos.pools import ProcessPool
#from numba import njit, prange, jit

def L_fixed_lambda(P, R, C, H, lamtime, start_state, gamma=0.95):
    NPROCS = P.shape[0]
    NSTATES = P.shape[1]
    NACTIONS = P.shape[2]

    # define parameter mu: one hot encoder of current state
    mu = np.zeros((NPROCS,NSTATES),dtype=object)
    for i in range(NPROCS):
        mu[i, int(start_state[i])] = 1


    # Create a new model
    m = Model("LP for Hawkins Lagrangian relaxation")
    m.setParam( 'OutputFlag', False )

    L = np.zeros((H+1,NPROCS,NSTATES),dtype=object)
    for h in range(H):
        for p in range(NPROCS):
            for i in range(NSTATES):
                    L[h,p,i] = m.addVar(vtype=GRB.CONTINUOUS, name='L_%s_%s_%s'%(h,p,i))

    # model objective
    m.modelSense=GRB.MINIMIZE
    m.setObjective(sum([L[0,i,:].dot(mu[i]) for i in range(NPROCS)]))

    # set constraints
    for h in range(H):
        for p in range(NPROCS):
            for i in range(NSTATES):
                for j in range(NACTIONS):                
                    m.addConstr( L[h,p,i] >= R[p,i] - lamtime[h]*C[j] + gamma*(L[h+1,p,:]).dot(P[p,i,j]))

    # Optimize model
    m.optimize()

    #get values
    L_vals = np.zeros((H+1,NPROCS,NSTATES))
    for v in m.getVars():
        if 'L' in v.varName:
            t = int(v.varName.split('_')[1])
            i = int(v.varName.split('_')[2])
            j = int(v.varName.split('_')[3])
            L_vals[t,i,j] = v.x

    return L_vals

def get_Q_vals(H, P, R, C, lamtime, current_state, gamma=0.95):
    # Q_vals: ndarray of size HxNxAxS
    N = P.shape[0]

    # solve value function L for fixed lambdas
    L_vals = L_fixed_lambda(P, R, C, H, lamtime, current_state, gamma=gamma)

    # N x A x S matrix
    Q_vals = np.zeros((H, N, P.shape[2], P.shape[1]))

    for t in range(H):
        for i in range(N):
            for a in range(P.shape[2]):
                for s in range(P.shape[1]):
                        Q_vals[t,i,a,s] = R[i,s] - C[a]*lamtime[t] + gamma*L_vals[t+1,i,:].dot(P[i,s,a])
    
    return Q_vals

def get_Q_actions(Q_vals, timestep, current_state):
    N = Q_vals.shape[1]
    A = Q_vals.shape[2]
    actions = np.zeros(N, dtype=int)

    # take action with higher Q val
    for i in range(N):
        s = current_state[i]
        actions[i] = np.argmax(Q_vals[timestep,i,:,s])

    return actions

def cost_one_sample(start_state, lamtime, H, P, R, C, gamma=0.95):
    cost_time = []
    current_state = start_state
    Q_vals = get_Q_vals(H, P, R, C, lamtime, start_state, gamma)

    for t in range(H):
        actions = get_Q_actions(Q_vals, t, current_state)
        total_cost = np.sum([C[int(i)] for i in actions])
        cost_time.append(total_cost)

        # take actions and transition
        current_state = evaluation.nextState(actions, current_state, P)
    return cost_time

# @jit
# def sample_expected_cost_jit(sample_size, start_state, lamtime, H, P, S, R, C, gamma=0.95):
#     costs = []
#     for _ in range(sample_size):
#         cost_time = []
#         current_state = start_state
#         Q_vals = get_Q_vals(H, P, R, C, lamtime, start_state, gamma)

#         for t in range(H):
#             actions = get_Q_actions(Q_vals, t, current_state)
#             total_cost = np.sum([C[int(i)] for i in actions])
#             cost_time.append(total_cost)

#             # take actions and transition
#             current_state = evaluation.nextState(actions, current_state, S, P)
#         costs.append(cost_time)
    
#     costs = np.array(costs)
#     costs = np.mean(costs,axis=0)
#     return costs

def sample_expected_cost(sample_size, start_state, lamtime, H, P, R, C, gamma=0.95):
    costs = []
    for _ in range(sample_size):
        cost_time = cost_one_sample(start_state, lamtime, H, P, R, C, gamma)
        costs.append(cost_time)
    
    costs = np.array(costs)
    costs = np.mean(costs,axis=0)
    return costs

def sample_expected_cost_parall(sample_size, start_state, lamtime, H, P, R, C, gamma=0.95):
    pool = ProcessPool()
    results = pool.map(
        cost_one_sample,
        [start_state for _ in range(sample_size)],
        [lamtime for _ in range(sample_size)],
        [H for _ in range(sample_size)],
        [P for _ in range(sample_size)],
        [R for _ in range(sample_size)],
        [C for _ in range(sample_size)],
        [gamma for _ in range(sample_size)]
    )
    results = np.array(results)
    mean_cost = np.mean(results,axis=0)
    return mean_cost

def LP_fixed_b(H, T, P, R, C, B, bvals, start_state, lambda_lim=None, gamma=0.95):
    #bvals: list of length H

    N = P.shape[0]
    S = P.shape[1]
    A = P.shape[2]

	# Create a new model
    m = Model("LP for Hawkins Lagrangian relaxation with flexible budget (given)")
    m.setParam('OutputFlag', False )
    b = np.array(bvals)
	
    mu = np.zeros((N,S),dtype=object)
    for i in range(N):
        mu[i, int(start_state[i])] = 1

    lamtime = np.zeros(H,dtype=object)
    for h in range(H):
        lamtime[h] = m.addVar(vtype=GRB.CONTINUOUS, name='lambda_%s'%(h), lb=0)

    lagrange_mu = m.addVar(vtype=GRB.CONTINUOUS, name='lagrange_mu', lb=0)

    L = np.zeros((N,S, H+1),dtype=object)
    for p in range(N):
        for i in range(S):
            for h in range(H):
                L[p,i,h] = m.addVar(vtype=GRB.CONTINUOUS, name='L_%s_%s_%s'%(p,i,h))

    # model objective
    m.modelSense=GRB.MINIMIZE
    m.setObjective(sum([L[i,:,0].dot(mu[i]) for i in range(N)]) + b.dot(lamtime) + lagrange_mu*(b.sum()-T*B))

    # set constraints
    for p in range(N):
        for i in range(S):
            for j in range(A):
                for h in range(H):
                    m.addConstr( L[p,i,h] >= R[p,i] - lamtime[h]*C[j] + gamma*(L[p,:,h+1]).dot(P[p,i,j]))
					

	# Optimize model
    m.optimize()

    lambda_vals = np.zeros(H)

    for v in m.getVars():
        if 'lambda' in v.varName:
            i = int(v.varName.split('_')[1])
            lambda_vals[i] = v.x

    obj = m.getObjective()

    return lambda_vals, obj.getValue()

def LP_fixed_lagrange(H, T, P, R, C, B, lagrange_vals, start_state, gamma=0.95):
    #bvals: list of length H

    N = P.shape[0]
    S = P.shape[1]
    A = P.shape[2]

	# Create a new model
    m = Model("LP with given lambdas and variable budget")
    m.setParam('OutputFlag', False )
    lamtime = np.array(lagrange_vals[:-1])
    lagrange_mu = lagrange_vals[-1]
	
    mu = np.zeros((N,S),dtype=object)
    for i in range(N):
        mu[i, int(start_state[i])] = 1

    b = np.zeros(H,dtype=object)
    for t in range(T):
        b[t] = m.addVar(vtype=GRB.CONTINUOUS, name='bt_%s'%(t), lb=0)
    if H>T:
        for t in range(T,H):
            b[t] = B

    L = np.zeros((N,S, H+1),dtype=object)
    for p in range(N):
        for i in range(S):
            for h in range(H):
                L[p,i,h] = m.addVar(vtype=GRB.CONTINUOUS, name='L_%s_%s_%s'%(p,i,h))

    # model objective
    m.modelSense=GRB.MINIMIZE
    m.setObjective(sum([L[i,:,0].dot(mu[i]) for i in range(N)]) + b.dot(lamtime) + lagrange_mu*(b.sum()-T*B))

    # set constraints
    for p in range(N):
        for i in range(S):
            for j in range(A):
                for h in range(H):
                    m.addConstr( L[p,i,h] >= R[p,i] - lamtime[h]*C[j] + gamma*(L[p,:,h+1]).dot(P[p,i,j]))

    # set budget constraint
    #m.addConstr(sum(b[:T]) == T*B)					

	# Optimize model
    m.optimize()

    b_vals = np.zeros(T)

    for v in m.getVars():
        if 'bt' in v.varName:
            i = int(v.varName.split('_')[1])
            b_vals[i] = v.x

    obj = m.getObjective()

    return b_vals, obj.getValue()

def chambolle_pock(tau, sigma, K, x, y, start_state, P, R, C, B, Bavai,  n_iter=50, tolerance=1e-06, sample_size=100,gamma=0.95):
# min_x max_y (x: lagrange multipliers, y: budget variables)
# x0: list with initial values for variabels to minimize over
# y0: list with initial values for variabels to maximize over
# K: ndarray of size len(y0) x len(x0)
# we want to have tau and sigma such that: tau*sigma*T<1
    T = K.shape[0]
    H = K.shape[1] - 1
    ytol = False
    xtol = False
    xbar = x

    xdiffs = []
    ydiffs = []
    #obj_diffs = []
    for _ in tqdm(range(n_iter)):
        diff_y = sigma*K.dot(xbar)
        ydiffs.append(np.abs(diff_y))
        if np.all(np.abs(diff_y) <= tolerance):
            ytol = True
        y += diff_y
        x_grad = x - tau*K.T.dot(y)
        lambda_vals = x_grad[:-1]

        #sample gradients evaluated in x_grad
        expected_cost = sample_expected_cost_parall(sample_size, start_state, lambda_vals, H, P, R, C, gamma)
        gradients = -1 * expected_cost
        if T<H:
            gradients[T+1:] += B
        gradients = np.append(gradients,Bavai)
        diff_x = -1*tau*(K.T.dot(y)+gradients)
        xdiffs.append(np.abs(diff_x))
        if np.all(np.abs(diff_x) <= tolerance):
            xtol = True
        x += diff_x
        xbar = x + diff_x
        if ytol and xtol:
            break
        #if T<H:  
        #    bvals = np.array(list(y) + [B]*(H-T))
        #else:
        #    bvals = y 
        #lagranges, min_val = LP_fixed_b(H, T, P, R, C, B, bvals, start_state)
        #budgets, max_val = LP_fixed_lagrange(H, T, P, R, C, B, x, start_state)
        #diff_obj = np.abs(max_val - min_val)
        #obj_diffs.append(diff_obj)
        #if diff_obj < tolerance:
        #    break

    return (x,xdiffs),(y,ydiffs)


def action_knapsack(values, C, b):
    # values: ndarray of size HxNxA
    # budgets: list of length H


    m = Model("Knapsack with budget per time step")
    m.setParam( 'OutputFlag', False )
    
    x = np.zeros(values.shape, dtype=object)
    
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i,j] = m.addVar(vtype=GRB.BINARY, name='x_%i_%i'%(i,j))

    m.modelSense=GRB.MAXIMIZE
    m.setObjective((x*values).sum())

    # budget constraint
    m.addConstr(x.dot(C).sum() == b)

    # assign one action per arm (including action 0, not acting)
    for i in range(values.shape[0]):
        m.addConstr(x[i].sum() == 1)

    # Optimize model
    m.optimize()

    # solved values
    x_out = np.zeros(x.shape)
    for v in m.getVars():
        if 'x' in v.varName:
            i = int(v.varName.split('_')[1])
            j = int(v.varName.split('_')[2])
            x_out[i,j] = v.x
        else:
            pass
    
    return x_out

def chambolle_pock_actions(tau, sigma, K, x, y, current_state, P, R, C, B, Bavai,  n_iter, tolerance, sample_size):
    (l,l_diff),(b,b_diff) = chambolle_pock(tau, sigma, K, x, y, current_state, P, R, C, B, Bavai, n_iter, tolerance, sample_size)
    lamtime = l[:-1]
    T = K.shape[0]
    H = K.shape[1] - 1
    N = P.shape[0]

    # Q vals for current state
    Q_vals = get_Q_vals(H, P, R, C, lamtime, current_state)
    Q_vals_state = np.zeros(Q_vals.shape[1:-1], dtype=object)
    for i in range(N):
        s = current_state[i]
        Q_vals_state[i,:] = Q_vals[0,i,:,s]

    # round budget variabels and append fixed budget for t>T
    budgets = np.round(b)
    budgets[budgets<0] = 0
    budgets = np.abs(budgets)
    if T<H:
        budgets = np.array(list(budgets) + [B]*(H-T))

    step_budget = np.min([Bavai,budgets[0]])
    actions_var = action_knapsack(Q_vals_state, C, step_budget)
    actions = np.argmax(actions_var,axis=1)
    
    return actions, l, budgets, Q_vals

# Define K matrix
def createK(T,H):
    K = np.zeros((T,H+1))
    for t in range(T):
        K[t,t] = 1
    K[:,H] = [-1]*T
    return K
