from gurobipy import *
import numpy as np 
import time


# https://dspace.mit.edu/handle/1721.1/29599
def hawkins(T, R, C, B, start_state, lambda_lim=None, gamma=0.95):

	start = time.time()

	NPROCS = T.shape[0]
	NSTATES = T.shape[1]
	NACTIONS = T.shape[2]

	# Create a new model
	m = Model("LP for Hawkins Lagrangian relaxation")
	m.setParam( 'OutputFlag', False )

	L = np.zeros((NPROCS,NSTATES),dtype=object)
	
	mu = np.zeros((NPROCS,NSTATES),dtype=object)
	for i in range(NPROCS):
		# mu[i] = np.random.dirichlet(np.ones(NSTATES))
		mu[i, int(start_state[i])] = 1

	c = C

	# Create variables
	lb = 0
	ub = GRB.INFINITY
	if lambda_lim is not None:
		ub = lambda_lim

	lam = m.addVar(vtype=GRB.CONTINUOUS, lb=lb, ub=ub, name='lambda')


	for p in range(NPROCS):
		for i in range(NSTATES):
			L[p,i] = m.addVar(vtype=GRB.CONTINUOUS, name='L_%s_%s'%(p,i))


	L = np.array(L)


	# print('Variables added in %ss:'%(time.time() - start))
	start = time.time()


	m.modelSense=GRB.MINIMIZE

	# Set objective
	# m.setObjectiveN(obj, index, priority) -- larger priority = optimize first
	# minimze the value function

	# In Hawkins, only min the value function of the start state
	# print(current_state)
	# m.setObjectiveN(sum([L[i][current_state[i]] for i in range(NPROCS)]) + index_variable*B*((1-gamma)**-1), 0, 1)

	m.setObjectiveN(sum([L[n].dot(mu[n]) for n in range(NPROCS)]) + lam*B*((1-gamma)**-1), 0, 1)

	# set constraints
	for n in range(NPROCS):
		for s in range(NSTATES):
			for j in range(NACTIONS):
				m.addConstr( L[n,s] >= R[n,s] - lam*c[j] + gamma*L[n].dot(T[n,s,j]) )
				# m.addConstr( L[p][i] >= R[p][i] - lam*c[j] + gamma*L[p].dot(T[p,i,j]) )
				# m.addConstr( L[p,i] >= R[p,i] - lam*c[j] + gamma*T[p,i,j].dot(L[p]))



	# print('Constraints added in %ss:'%(time.time() - start))
	start = time.time()

	# Optimize model

	m.optimize()
	# m.printStats()

	# print('Model optimized in %ss:'%(time.time() - start))
	start = time.time()


	L_vals = np.zeros((NPROCS,NSTATES))

	lam_solved_value = 0
	for v in m.getVars():
		if 'lambda' in v.varName:
			lam_solved_value = v.x

		if 'L' in v.varName:
			i = int(v.varName.split('_')[1])
			j = int(v.varName.split('_')[2])

			L_vals[i,j] = v.x

	# print('Variables extracted in %ss:'%(time.time() - start))
	start = time.time()

	return L_vals, lam_solved_value

def hawkins_fixed_bt(H, T, P, R, C, B, bvals, start_state, lambda_lim=None, gamma=0.95):
	#bts: list of length T

	start = time.time()

	NPROCS = P.shape[0]
	NSTATES = P.shape[1]
	NACTIONS = P.shape[2]

	# Create a new model
	m = Model("LP for Hawkins Lagrangian relaxation with flexible budget")
	m.setParam('OutputFlag', False )

	L = np.zeros((NPROCS,NSTATES, H+1),dtype=object)
	
	mu = np.zeros((NPROCS,NSTATES),dtype=object)
	for i in range(NPROCS):
		mu[i, int(start_state[i])] = 1

	c = C

	# Create variables
	lb = 0
	ub = GRB.INFINITY
	if lambda_lim is not None:
		ub = lambda_lim

	lamtime = np.zeros(H,dtype=object)
	for h in range(H):
		lamtime[h] = m.addVar(vtype=GRB.CONTINUOUS, name='lambda_%s'%(h), lb=0)

	b = np.zeros(H,dtype=object)
	for t in range(T):
		b[t] = m.addVar(vtype=GRB.CONTINUOUS, name='bt_%s'%(t), lb=0)
	if H>T:
		for t in range(T,H):
			b[t] = B

	lam_x_b = np.zeros(T,dtype=object)
	for t in range(T):
		lam_x_b[t] = m.addVar(vtype=GRB.CONTINUOUS, name='lam_x_b_%s'%(t), lb=0)

	for p in range(NPROCS):
		for i in range(NSTATES):
			for h in range(H):
				L[p,i,h] = m.addVar(vtype=GRB.CONTINUOUS, name='L_%s_%s_%s'%(p,i,h))


	# print('Variables added in %ss:'%(time.time() - start))
	start = time.time()


	m.modelSense=GRB.MINIMIZE

	m.setObjective(sum([L[i,:,0].dot(mu[i]) for i in range(NPROCS)]) + b.dot(lamtime))

	for t in range(T):
		m.addConstr(b[t] == bvals[t]) #set fixed value for b
	
	# set constraints
	for p in range(NPROCS):
		for i in range(NSTATES):
			for j in range(NACTIONS):
				for h in range(H):
					m.addConstr( L[p,i,h] >= R[p,i] - lamtime[h]*c[j] + gamma*(L[p,:,h+1]).dot(P[p,i,j]))
					#m.addConstr( L[p,i,h] >= - lamtime[h]*c[j] + gamma*(L[p,:,h+1]+R[p,:]).dot(P[p,i,j]) ) #collect reward after taking action

	# set budget constraint
	m.addConstr(sum(b[:T]) == T*B)

	# print('Constraints added in %ss:'%(time.time() - start))
	start = time.time()

	# Optimize model

	m.optimize()
	# m.printStats()

	# print('Model optimized in %ss:'%(time.time() - start))
	start = time.time()


	L_vals = np.zeros((NPROCS,NSTATES,H))
	b_vals = np.zeros(T)
	lambda_vals = np.zeros(H)

	for v in m.getVars():
		if 'lambda' in v.varName:
			i = int(v.varName.split('_')[1])
			lambda_vals[i] = v.x
		
		if 'bt' in v.varName:
			i = int(v.varName.split('_')[1])
			b_vals[i] = v.x


		if 'L' in v.varName:
			i = int(v.varName.split('_')[1])
			j = int(v.varName.split('_')[2])
			t = int(v.varName.split('_')[3])

			L_vals[i,j,t] = v.x

	start = time.time()

	return L_vals, lambda_vals, b_vals, m

def hawkins_flexible(H, T, P, R, C, B, start_state, lambda_lim=None, gamma=0.95):

	# B is the one step budget
	# T timesteps to respect budget contraing (TB)
	# H time horizon
	# 1 constraint (sum_0^T b_t = T*B, b_t for 0<t<T are variable) (second option: H/T constraints, all b_t 0<t<H are variable )
	# b_t = b for T<t<H


	start = time.time()

	NPROCS = P.shape[0]
	NSTATES = P.shape[1]
	NACTIONS = P.shape[2]

	# Create a new model
	m = Model("LP for Hawkins Lagrangian relaxation with flexible budget")
	m.setParam('OutputFlag', False )

	L = np.zeros((NPROCS,NSTATES, H+1),dtype=object)
	
	mu = np.zeros((NPROCS,NSTATES),dtype=object)
	for i in range(NPROCS):
		mu[i, int(start_state[i])] = 1

	c = C

	# Create variables
	lb = 0
	ub = GRB.INFINITY
	if lambda_lim is not None:
		ub = lambda_lim

	lamtime = np.zeros(H,dtype=object)
	for h in range(H):
		lamtime[h] = m.addVar(vtype=GRB.CONTINUOUS, name='lambda_%s'%(h), lb=0)

	b = np.zeros(H,dtype=object)
	for t in range(T):
		b[t] = m.addVar(vtype=GRB.CONTINUOUS, name='bt_%s'%(t), lb=0)
	if H>T:
		for t in range(T,H):
			b[t] = B

	lam_x_b = np.zeros(T,dtype=object)
	for t in range(T):
		lam_x_b[t] = m.addVar(vtype=GRB.CONTINUOUS, name='lam_x_b_%s'%(t), lb=0)

	z = m.addVar(vtype=GRB.CONTINUOUS, name='min_obj')

	for p in range(NPROCS):
		for i in range(NSTATES):
			for h in range(H):
				L[p,i,h] = m.addVar(vtype=GRB.CONTINUOUS, name='L_%s_%s_%s'%(p,i,h))


	# print('Variables added in %ss:'%(time.time() - start))
	start = time.time()


	m.modelSense=GRB.MINIMIZE

	m.setObjective(z)
	m.params.NonConvex = 2

	fixedb = [1,1]
	for t in range(T):
		m.addConstr(b[t] == fixedb[t]) #set fixed value for b

	m.addConstr(z >= sum([L[i,:,0].dot(mu[i]) for i in range(NPROCS)]) + b.dot(lamtime))

#	for bvals in [[1,1],[2,0],[0,2]]:
#		bvals = np.array(bvals)
#		m.addConstr(z >= sum([L[i,:,0].dot(mu[i]) for i in range(NPROCS)]) + bvals.dot(lamtime))
	#m.addConstr(sum([L[i,:,0].dot(mu[i]) for i in range(NPROCS)]) + b.dot(lamtime),0,1)
	

	# set lambda*b variable as constraint
	#for t in range(T):
	#	m.addConstr(lam_b[t] == b[t]*lamtime[t])

	# constraint to capture minimization over lambdas
	#m.addConstr(z >= sum([L[i,:,0].dot(mu[i]) for i in range(NPROCS)]) + b.dot(lamtime))

	# set constraints
	for p in range(NPROCS):
		for i in range(NSTATES):
			for j in range(NACTIONS):
				for h in range(H):
					m.addConstr( L[p,i,h] >= - lamtime[h]*c[j] + gamma*(L[p,:,h+1]+R[p,:]).dot(P[p,i,j]) )

	# set budget constraint
	m.addConstr(sum(b[:T]) == T*B)

	# print('Constraints added in %ss:'%(time.time() - start))
	start = time.time()

	# Optimize model

	m.optimize()
	# m.printStats()

	# print('Model optimized in %ss:'%(time.time() - start))
	start = time.time()


	L_vals = np.zeros((NPROCS,NSTATES,H))
	b_vals = np.zeros(T)
	lambda_vals = np.zeros(H)

	for v in m.getVars():
		if 'lambda' in v.varName:
			i = int(v.varName.split('_')[1])
			lambda_vals[i] = v.x
		
		if 'bt' in v.varName:
			i = int(v.varName.split('_')[1])
			b_vals[i] = v.x


		if 'L' in v.varName:
			i = int(v.varName.split('_')[1])
			j = int(v.varName.split('_')[2])
			t = int(v.varName.split('_')[3])

			L_vals[i,j,t] = v.x

	start = time.time()

	return L_vals, lambda_vals, b_vals, m


# https://dspace.mit.edu/handle/1721.1/29599
def hawkins_set_lambda(T, R, C, B, start_state, lambda_val=None, lambda_lim=None, gamma=0.95):

	start = time.time()

	NPROCS = T.shape[0]
	NSTATES = T.shape[1]
	NACTIONS = T.shape[2]

	# Create a new model
	m = Model("LP for Hawkins Lagrangian relaxation")
	m.setParam( 'OutputFlag', False )

	L = np.zeros((NPROCS,NSTATES),dtype=object)
	
	mu = np.zeros((NPROCS,NSTATES),dtype=object)
	for i in range(NPROCS):
		# mu[i] = np.random.dirichlet(np.ones(NSTATES))
		# mu[i, int(start_state[i])] = 1
		mu[i] = np.ones(NSTATES)/NSTATES

	c = C

	# Create variables
	lb = -GRB.INFINITY
	ub = GRB.INFINITY
	if lambda_lim is not None:
		ub = lambda_lim

	index_variable = m.addVar(vtype=GRB.CONTINUOUS, lb=lb, ub=ub, name='index')


	for p in range(NPROCS):
		for i in range(NSTATES):
			L[p,i] = m.addVar(vtype=GRB.CONTINUOUS, lb=lb, ub=ub, name='L_%s_%s'%(p,i))


	L = np.array(L)


	# print('Variables added in %ss:'%(time.time() - start))
	start = time.time()


	m.modelSense=GRB.MINIMIZE

	# Set objective
	# m.setObjectiveN(obj, index, priority) -- larger priority = optimize first
	# minimze the value function

	# In Hawkins, only min the value function of the start state
	# print(current_state)
	# m.setObjectiveN(sum([L[i][current_state[i]] for i in range(NPROCS)]) + index_variable*B*((1-gamma)**-1), 0, 1)

	# m.setObjectiveN(sum([L[i].dot(mu[i]) for i in range(NPROCS)]) + index_variable*B*((1-gamma)**-1), 0, 1)
	m.setObjectiveN(sum([L[i].dot(mu[i]) for i in range(NPROCS)]), 0, 1)

	m.addConstr(index_variable==lambda_val)

	# set constraints
	for p in range(NPROCS):
		for i in range(NSTATES):
			for j in range(NACTIONS):
				# m.addConstr( L[p][i] >= R[p][i] - index_variable*c[j] + gamma*L[p].dot(T[p,i,j]) )
				m.addConstr( L[p][i] >= R[p][i] - index_variable*c[j] + gamma*LinExpr(T[p,i,j], L[p])) 



	# print('Constraints added in %ss:'%(time.time() - start))
	start = time.time()

	# Optimize model

	m.optimize()
	# m.printStats()

	# print('Model optimized in %ss:'%(time.time() - start))
	start = time.time()


	L_vals = np.zeros((NPROCS,NSTATES))

	index_solved_value = 0
	for v in m.getVars():
		if 'index' in v.varName:
			index_solved_value = v.x

		if 'L' in v.varName:
			i = int(v.varName.split('_')[1])
			j = int(v.varName.split('_')[2])

			L_vals[i,j] = v.x

	# print('Variables extracted in %ss:'%(time.time() - start))
	start = time.time()

	return L_vals, index_solved_value






# Transition matrix, reward vector, action cost vector
def action_knapsack(values, C, B):


	m = Model("Knapsack")
	m.setParam( 'OutputFlag', False )

	c = C

	x = np.zeros(values.shape, dtype=object)

	for i in range(x.shape[0]):
		for j in range(x.shape[1]):
			x[i,j] = m.addVar(vtype=GRB.BINARY, name='x_%i_%i'%(i,j))



	m.modelSense=GRB.MAXIMIZE

	# Set objective
	# m.setObjectiveN(obj, index, priority) -- larger priority = optimize first

	# minimze the value function
	m.setObjectiveN((x*values).sum(), 0, 1)

	# set constraints

	m.addConstr( x.dot(C).sum() == B )
	for i in range(values.shape[0]):
		# m.addConstr( x[i].sum() <= 1 )
		m.addConstr( x[i].sum() == 1 )


	# Optimize model

	m.optimize()

	x_out = np.zeros(x.shape)

	for v in m.getVars():
		if 'x' in v.varName:
			i = int(v.varName.split('_')[1])
			j = int(v.varName.split('_')[2])

			x_out[i,j] = v.x

		else:
			pass
			# print((v.varName, v.x))


	return x_out




