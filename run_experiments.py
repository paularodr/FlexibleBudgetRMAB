import pdb
import os
import pickle
import numpy as np
from environments import dropOutState, immediateRecovery, twoStateProcess
from test_utils import planning
import argparse

parser = argparse.ArgumentParser(description='Experiments')
parser.add_argument('--seed', metavar='seed', type=int,
                    help='Set random seed')
parser.add_argument('--domain', metavar='dom', type=str, choices = ['dropOutState', 'twoStateProcess', 'immediateRecovery'],
                    help='Set domain')
parser.add_argument('--F', metavar='F', type=int,
                    help='Flexible time horizon')
parser.add_argument('--H', metavar='H', type=int,
                    help='Total time horizon')
parser.add_argument('--N', metavar='N', type=int,
                    help='Number of arms', default=10)
parser.add_argument('--B', metavar='B', type=int,
                    help='Per round budget', default=1)
parser.add_argument('--S', metavar='S', type=int,
                    help='Number of states', default=2)
parser.add_argument('--niters', metavar='niters',nargs='*', type=int, default=[50,100,200])
parser.add_argument('--compress', metavar='compress', type=bool, choices=[False,True])
args = parser.parse_args()
 
niters = args.niters
compress = args.compress
pdgs_algos = [f'PDSG-{i}' for i in niters]
heuristics = ['compress_static', 'compress_closing'] if compress else []
algos = ['random', 'hawkins'] + heuristics + pdgs_algos

# Parameters
seed = args.seed
domain = args.domain
F = args.F #flexible time horizon
H = args.H #total time horizon
N = args.N #arms
S = args.S #states
B = float(args.B) #one step budget
C = [0,1]


# chambolle-pock algorithms
gamma = 1
tau = 0.1
sigma = 0.1
x = np.zeros(H+1) # initial lagrange multipliers
y = np.ones(F) # initial budget variables
tolerance=1e-06
sample_size=10*N

HORIZON = int(H/F) #T steps

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
    if domain == 'immediateRecovery':
        envs[algo] = immediateRecovery(N,B,S)
    if domain == 'twoStateProcess':
        envs[algo] = twoStateProcess(N,B)
results['environment'] = envs[algo]

for algo in pdgs_algos:
    results[algo]['runtimes_lp'] = []
    results[algo]['runtimes_sample'] = []
    results[algo]['optgap'] = []

total_runtime = 0
for t in range(HORIZON):
    print(f'timestep: {t}/{HORIZON}')

    # T steps of Hawkins (fixed per round  budget)
    algo = 'hawkins'
    results, envs, random_states = planning.plan_hawkins_single(H,F,N,C,B,envs,algo,gamma,results)

    # randomly choose B at each round
    algo = 'random'
    results, envs = planning.plan_random(F,N,B,envs,algo,results,random_states,seed)

    if compress:
        # Compress with static window
        algo = 'compress_static'
        results, envs = planning.plan_hawkins_fixed(t,H,F,B,N, C, envs, algo, gamma, random_states,results)

        # Compress with closing window
        algo = 'compress_closing'
        results, envs = planning.plan_hawkins_closing(t,H,F,B,N,C,envs,algo,gamma,random_states,results)

    #PDGS: Minmax using Chmbolle-Pock
    for i, algo in enumerate(pdgs_algos):
        niter = niters[i]
        results, envs = planning.plan_chambolle_pock(F,N,B,H,C,t,tau,sigma,envs,algo,niter,tolerance, sample_size, random_states,results)


# results from list to numpy array
for k in ['actions','states','rewards']:
    for d in algos:
        results[d][k] = np.array(results[d][k])

#save results
if domain != 'immediateRecovery':
    experiment = f'F_{F}_H_{H}_N_{N}_B_{int(B)}'
else:
    experiment = f'F_{F}_H_{H}_N_{N}_S_{S}_B_{int(B)}'
dir_path = f'results/{domain}/{experiment}'
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
with open(f'{dir_path}/{experiment}_seed_{seed}.pkl', 'wb') as f:
    pickle.dump(results, f)