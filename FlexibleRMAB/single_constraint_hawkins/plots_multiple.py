import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import rc


rc('font',**{'family':'serif','serif':['Times']})
rc('text', usetex=True)


algos = ['hawkins_single','hawkins_fixed','hawkins_closing','chambolle-pock']

T = [2,3,5,10]
H = 30
N = 10
B = 1

experiments = [f'T_{t}_H_{H}_N_{N}_B_{B}' for t in T]
rewards = pd.DataFrame()
resources = pd.DataFrame()

for t in T:
    exp = f'T_{t}_H_{H}_N_{N}_B_{B}'
    for seed in range(29):
        with open(f'results/dropOutState/closing_window/{exp}/{exp}_seed_{seed}.pkl', 'rb') as f:
            results = pickle.load(f)
        rw = []
        rs = []
        for algo in algos:
             rw.append(results[algo]['rewards'].sum())
             rs.append(results[algo]['actions'].sum(axis=1))
        rw = pd.DataFrame(rw)
        rw['algo'] = algos
        rw['T'] = [t]*len(algos)
        rw['seed'] = [seed]*len(algos)
        rs = pd.DataFrame(rs)
        rs['algo'] = algos
        rs['T'] = [t]*len(algos)
        rs['seed'] = [seed]*len(algos)
        rewards = rewards.append(rw,ignore_index=True)
        resources = resources.append(rs,ignore_index=True)

rewards['algo'] = rewards.algo.str.replace('_','\_')
resources['algo'] = resources.algo.str.replace('_','\_')

flexible = resources.copy()
flexible['flexible'] = (resources.iloc[:,:30]!=1).sum(axis=1)

fig, (ax1, ax2) = plt.subplots(figsize=(10,2.5),ncols=2)

#rewards
g = sns.barplot(
    x="T", 
    y=0, 
    hue="algo", 
    data=rewards, 
    ci="sd",
    palette="rocket",
    estimator=np.mean,
    edgecolor="black",
    ax=ax1
    )
g.legend_.remove()
ax1.set_ylabel("Cumulative reward")
ax1.set_xlabel("Flexible time horizon \n \n (a)")
ax1.set_xticklabels([f'T={t}' for t in T])

# flexibility
g = sns.barplot(
    x="T", 
    y="flexible", 
    hue="algo", 
    data=flexible, 
    ci="sd",
    palette="rocket",
    estimator=np.mean,
    edgecolor='black',
    ax=ax2
    )
g.legend_.remove()
ax2.set_ylabel(f"Num. of steps \n with flexibility (H={H})")
ax2.set_xlabel("Flexible time horizon \n \n (b)")
ax2.set_xticklabels([f'T={t}' for t in T])

lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
fig.legend(lines[:4], labels[:4], loc='upper center',ncol=4)
fig.tight_layout()
plt.subplots_adjust(top=0.8)
plt.show()


# RESORUCES

fig, ax1 = plt.subplots(figsize=(5,2.5))

f = 5
r = resources.loc[resources['T']==f,:].reset_index(drop=True)
counts = pd.DataFrame()
for t in range(f+1):
    x = r.iloc[:,H:]
    x['count'] = (r == t).sum(axis=1)   
    x['resources'] = t
    counts = counts.append(x,ignore_index=True)

g = sns.barplot(
    x="resources", 
    y="count", 
    hue="algo", 
    data=counts, 
    ci="sd",
    palette="rocket",
    estimator=np.mean,
    edgecolor='black',
    ax=ax1
    )
plt.ylim(0,H)
plt.show()