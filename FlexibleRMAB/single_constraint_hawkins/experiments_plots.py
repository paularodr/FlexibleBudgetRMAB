import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

version = 'v4'
#algos = ['fixed_window','closing_window', 'singlestep', 'singlestep_rs']
algos = ['singlestep','fixed_window','closing_window']

fdrop = {}
rewards = {}
nactions = {}

for algo in algos:
    fdrop[algo] = pd.DataFrame()
    rewards[algo] = pd.DataFrame()
    nactions[algo] = pd.DataFrame()

for epoch in range(100):
    with open(f'experiments/dropOutState_{version}/epoch_{epoch}.pkl', 'rb') as f:
        results = pickle.load(f)

    states = {}
    for algo in algos:
        states[algo] = pd.DataFrame(results[algo]['states'])
        fdrop[algo] = fdrop[algo].append(pd.Series((states[algo] == 0).mean(axis=1)),ignore_index=True)
        rewards[algo] = rewards[algo].append(pd.Series(results[algo]['rewards']),ignore_index=True)
        nactions[algo] = nactions[algo].append(pd.Series(results[algo]['actions'].sum(axis=1)),ignore_index=True)


# fraction of arms in drop out state 
for algo in algos:
    data = fdrop[algo].stack().reset_index().rename(columns={'level_0':'epoch','level_1':'time_horizon',0:'prop'})
    sns.lineplot(data = data, x = 'time_horizon', y='prop', label=algo)
plt.legend(loc=4)
plt.xlabel('Time horizon')
plt.ylabel('')
plt.title('Fraction of arms in drop out state (N=10, B=2)')
plt.show()

# cumulative reward
for algo in algos:
    cum = rewards[algo].apply(lambda x: np.cumsum(x),axis=1).apply(lambda x: x/np.array(range(1,51)),axis=1)
    data = cum.stack().reset_index().rename(columns={'level_0':'epoch','level_1':'time_horizon',0:'reward'})
    sns.lineplot(data = data, x = 'time_horizon', y='reward', label=algo)
plt.xlabel('Time horizon')
plt.ylabel('')
plt.title('Average cumulative reward')
plt.show()

# amount of resources used (frequency)
data = pd.DataFrame()
for algo in algos:
    count = nactions[algo].apply(lambda x: x.value_counts(),axis=1)
    count = count.stack().reset_index()
    count['level_0'] = algo
    count = count.rename(columns={'level_0':'algo','level_1':'resources',0:'count'})
    data = data.append(count,ignore_index=True)

data['resources'] = data['resources'].astype(int)

fig, ax = plt.subplots(figsize=(5,3))
sns.barplot(x='resources',y='count',hue='algo',data=data)
sns.move_legend(ax, "lower center", bbox_to_anchor=(.5, 1), ncol=3, title=None)
plt.xlabel('Per step resources')
plt.ylabel('Frequency (over 50 steps)')
plt.tight_layout()
plt.show()

# amount of resources used (at each time step)
data = pd.DataFrame()
for algo in ['fixed_window','closing_window']:
    count = nactions[algo].stack().reset_index()
    count['algo'] = algo
    count = count.rename(columns={'level_0':'epoch','level_1':'timestep',0:'resources'})
    data = data.append(count,ignore_index=True)

data['resources'] = data['resources'].astype(int)
data['timestep'] = data['timestep']+1

algo = 'closing_window'
color = 'tab:orange' if algo=='fixed_window' else 'tab:green'
epochs = [0,10,50,100,150,200]
fig, ax = plt.subplots(figsize=(6,6),nrows=len(epochs),sharex=True, sharey=True)
for i, epoch in enumerate(epochs):
    dataplot = data[(data.algo==algo)*(data.epoch==epoch)]
    sns.barplot(x='timestep',y='resources',hue='resources',data=dataplot,ax=ax[i],dodge=False, color=color)
    ax[i].legend([],[], frameon=False)
    ax[i].set_ylabel('')
    ax[i].set_xlabel('')
    ax[i].xaxis.set_visible(False)
plt.ylim(0,4)
fig.supxlabel('Timestep (1 to 50)')
fig.supylabel('Used resources')
fig.suptitle(algo)
plt.tight_layout()
plt.show()


algo = 'fixed_window'
color = 'tab:orange' if algo=='fixed_window' else 'tab:green'
fig, ax = plt.subplots(figsize=(12,4))
dataplot = data[(data.algo==algo)]
sns.barplot(x='timestep',y='resources',data=dataplot,ax=ax, color=color)
ax.set_ylabel('')
ax.set_xlabel('')
fig.supxlabel('Timestep')
fig.supylabel('Used resources (250 epochs)')
fig.suptitle(algo)
plt.tight_layout()
plt.show()
