import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import rc

rc('font',**{'family':'serif','serif':['Times']})
rc('text', usetex=True)


niters = [50,100,200]
pdgs_algos = [f'PDGS-{i}' for i in niters]
algos = ['hawkins', 'compress_static', 'compress_closing']+pdgs_algos
names_algos = ['Hawkins [2003]', 'Compress (static)', 'Compress (closing)','PDGS-50','PDGS-100','PDGS-200']

def read_data(domain, T, H, N, S=5, B=1):

    if domain == 'dropOutState':
        experiments = [f'T_{T[i]}_H_{H[i]}_N_{N}_B_{B}' for i in range(len(T))]
    if domain == 'immediateRecovery':
        experiments = [f'T_{T[i]}_H_{H[i]}_N_{N}_S_{S}_B_{B}' for i in range(len(T))]
    if domain == 'twoStateProcess':
        experiments = [f'T_{T[i]}_H_{H[i]}_N_{N}_B_{B}' for i in range(len(T))]


    rewards = pd.DataFrame()
    resources = pd.DataFrame()
    runtimes = pd.DataFrame()

    for i,exp in enumerate(experiments):
        for seed in range(29):
            with open(f'results/{domain}/{exp}/{exp}_seed_{seed}.pkl', 'rb') as f:
                results = pickle.load(f)
            rw = []
            rs = []
            rt = []
            for algo in algos:
                rw.append(results[algo]['rewards'].sum())
                rs.append(results[algo]['actions'].sum(axis=1))
                rt.append(np.log10(results[algo]['runtime']))
            rw = pd.DataFrame(rw)
            rw['algo'] = algos
            rw['T'] = [T[i]]*len(algos)
            rw['seed'] = [seed]*len(algos)

            rs = pd.DataFrame(rs)
            rs['algo'] = algos
            rs['T'] = [T[i]]*len(algos)
            rs['seed'] = [seed]*len(algos)

            rt = pd.DataFrame(rt)
            rt['algo'] = algos
            rt['T'] = [T[i]]*len(algos)
            rt['seed'] = [seed]*len(algos)

            rewards = rewards.append(rw,ignore_index=True)
            resources = resources.append(rs,ignore_index=True)
            runtimes = runtimes.append(rt,ignore_index=True)

    rewards['algo'] = rewards.algo.str.replace('_','\_')
    resources['algo'] = resources.algo.str.replace('_','\_')
    runtimes['algo'] = runtimes.algo.str.replace('_','\_')

    flexible = resources.copy()
    flexible['flexible'] = (resources.iloc[:,:-3]!=1).sum(axis=1)

    return rewards, resources, flexible, runtimes


def read_data_random(domain, T, H, N, S=5, B=1):

    if domain == 'dropOutState':
        exp = 'T_2_H_30_N_10_B_1'
    if domain == 'birthDeathProcess':
        exp = 'T_2_H_10_N_10_S_5_B_1'
    if domain == 'riskProneArms':
        exp = 'T_2_H_6_N_10_B_1'

    rewards = pd.DataFrame()
    for seed in range(100):
        with open(f'results/{domain}/{exp}/{exp}_seed_{seed}_random.pkl', 'rb') as f:
            results = pickle.load(f)
        rw = []
        algo='random'
        rw.append(results[algo]['rewards'].sum())
        rw = pd.DataFrame(rw)
        rw['seed'] = [seed]
        rewards = rewards.append(rw,ignore_index=True)
    return rewards

S=5
Ts = [[2,3,5],[2,5,10],[2,3,6]]
Hs = [[30,30,30],[10,10,10],[6,6,6]]
sub = ['a','b','c']
names = [f'Dropout state', 'Immediate recovery', 'Two-state process']
rewards_ds, resources_ds, flexible_ds, runtimes_ds = read_data('dropOutState', Ts[0], H=Hs[0], N=10)
rewards_bd, resources_bd, flexible_bd, runtimes_bd = read_data('immediateRecovery', Ts[1], H=Hs[1], N=10, S=S)
rewards_ts, resources_ts, flexible_ts, runtimes_ts = read_data('twoStateProcess', Ts[2], H=Hs[2], N=10)

#rewards
mins = [50,0,22]
fig, ax = plt.subplots(figsize=(8,2),ncols=3)
for i, df in enumerate([rewards_ds, rewards_bd,rewards_ts]):
    g = sns.barplot(
        x="T", 
        y=0, 
        hue="algo", 
        data=df, 
        ci=68,
        errwidth=1,
        palette="rocket",
        estimator=np.mean,
        edgecolor="black",
        ax=ax[i]
        )
    g.legend_.remove()
    ax[i].set_ylim(bottom=mins[i])
    ax[i].set_ylabel("")
    ax[i].set_xlabel(f"({sub[i]}) {names[i]}")
    ax[i].set_xticklabels([f'F={t}' for t in Ts[i]])
    if i==2:
        plt.axhline(y=25, color='gray', linestyle='-')
ax[0].set_ylabel("Cumulative reward")
lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
fig.legend(lines[:len(algos)], names_algos, loc='upper center',ncol=6, columnspacing=1, handletextpad=0.2, handlelength=1)
fig.tight_layout()
plt.subplots_adjust(top=0.8)
plt.show()


# RESORUCES

fig, ax = plt.subplots(figsize=(10,5), ncols=3, nrows=3,gridspec_kw={'width_ratios': [2,3,5]})
for j, df in enumerate([resources_ds, resources_bd,resources_ts]):
    for i,f in enumerate(Ts[j]):
        r = df.loc[df['T']==f,:].reset_index(drop=True)
        counts = pd.DataFrame()
        for t in range(f+1):
            x = r.iloc[:,-3:]
            x['count'] = (r.iloc[:,:-3] == t).sum(axis=1)
            x['resources'] = t
            counts = counts.append(x,ignore_index=True)

        g = sns.barplot(
            x="resources", 
            y="count", 
            hue="algo", 
            data=counts, 
            ci=68,
            errwidth=1,
            palette="rocket",
            estimator=np.mean,
            edgecolor='black',
            ax=ax[j,i]
            )
        g.legend_.remove()
        ax[j,i].set_ylim(bottom=0)
        ax[j,i].set_xlabel(f'({sub[j]}.{i+1}) {names[j]} F={t}')
        if i!=0:
            ax[j,i].set_yticks([])
            ax[j,i].set_ylabel('')
ax[0,0].set_ylabel('')
ax[2,0].set_ylabel('')
fig.legend(lines[:len(algos)], names_algos, loc='upper center',ncol=len(algos))
fig.tight_layout()
fig.text(0.5, 0.04, 'Number of per round resources', ha='center')
plt.subplots_adjust(top=0.9,bottom=0.18)
plt.show()

# RUNTIMES

fig, ax = plt.subplots(figsize=(7,2.5),ncols=3)
for i, df in enumerate([runtimes_ds, runtimes_bd,runtimes_ts]):
    scatter = df.groupby(['T','algo'],sort=False).mean().reset_index()
    scatter = scatter
    g = sns.scatterplot(
        x="T", 
        y=0, 
        hue="algo", 
        data=scatter,
        s=8,
        palette="rocket",
        estimator=np.mean,
        ax=ax[i]
        )
    g.legend_.remove()
    g = sns.lineplot(
        x="T", 
        y=0, 
        hue="algo", 
        data=df, 
        linewidth=1,
        ci=68,
        err_style='bars',
        palette="rocket",
        estimator=np.mean,
        ax=ax[i]
        )
    g.legend_.remove()
    ax[i].set_ylabel("")
    ax[i].set_xlabel(f"({sub[i]}) {names[i]}")
    ax[i].set_yticks(np.arange(-2, 4, 0.5), minor=True)
    ax[i].grid(which='both', axis='y',alpha=0.2) 
ax[0].set_ylabel("Seconds (log10)")
lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
fig.legend(lines[:len(algos)], names_algos, loc='upper center',ncol=6, columnspacing=0.5, handletextpad=0.2, handlelength=1)
fig.tight_layout()
fig.text(0.5, 0.04, 'Flexible time window ($F$)', ha='center')
plt.subplots_adjust(top=0.85, bottom=0.3)
plt.show()

#RUNTIMES F=2
rocket = sns.color_palette("rocket").as_hex()
palette = {algos[2].replace('_','\_'):rocket[2],
           algos[3]:rocket[3], 
           algos[4]:rocket[4],
           algos[5]:rocket[5]
           }


doms = ['Dropout \n state', 'Immediate \n Recovery', 'Two-state']
fig, ax = plt.subplots(figsize=(2.8,2))
ax.set_yticks(np.arange(-1.5,4.5,0.5), minor=True)
#ax.grid(axis='y', which='both',alpha=0.8, zorder=1) 
runtimes = pd.DataFrame()
for i, df in enumerate([runtimes_ds, runtimes_bd,runtimes_ts]):
    df = df.loc[df['T']==2,[0,'algo']]
    df['domain'] = doms[i]
    runtimes = runtimes.append(df,ignore_index=True)

runtimes = runtimes[runtimes['algo'].isin(['hawkins\_closing','chambolle-pock-50','chambolle-pock-100', 'chambolle-pock-200'])]
g = sns.barplot(
    x="domain", 
    y=0, 
    hue="algo", 
    data=runtimes, 
    ci=68,
    errwidth=1,
    palette=palette,
    estimator=np.mean,
    edgecolor='black',
    ax=ax,
    zorder=0
    )
g.legend_.remove()
ax.set_xlabel('')
ax.set_ylabel('Seconds (log10)')
plt.axhline(y=0, color='black', linestyle='-')
lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
# fig.legend(lines[:len(algos)], ['Compress (closing)','PDGS-50','PDGS-100','PDGS-200'],
# loc='upper right',ncol=1, bbox_to_anchor=(1, 0.9),handlelength=1,handletextpad=0.2)
fig.legend(lines[:len(algos)], ['Compress (closing)','PDGS-50','PDGS-100','PDGS-200'],
loc='upper center',ncol=2,handlelength=1,handletextpad=0.2,bbox_to_anchor=(0.55, 1),columnspacing=0.5)
fig.tight_layout()
plt.subplots_adjust(top=0.7)
plt.show()

gap = []
df1 = rewards_ds.groupby(['T','algo']).mean().reset_index()
ch = df1.loc[df1.algo=='chambolle-pock-200',0].values
cl = df1.loc[df1.algo=='hawkins\_closing',0].values
gap.append(list((ch-cl)/ch))

df1 = rewards_bd.groupby(['T','algo']).mean().reset_index()
ch = df1.loc[df1.algo=='chambolle-pock-200',0].values
cl = df1.loc[df1.algo=='hawkins\_closing',0].values
gap.append(list((ch-cl)/ch))

df1 = rewards_ts.groupby(['T','algo']).mean().reset_index()
ch = df1.loc[df1.algo=='chambolle-pock-200',0].values
cl = df1.loc[df1.algo=='hawkins\_closing',0].values
gap.append(list((ch-cl)/ch))

gaps = pd.DataFrame()
gaps['T'] = sum(Ts,[])
gaps['domain'] = sum([[x]*3 for x in names],[])
gaps['gap'] = sum(gap,[])


fig, ax = plt.subplots(figsize=(4,2),ncols=3)
ax.set_yticks(np.arange(-1.5,4.5,0.5), minor=True)

for i,dom in enumerate(names):
    g = sns.barplot(
        x="domain", 
        y="gap", 
        hue="T", 
        data=gaps[gaps.domain==dom],
        edgecolor='black',
        ax=ax[i]
        )
    ax[i].set_xlabel('')
ax[0].set_ylabel('Gap in reward (\%)')
fig.tight_layout()
plt.subplots_adjust(top=0.7)
plt.show()