import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import hawkins_actions
from environments import dropOutState

gamma = 0.95
N = 5
B = 2.0 #two step budget
start_state = np.array([2,2,2,2,1])


env = dropOutState(N, B, start_state)

actions_hawkins, L, Q, lambda_val, Q_vals_per_state = hawkins_actions.get_hawkins_actions(env.N, env.T, env.R, env.C, env.B, env.current_state_twostep, gamma)

lambdas = np.arange(0, 1, 0.01)
qvals = []
for new_lambda in lambdas:
    actions_lambda, L_vals, Q_vals, new_lambda, Q_vals_per_state = hawkins_actions.get_hawkins_actions_preset_lambda(env.N, env.T, env.R, env.C, env.B, env.current_state_twostep, lambda_val=new_lambda, gamma=gamma)
    qvals.append(pd.Series(Q_vals_per_state[-1]))

qvals = pd.DataFrame(qvals)
qvals['lambda'] = lambdas
qvals = qvals.set_index('lambda',drop=True)

fig, ax = plt.subplots(figsize=(6,4))
qvals.plot(ax=ax)
ax.axes.axvline(lambda_val,color='k', linestyle='--')
plt.title('Q functions for state 1')
plt.ylabel('Q value')
plt.show()
