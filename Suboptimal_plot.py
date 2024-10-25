#%%
import pickle
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

dir_beh = '/home/owo/CLAB/Code/RL/Composition/Behavior'
exp_name = 'exp_ego'
exp_type = 'agent'

path = f'{dir_beh}/{exp_type}/{exp_name}'
path_add = '[S4][DDQN,ego]'

with open(f'{path}/{path_add}trial_type.pkl', 'rb') as f:
    trial_type = pickle.load(f)
with open(f'{path}/{path_add}actual_order.pkl', 'rb') as f:
    actual_order = pickle.load(f)
with open(f'{path}/{path_add}optimal_order.pkl', 'rb') as f:
    optimal_order = pickle.load(f)
with open(f'{path}/{path_add}bias_amount.pkl', 'rb') as f:
    bias_amount = pickle.load(f)

#%%
trial_opt = np.zeros(max(bias_amount))
trial_total = np.zeros(max(bias_amount))
for trial, typ in enumerate(trial_type):
    if (typ==0): # Suboptimal Case
        if (list(optimal_order[trial])==actual_order[trial]):
            trial_opt[bias_amount[trial]-1] += 1
        trial_total[bias_amount[trial]-1] += 1
opt_prob = np.nan_to_num(trial_opt/trial_total, copy=False)
bias = np.arange(1,max(bias_amount)+1)

# psychometric function (sigmoid)
def sigmoid(x, alpha, T):
    return 1./(1+np.exp(-T*(x-alpha)))

# fitting
popt, pcov = curve_fit(sigmoid, bias, opt_prob, maxfev=1000)
X = np.linspace(bias.min(), bias.max(), 100)
psyc = sigmoid(X, *popt)

# plot
plt.figure(figsize=(10,5))
plt.plot(X, psyc*100, c='r', linewidth=2)
plt.scatter(bias, opt_prob*100, c='k')

plt.xlabel('Bias Amount')
plt.ylabel('Optimal Percentage (%)')
plt.title('Optimal Percentage vs Bias Amount')

plt.grid(True)
plt.ylim(0, 100)  # Ensure y-axis starts at 0 and ends at 100 for percentage representation
plt.show()