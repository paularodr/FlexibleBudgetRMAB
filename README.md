# Flexible Budgets in Restless Bandits: A Proximal Primal-Dual Algorithm for Efficient Budget Allocation

## Setup
The main file to run the experiments is `run_experiments.py`. The algorithms can be found in the `algos` folder where the following files contain the implementations for the following algorithms:
- `algos/hawkins_actions` and `algos/hawkins_methods` contain the implementation for *Hawkins [2003]* baseline.
- `alogs/compressing_methos` contains the implementation for *Compress (static)* and *Compress (window)* heuristics.
- `algos/minmax_methods` contains the implementation for the *PDGS* algorithm.

The file `environments.py` defines the three experimental domains: drop out state, immediate recovery, and two-state process.

#### To install follow these directions (generic version):

- Install the repo:
- `pip3 install -e .`
- Create the directory structure:
- `bash make_dirs.sh`

#### To run Hawkins baseline and compressing methods, you need to install gurobi and gurobipy by following these steps
- In a web browser Register for a Gurobi account or login at https://www.gurobi.com/downloads/end-user-license-agreement-academic/ 
- Navigate to https://www.gurobi.com/downloads/ and select `Gurobi Optimizer`
- Review the EULA, then click `I accept the End User License Agreement`
- Identify the latest installation... as of writing, it is 9.5.0, and the following commands will reflect that. However, if the latest version has changed, you can replace 9.5.0 in the following commands with the newer version number and/or links on the Gurobi website.
- `cd ~`
- `mkdir tools`
- `cd tools`
- `wget https://packages.gurobi.com/9.5/gurobi9.5.0_linux64.tar.gz`
- `tar xvzf gurobi9.5.0_linux64.tar.gz`
- Add the following lines to your ~/.bashrc file, e.g., via `vim ~/.bashrc`
```
export GUROBI_HOME="~/tools/gurobi912/linux64"
export PATH="${PATH}:${GUROBI_HOME}/bin"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${GUROBI_HOME}/lib"
``` 
- Run `source ~/.bashrc`
- On the browser, navigate to https://www.gurobi.com/downloads/end-user-license-agreement-academic/
- Review the conditions, then click `I accept these conditions`
- Copy the command that looks like `grbgetkey 00000000-0000-0000-0000-000000000000`, then paste and run in the server terminal window
- Enter `Y` to select the default options when prompted
- `cd ~/FlexibleBudgetRMAB/`
- `pip install gurobipy`

## Running Code 
### Experiment Domains

We propose `PDSG`, an algorithm that solves a primal-dual optimization problem in the F-RMAB setting to design reward-maximizing policies under budget flexibility. We compare our algorithm against `Hawkins [2003]`, a Lagrange-based policy for classic RMABs which computes values of Lagrange multipliers after relaxing per-round budget constraints with fixed budget, and againts two additional F-RMAB heuristics: `Compress (static)`, which plans across the flexible window *F* in each timestep and executes the first action before recomputing, and `Compress (closing)`, which plans across a flexible window of size *F*, then *F-1*, and so on until a window of size 1, repeating every *F* steps.

We design three three synthetic domains to show the benefits of allowing for budget flexibility and test our algorithms and baseliens in all three domains:
- **Drop out state**: Characterizes settings with potential urgent interventions by considering three states: a drop out state, a risk state and a safe state.
- **Immediate recovery**: Models maintenance-style RMAB problems. Each arm corresponds to one item that gradually decays over time, and intervention is guarantee to restore that item to peak condition
- **Two-state process**: Models approaches in health intervention planning such as maternal health care. This domain models an entity with two states, a *good* and a *bad* state, with reward 1 for each arm in the good state and 0 for the bad state.

The main file to run the algorithms and baselines on each experimental domain with the hyperparameters indicated in the Appendix is `run_experiments.py`. Details on the parameters can be found by running `python3 run_experiments.py --help`

For example, to run all algorithms on the drop out state domain with *F=2* and *H=30* as in Figure 2 of the paper run:\
`python3 run_experiments.py --seed seed --domain dropOutState --F 2 --H 30 --N 10 --S 3 --niters 50 100 200`\
for `seed` from 0 to 29.
