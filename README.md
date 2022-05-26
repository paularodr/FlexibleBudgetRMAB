# Flexible Budgets in Restless Bandits: A Proximal Primal-Dual Algorithm for Efficient Budget Allocation

## Setup
We propose `PDSG`, an algorithm that solves a primal-dual optimization problem in the F-RMAB setting to design reward-maximizing policies under budget flexibility. We compare our algorithm against Lagrange-based policies for classic RMABs (`Hawkins`[2003]) which computes values of Lagrange multipliers after relaxing per-round budget constraints with fixed budget and two additional F-RMAB heuristics: `Compress (static)`, which plans across the flexible window *F* in each timestep and executes the first action before recomputing and `Compress (closing)`, which plans across a flexible window of size *F*, then *F-1*, and so on until a window of size 1, repeating every *F* steps.

We design three three synthetic domains to show the benefits of allowing for budget flexibility and test our algorithms and baseliens in all three domains:
- **Drop out state**: Characterizes settings with potential urgent interventions by considering three states: a drop out state, a risk state and a safe state.
- **Immediate recovery**: Models maintenance-style RMAB problems. Each arm corresponds to one item that gradually decays over time, and intervention is guarantee to restore that item to peak condition
- **Two-state process**: Models approaches in health intervention planning such as maternal health care. This domain models an entity with two states, a *good* and a *bad* state, with reward 1 for each arm in the good state and 0 for the bad state.

The main file to run the algorithms and baselines on each experimental domain is `run_experiments.py`. Details on the parameters can be found by running `python3 run_experiments.py --help`

#### To install follow these directions (generic version):

- Install the repo:
- `pip3 install -e .`
- Create the directory structure:
- `bash make_dirs.sh`
