# Flexible Budgets in Restless Bandits: A Proximal Primal-Dual Algorithm for Efficient Budget Allocation

## Setup
We propose `PDSG`, an algorithm that solves a primal-dual optimization problem in the F-RMAB setting to design reward-maximizing policies under budget flexibility. We compare our algorithm against Lagrange-based policies for classic RMABs (`Hawkins`[2003]) which computes values of Lagrange multipliers after relaxing per-round budget constraints with fixed budget and two additional F-RMAB heuristics: `Compress (static)`, which plans across the flexible window *F* in each timestep and executes the first action before recomputing and `Compress (closing)`, which plans across a flexible window of size *F*, then *F-1*, and so on until a window of size 1, repeating every *F* steps.

#### To install follow these directions (generic version):

- Install the repo:
- `pip3 install -e .`
- Create the directory structure:
- `bash make_dirs.sh`
