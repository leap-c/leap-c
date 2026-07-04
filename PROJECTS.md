# Projects Using leap-c

This page collects projects that build on the core `leap-c` differentiable acados/PyTorch layer,
plus related work that may be useful to users.

## leap-c Ecosystem

- [leap-c-lab](https://github.com/leap-c/leap-c-lab): example environments, OCP definitions,
  planners, and controller/planner abstractions for `leap-c`. Use this when you want ready-made
  planners and environments instead of building directly from the core layer.
- [mpc-sac](https://github.com/leap-c/mpc-sac): SAC, CrossQ, trainer, neural-network, and script
  utilities for training MPC controllers. It shows how `leap-c` can be used in reinforcement
  learning through `leap-c-lab` planners and environments.

## Related Work

- [mpc.pytorch](https://github.com/locuslab/mpc.pytorch): early work on embedding MPC in PyTorch
  for end-to-end learning, with a more restricted class of MPC problems.
- [mpcrl](https://github.com/FilippoAiraldi/mpc-reinforcement-learning): a compact codebase for
  using RL with MPC as a function approximator.
- [Neuromancer](https://github.com/pnnl/neuromancer): a differentiable programming library that
  can include parametric optimization layers, including MPC, in PyTorch computational graphs.
- [ntnu-itk-autonomous-ship-lab/rlmpc](https://github.com/ntnu-itk-autonomous-ship-lab/rlmpc):
  RL and MPC code tailored for marine vessel control.

## Add Your Project

If you use `leap-c` in a public project, paper, or benchmark, open a pull request adding a short
entry with a link and one-sentence description.
