# Getting Started
To start using `leap-c` for solving a given problem, several steps are required. We highly recommend consulting the [examples](https://github.com/leap-c/leap-c/tree/main/leap_c/examples) for inspiration.

## The environment (`env.py`)

First, define the [gym environment](https://gymnasium.farama.org/api/env/) with which the agent will interact. This module can contain explicit models (e.g., differential equations), but also serve as a bridge to more complex simulators (e.g., through the simulator's API, the FMI standard or the UDP communication protocol). For more information on the required steps, see `gym`'s documentation on [getting started](https://gymnasium.farama.org/introduction/basic_usage/) and [building a custom environment](https://gymnasium.farama.org/introduction/create_custom_env/).

## The controller (`controller.py`)

Then, create a controller to be applied. Our examples use `acados` to provide an MPC controller. More information on the possibilities of `acados` can be found in the [problem formulation document](https://github.com/acados/acados/blob/main/docs/problem_formulation/problem_formulation_ocp_mex.pdf) or the [Python interface docs](https://docs.acados.org/python_interface/index.html). Generally, our algorithms use the 
[ParameterizedController](https://github.com/leap-c/leap-c/blob/main/leap_c/controller.py) interface.

## Training

Finally, define the configuration of the training loop (training length, learning rates, update frequency etc.) and watch the objective improve :). You can find examples for training configurations in the folder [scripts](https://github.com/leap-c/leap-c/tree/main/scripts).

For hyperparameter optimization, you can also wrap the training loop in an HPO framework such as [Optuna](https://optuna.org/).
