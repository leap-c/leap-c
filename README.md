# leap-c (Learning Predictive Control)

## Introduction

`leap-c` provides tools for learning optimal control policies using Imitation learning (IL) and Reinforcement Learning (RL) to enhance Model Predictive Control (MPC) algorithms. It is built on top of [CasADi](https://web.casadi.org/), [acados](https://docs.acados.org/index.html) and [PyTorch](https://pytorch.org/).

## Installation

`leap-c` can be set up up by following the [installation guide](https://leap-c.github.io/leap-c/installation.html).

## Usage
Use pyenv, pyenv-virtual env

Linux/MacOS can use

1. Clone this repository
```
git clone --recursive https://github.com/HAN2496/rlhf-mpc.git
```

2. Make virtual environment using `pyenv`
```
pyenv install -v 3.11.10
pyenv virtualenv 3.11.10 rlhf_mpc_env
pyenv local rlhf_mpc_env
```

3. Install libraries
```
pip install casadi
pip install torch
pip install -e .
```

4. Install acados
https://docs.acados.org/installation/
```
cd external/acados
mkdir -p build
cd build
cmake -DACADOS_WITH_QPOASES=ON -DCMAKE_POLICY_VERSION_MINIMUM=3.5 ..
make install -j4
```
You can erase `-DCMAKE_POLICY_VERSION_MINIMUM=3.5` if you can (my MacOS can't)


5. Install acados_template python packages
https://docs.acados.org/python_interface/index.html
```
cd external/acados/interfaces/acados_template
pip install -e .
```

Please see the [Getting started section](https://leap-c.github.io/leap-c/getting_started/index.html) or the [examples folder](https://github.com/leap-c/leap-c/tree/main/leap_c/examples).

6. Execute race_car
- train and test
```
python scripts/train_racecar_sac_qmatrix.py --max-steps 10000 --output-dir ./outputs/racecar_q_learning
python scripts/evaluate_racecar_qmatrix.py --model-path ./outputs/racecar_q_learning/final_racecar_model.pt
```

- render
```
python scripts/render_trained_model.py output/2025_09_10/16_24_20_sac_race_car_seed_0 --env race_car --render_mode rgb_array --save_video race_car_trained.mp4 --episodes 3 --max_steps 500
```

## Questions?

Open a new thread or browse the existing ones on the [GitHub discussions](https://github.com/leap-c/leap-c/discussions) page.
