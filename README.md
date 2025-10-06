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
pip install -e ".[dev]"
```

4. Install acados
https://docs.acados.org/installation/
```
cd external/acados
mkdir -p build
cd build
cmake -DACADOS_WITH_QPOASES=ON -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DACADOS_WITH_OPENMP=ON -DACADOS_PYTHON=ON -DACADOS_NUM_THREADS=1 ..
# If you use ubuntu, just ..
cmake -DACADOS_WITH_QPOASES=ON -DACADOS_WITH_OPENMP=ON -DACADOS_PYTHON=ON -DACADOS_NUM_THREADS=1 ..
make install -j4
```
You can erase `-DCMAKE_POLICY_VERSION_MINIMUM=3.5` if you can (my MacOS can't)


5. Install acados_template python packages
https://docs.acados.org/python_interface/index.html
```
cd external/acados/interfaces/acados_template
pip install -e .
```
Add path
```
export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:"<acados_root>/lib"
export ACADOS_SOURCE_DIR="<acados_root>"
```

Please see the [Getting started section](https://leap-c.github.io/leap-c/getting_started/index.html) or the [examples folder](https://github.com/leap-c/leap-c/tree/main/leap_c/examples).

6. Execute race_car
- train and test
```
python scripts/run_sac_fop.py --
python scripts/train_racecar_sac_qmatrix.py
python scripts/train_racecar_sac_qmatrix.py --device cuda
python scripts/evaluate_racecar_qmatrix.py --model-path ./output/racecar_q_learning/final_racecar_model.pt
```

- render
```
python scripts/render_trained_model.py output/2025_09_10/16_24_20_sac_race_car_seed_0 --env race_car --render_mode rgb_array --save_video race_car_trained.mp4 --episodes 3 --max_steps 500
```

## Questions?

Open a new thread or browse the existing ones on the [GitHub discussions](https://github.com/leap-c/leap-c/discussions) page.

## Windows
1. WSL2 (Windows Subsystem for Linux)
```
# first time
wsl --install
# after
wsl
```
2. Move to project Folder
`mnt/c` (WSL) to `/Users/username/~`
3. WSL에서 Windows 드라이브는 /mnt/c로 접근)
```
cd /mnt/c/Users/username/~
```

1. cmake 및 기타 설치
```
sudo apt update
sudo apt install -y cmake build-essential libssl-dev zlib1g-dev libbz2-dev \
libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev \
libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev git
```

1. pyenv 설치
```
curl https://pyenv.run | bash
```

1. bashrc에 pyenv 설정 추가
```
echo 'export PATH="$HOME/.pyenv/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init --path)"' >> ~/.bashrc
echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc
```

1. 쉘 재시작
```
exec $SHELL
```

1. 다음부터는 Usage 2번 이어서