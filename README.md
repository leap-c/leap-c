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
- You can erase `-DCMAKE_POLICY_VERSION_MINIMUM=3.5` if you can (my MacOS can't)
- Maybe, there are more errors. Contact to `hanjw2496@yonsei.ac.kr` (or talk with GPT ..)


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
- train
```
python scripts/run_sac_fop.py --
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

2. pyenv 설치
```
curl https://pyenv.run | bash
```

3. bashrc에 pyenv 설정 추가
```
echo 'export PATH="$HOME/.pyenv/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init --path)"' >> ~/.bashrc
echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc
```

4. Restart shell
```
exec $SHELL
```

5. Then, Go to Usage 2 ..

### 참고
- Mac 설치시, llvm, libomp 설치하고 zshrc에 넣어주었음 (openmp 우회하기 위함)
- 이렇게 하면 Openmp 우회가 되는데, libqpOASES_e.dylib이 인식 안되는 문제가 생김. 이것도 해결해주면 됨 (자세한 내용은 기억이 안남)