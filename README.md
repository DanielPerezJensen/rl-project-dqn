# rl-project-dqn

### Cartpole:
DQN: `python train.py --algo vanilla --render`

Double-DQN: `python train.py --algo double --render`

### Acrobot: 
- Need to find working parameters

DQN: `python train.py --task Acrobot-v1 --algo vanilla --render`

Double-DQN: `python train.py --task Acrobot-v1 --algo double --render`

### Mountain Car: 
- Need to find working parameters

DQN: `python train.py --task MountainCar-v0 --algo vanilla --render`

Double-DQN: `python train.py --task MountainCar-v0 --algo double --render`

### Pong:
DQN: `python train.py --task Pong-ram-v0 ---algo vanilla --render -epoch 30 --step_per_epoch 100000 --test_num 10 --lr 1e-4 --gamma 0.99`

Double DQN: `python train.py --task Pong-ram-v0 ---algo double --render -epoch 30 --step_per_epoch 100000 --test_num 10 --lr 1e-4 --gamma 0.99`

### Breakout:
- Need to find working parameters
- Long to train

DQN: `python train.py --task Breakout-ram-v0 ---algo vanilla --render -epoch 30 --step_per_epoch 100000 --test_num 10 --lr 1e-4 --gamma 0.99`

Double DQN: `python train.py --task Breakout-ram-v0 ---algo double --render -epoch 30 --step_per_epoch 100000 --test_num 10 --lr 1e-4 --gamma 0.99`
