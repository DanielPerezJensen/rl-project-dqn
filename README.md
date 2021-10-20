# rl-project-dqn

### Cartpole:
DQN: `python train.py --epoch 10 --task CartPole-v0 --algo vanilla --eps_train_start 1 --eps_train_end 0.1 --eps_train_decay_length 100_000 --n_step 4 --target_update_freq 1000 --hidden_sizes 256 512`

Double-DQN: `python train.py --epoch 10 --task CartPole-v0 --algo double --eps_train_start 1 --eps_train_end 0.1 --eps_train_decay_length 100_000 --n_step 4 --target_update_freq 1000 --hidden_sizes 256 512`

### Acrobot: 
DQN: `python train.py --epoch 40 --task Acrobot-v1 --algo vanilla --eps_train_start 1 --eps_train_end 0.1 --eps_train_decay_length 400_000 --n_step 4 --target_update_freq 1000 --hidden_sizes 256 512`

Double-DQN: `python train.py --epochs 40 --task Acrobot-v1 --algo double --eps_train_start 1 --eps_train_end 0.1 --eps_train_decay_length 400_000--n_step 4 --target_update_freq 1000 --hidden_sizes 256 512 --render`

### Pong:
DQN: `python train.py --task Pong-ram-v0 ---algo vanilla --render -epoch 30 --step_per_epoch 100000 --test_num 10 --lr 1e-4 --gamma 0.99`

Double DQN: `python train.py --task Pong-ram-v0 ---algo double --render -epoch 30 --step_per_epoch 100000 --test_num 10 --lr 1e-4 --gamma 0.99`

### Breakout:
- Need to find working parameters
- Long to train

DQN: `python train.py --task Breakout-ram-v0 ---algo vanilla --render -epoch 30 --step_per_epoch 100000 --test_num 10 --lr 1e-4 --gamma 0.99`

Double DQN: `python train.py --task Breakout-ram-v0 ---algo double --render -epoch 30 --step_per_epoch 100000 --test_num 10 --lr 1e-4 --gamma 0.99`
