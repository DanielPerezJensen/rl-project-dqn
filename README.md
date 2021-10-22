# rl-project-dqn

### Cartpole:
DQN: `python train.py --epoch 10 --task CartPole-v0 --algo vanilla --eps_train_start 1 --eps_train_end 0.1 --eps_train_decay_length 100_000 --n_step 4 --target_update_freq 1000 --hidden_sizes 256 512`

Double-DQN: `python train.py --epoch 10 --task CartPole-v0 --algo double --eps_train_start 1 --eps_train_end 0.1 --eps_train_decay_length 100_000 --n_step 4 --target_update_freq 1000 --hidden_sizes 256 512`

### Acrobot: 
DQN: `python train.py --epoch 40 --task Acrobot-v1 --algo vanilla --eps_train_start 1 --eps_train_end 0.1 --eps_train_decay_length 400_000 --n_step 4 --target_update_freq 1000 --hidden_sizes 256 512`

Double-DQN: `python train.py --epochs 40 --task Acrobot-v1 --algo double --eps_train_start 1 --eps_train_end 0.1 --eps_train_decay_length 400_000--n_step 4 --target_update_freq 1000 --hidden_sizes 256 512 --render`

### Pong:
DQN: `python train.py --algo vanilla --task Pong-ram-v0 --epoch 50 --step_per_epoch 200000 --eps_test 0.005 --eps_train_start 1 --eps_train_end 0.05 --eps_train_decay_length 1000000 --buffer_size 100000 --lr 1e-4 --n_step 3 --target_update_freq 10000 --batch_size 64`

Double-DQN: `python train.py --algo double --task Pong-ram-v0 --epoch 50 --step_per_epoch 200000 --eps_test 0.005 --eps_train_start 1 --eps_train_end 0.05 --eps_train_decay_length 1000000 --buffer_size 100000 --lr 1e-4 --n_step 3 --target_update_freq 10000 --batch_size 64`

### DoubleDunk:
DQN: `python train.py --algo vanilla --task DoubleDunk-ram-v0 --epoch 50 --step_per_epoch 200000 --eps_test 0.005 --eps_train_start 1 --eps_train_end 0.05 --eps_train_decay_length 1000000 --buffer_size 100000 --lr 1e-4 --n_step 3 --target_update_freq 10000 --batch_size 64 --test_num 10`

Double-DQN: `python train.py --algo double --task DoubleDunk-ram-v0 --epoch 50 --step_per_epoch 200000 --eps_test 0.005 --eps_train_start 1 --eps_train_end 0.05 --eps_train_decay_length 1000000 --buffer_size 100000 --lr 1e-4 --n_step 3 --target_update_freq 10000 --batch_size 64 --test_num 10`
