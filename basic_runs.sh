#!/bin/bash

echo "Task: CartPole-v0"
for ALGO in vanilla double
do
    echo "Algorithm: ${ALGO}"
    for SEED in 22 45 7 33 36 13 21 42 50 43
    do  
        echo "Seed: ${SEED}"
        python train.py --epoch 10 --task CartPole-v0 --algo ${ALGO} --eps_train_start 1 --eps_train_end 0.1 --n_step 4 --target_update_freq 1000 --hidden_sizes 256 512 --seed ${SEED}
    done
done

echo "Task: Acrobot-v1"
for ALGO in vanilla double
do
    echo "Algorithm: ${ALGO}"
    for SEED in 22 45 7 33 36 13 21 42 50 43
    do
        echo "Seed: ${SEED}"
        python train.py --epoch 40 --task Acrobot-v1 --algo ${ALGO} --eps_train_start 1 --eps_train_end 0.1 --n_step 4 --target_update_freq 1000 --hidden_sizes 256 512 --seed ${SEED}
    done
done
