import argparse
import glob
import os
import pprint
from datetime import datetime

import gym
import numpy as np
import pandas as pd
import torch
from tensorboard.backend.event_processing import event_accumulator
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.policy import DQNPolicy
from tianshou.trainer import offpolicy_trainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net
from torch.utils.tensorboard import SummaryWriter


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task", type=str, default="CartPole-v0", help="Name of gym environment"
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="vanilla",
        help="Q-learning variation (vanilla | double)",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed ('None' uses no seed)"
    )
    parser.add_argument(
        "--eps_test", type=float, default=0.05, help="Testing policy epsilon"
    )
    parser.add_argument(
        "--eps_train_start",
        type=float,
        default=0.1,
        help="Starting training policy epsilon",
    )
    parser.add_argument(
        "--eps_train_end", type=float, default=0.1, help="End training policy epsilon"
    )
    parser.add_argument(
        "--eps_train_decay_length",
        type=int,
        default=0,
        help="Number of steps to decay training policy epsilon for",
    )
    parser.add_argument(
        "--buffer_size", type=int, default=20000, help="Replay buffer size"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--gamma", type=float, default=0.99, help="Environment discount factor"
    )
    parser.add_argument(
        "--n_step", type=int, default=4, help="Number of steps ahead to estimate"
    )
    parser.add_argument(
        "--target_update_freq",
        type=int,
        default=320,
        help="Every how many steps to update target policy",
    )
    parser.add_argument("--epoch", type=int, default=20, help="Max epochs")
    parser.add_argument(
        "--step_per_epoch", type=int, default=10000, help="Number of steps per epoch"
    )
    parser.add_argument(
        "--step_per_collect",
        type=int,
        default=10,
        help="Number of steps between network updates",
    )
    parser.add_argument(
        "--update_per_step",
        type=float,
        default=0.1,
        help="Fraction of times network is updated per step",
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument(
        "--hidden_sizes",
        type=int,
        nargs="*",
        default=[128, 128, 128, 128],
        help="List of hidden layer sizes",
    )
    parser.add_argument(
        "--training_num", type=int, default=10, help="Number of training processes"
    )
    parser.add_argument(
        "--test_num", type=int, default=100, help="Number of testing processes"
    )
    parser.add_argument(
        "--logdir", type=str, default="log", help="Log output directory"
    )
    parser.add_argument(
        "--render", action="store_true", help="Visualize final policy after training"
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_known_args()[0]
    return args


def save_tb_to_csv(writer):
    # Get paths
    log_dir = writer.log_dir
    tb_path = glob.glob(log_dir + "/events*")[0]

    # Load tensorboard log file
    ea = event_accumulator.EventAccumulator(tb_path)
    ea.Reload()

    # Save each logged scalar to csv
    if not os.path.exists(os.path.join(log_dir, "csv")):
        os.makedirs(os.path.join(log_dir, "csv"))

    for tag in ea.Tags()["scalars"]:
        filename = os.path.join(log_dir, "csv", tag.replace("/", "_") + ".csv")
        df = pd.DataFrame(ea.Scalars(tag))
        df.to_csv(filename)


def train(args):
    # Setup environment
    env = gym.make(args.task)
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    train_envs = DummyVectorEnv(
        [lambda: gym.make(args.task) for _ in range(args.training_num)]
    )
    test_envs = DummyVectorEnv(
        [lambda: gym.make(args.task) for _ in range(args.test_num)]
    )

    # Seed
    if args.seed:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        train_envs.seed(args.seed)
        test_envs.seed(args.seed)

    # Define model
    net = Net(
        args.state_shape,
        args.action_shape,
        hidden_sizes=args.hidden_sizes,
        device=args.device,
    ).to(args.device)
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)

    # Define policy
    if args.algo == "vanilla":
        policy = DQNPolicy(
            net,
            optim,
            args.gamma,
            args.n_step,
            target_update_freq=args.target_update_freq,
            is_double=False,
        )
    elif args.algo == "double":
        policy = DQNPolicy(
            net,
            optim,
            args.gamma,
            args.n_step,
            target_update_freq=args.target_update_freq,
            is_double=True,
        )
    else:
        raise NotImplementedError(f"{args.algo} is not a valid Q-learning variation.")

    # Define replay buffer
    buf = VectorReplayBuffer(args.buffer_size, buffer_num=len(train_envs))

    # Define collectors
    train_collector = Collector(policy, train_envs, buf, exploration_noise=True)
    test_collector = Collector(policy, test_envs, exploration_noise=True)
    train_collector.collect(n_step=args.batch_size * args.training_num)

    # Setup logging
    log_path = os.path.join(
        args.logdir,
        args.task,
        "dqn" if args.algo == "vanilla" else "double_dqn",
        datetime.now().strftime("%m-%d-%H-%M-%S"),
    )
    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer)

    def save_fn(policy):
        filename = "policy.pth"
        torch.save(policy.state_dict(), os.path.join(log_path, filename))

    def stop_fn(mean_rewards):
        if env.spec.reward_threshold:
            return mean_rewards >= env.spec.reward_threshold
        elif "Pong" in args.task:
            return mean_rewards >= 20
        else:
            return False

    def train_fn(epoch, env_step):
        # Linear decay from eps_train_start to eps_train_end for eps_train_decay_length steps
        if env_step < args.eps_train_decay_length:
            eps = args.eps_train_start - env_step / args.eps_train_decay_length * (
                args.eps_train_start - args.eps_train_end
            )
        else:
            eps = args.eps_train_end
        policy.set_eps(eps)
        if env_step % 1000 == 0:
            logger.write("train/env_step", env_step, {"train/eps": eps})

    def test_fn(epoch, env_step):
        policy.set_eps(args.eps_test)

    # trainer
    result = offpolicy_trainer(
        policy,
        train_collector,
        test_collector,
        args.epoch,
        args.step_per_epoch,
        args.step_per_collect,
        args.test_num,
        args.batch_size,
        update_per_step=args.update_per_step,
        train_fn=train_fn,
        test_fn=test_fn,
        stop_fn=stop_fn,
        save_fn=save_fn,
        logger=logger,
    )

    # Print results
    pprint.pprint(result)

    # Save metrics as csv
    save_tb_to_csv(writer)

    # Get value estimates
    q_vals = []
    for _ in range(10000):
        idx = np.random.randint(args.buffer_size)
        obs = buf.get(idx, key="obs")
        vals = policy.model(np.expand_dims(obs, 0))[0].detach().cpu().numpy()[0]
        q_vals.append(np.max(vals))
    mean_q_val = np.mean(q_vals)
    std_q_val = np.std(q_vals)

    # Save value estimates to csv
    df = pd.DataFrame([[mean_q_val, std_q_val]], columns=["mean", "std"])
    filename = os.path.join(writer.log_dir, "csv", "q_vals.csv")
    df.to_csv(filename)

    # Test final policy
    env = gym.make(args.task)
    policy.eval()
    policy.set_eps(args.eps_test)
    collector = Collector(policy, env)
    if args.render:
        fps = 1 / 30
    else:
        fps = 0
    result = collector.collect(n_episode=1, render=fps)
    rews, lens = result["rews"], result["lens"]
    print(f"Final reward: {rews.mean()}, length: {lens.mean()}")


if __name__ == "__main__":
    args = get_args()
    train(args)
