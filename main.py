import gym
from train_pendulum import train_pendulum
from train_cartpole import train_cartpole
from train_pongram import train_pongram
import argparse


# Select which env to use
def select_env(env_name):
    if env_name == 'cartpole':
        return gym.make('CartPole-v0').unwrapped
    if env_name == 'pendulum':
        return gym.make('Pendulum-v1')
    if env_name == 'pongram':
        return gym.make('Pong-ram-v0')

if __name__ == "__main__":

    #Overall
    parser = argparse.ArgumentParser(description='Solve multiple environments with DQN or DDQN')
    parser.add_argument('--env_name', type=str, default="cartpole", help="Choose environment, options: cartpole,"
                                                                         "pendulum or pongram")
    parser.add_argument(
        '--gamma', type=float, default=0.9, metavar='G', help='Discount factor (default: 0.9)')
    parser.add_argument('--q_function', type=str, default="double", help="Type of Q function, "
                                                                          "choose between 'vanilla' or 'double'"
                                                                          "(default: vanilla)")
    #Cartpole
    parser.add_argument('--max_size', type=int, default=10000, help='(Cartpole) Size of Replay memory (default: 10000)')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='(Cartpole) Learning rate (default: 0.001)')
    parser.add_argument('--episodes', type=int, default=200, help='(Cartpole) Amount of episodes (default: 200)')
    parser.add_argument('--eps_start', type=float, default=0.9, help='(Cartpole) Starting value of epsilon (default: 0.9)')
    parser.add_argument('--eps_end', type=float, default=0.05, help='(Cartpole) End value of epsilon (default: 0.05)')
    parser.add_argument('--eps_decay', type=int, default=200, help='(Cartpole) Amount of episodes (default: 200)')
    parser.add_argument('--hidden_layer', type=int, default=256, help='(Cartpole) Hidden layer size')
    parser.add_argument('--batch_size', type=int, default=64, help='(Cartpole) Batch size')

    #Pendulum
    parser.add_argument(
        '--num_actions', type=int, default=5, metavar='N', help='(Pendulum) discretize action space (default: 5)')
    parser.add_argument('--render', action='store_true', help='(Pendulum) render the environment')
    parser.add_argument(
        '--log_interval',
        type=int,
        default=10,
        metavar='N',
        help='interval between training status logs (default: 10)')

    #Pongram

    args = parser.parse_args()

    env = select_env(args.env_name)
    if args.env_name == "cartpole":
        train_cartpole(env, args.episodes, args.eps_start, args.eps_end, args.eps_decay, args.gamma,
                       args.learning_rate, args.hidden_layer, args.batch_size, args.max_size)
    if args.env_name == "pendulum":
        train_pendulum(env, args.gamma, args.num_actions, args.render, args.log_interval, args.q_function)
    # if args.env_name == "pongram":
    #     train_pongram(env)