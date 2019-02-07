from comet_ml import Experiment

from agent import Agent
import argparse
from collections import deque
from env import Environment
import numpy as np
import torch

hyper_params = {
    'memory_size': 20000,   # replay buffer size
    'batch_size': 64,          # minibatch size
    'gae_tau': 0.95,
    'ppo_ratio_clip': 0.2,
    'entropy_weight': 0,
    'optimization_epochs': 10,
    'discount_rate': .99,
    'traj_coll_random_steps': 3,
    'value_pred_loss_coefficient': 0.5,
    'clip_param': 0.2,
    'beta': 0.001,
    'tmax': 1000,
    'num_epochs': 10,
    'curation_percentile': 0,
    'gradient_clip': 5,
}

env_files = {
    'reacher': {
        'single': 'Reacher.app',
        'multi': 'Multiagent_Reacher.app'
    },
    'crawler': {
        'Crawler.app'
    }
}


def main(args):
    env_file = _get_env_file(args.single)
    if args.examine:
        with Environment(file_name=env_file, no_graphics=True) as env:
            examine(env)
    if args.random:
        with Environment(file_name=env_file, no_graphics=False) as env:
            random(env)
    if args.train:
        with Environment(file_name=env_file, no_graphics=True) as env:
            experiment = _setup_experiment(disabled=(not args.log))
            train(env, experiment)
    if args.test:
        with Environment(file_name=env_file, no_graphics=False) as env:
            test(env)


def _get_env_file(single=False, crawler=False):
    file = ''
    if crawler:
        file = env_files['crawler']
    elif single:
        file = env_files['reacher']['single']
    else:
        file = env_files['reacher']['multi']
    return 'environments/' + file


def _setup_experiment(disabled=False):
    experiment = Experiment(project_name="udacity-deeprl-project-2", log_code=False,
                            log_env_details=False, disabled=disabled)
    return experiment


def examine(env):
    env_info = env.reset()
    print('Number of agents:', len(env_info.agents))
    action_size = env.action_space_size
    print('Number of actions:', action_size)
    states = env_info.vector_observations
    state_size = states.shape[1]
    print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
    print('The state for the first agent looks like:', states[0])


def random(env):
    env_info = env.reset(train_mode=False)
    num_agents = len(env_info.agents)
    action_size = env.action_space_size
    rewards = np.zeros(num_agents)
    while True:
        actions = np.random.randn(num_agents, action_size)
        actions = np.clip(actions, -1, 1)
        env_info = env.step(actions)
        dones = env_info.local_done
        rewards += env_info.rewards
        if np.any(dones):
            break
        print('\rReward: {:.2f}'.format(np.mean(rewards)), end="")
    print('\rReward: {:.2f}'.format(np.mean(rewards)))


def test(env):
    env.reset(train_mode=False)
    states = env.info.vector_observations
    agent = Agent(env, hyper_params)
    agent.policy.load_state_dict(torch.load('results/checkpoint.pth'))
    rewards = 0
    while True:
        dist, _ = agent.policy(states)
        actions = dist.sample()
        env.step(actions.detach().numpy())
        next_states = env.info.vector_observations
        dones = env.info.local_done
        rewards += np.array(env.info.rewards)
        states = next_states
        if np.any(dones):
            break
    print('\rRewards: ', rewards)


def train(env, experiment, max_episodes=1000):
    agent = Agent(env, hyper_params)
    rewards_window = [deque(maxlen=100) for n in range(env.num_agents)]
    with experiment.train():
        for i_episode in range(1, max_episodes + 1):
            env.reset(train_mode=True)
            rewards_total = 0
            while True:
                rewards, done = agent.collect_trajectories()
                rewards_total += rewards
                agent.clipped_surrogate_update()
                if done:
                    break
            # Track rewards
            for i, udr in enumerate(rewards_total):
                rewards_window[i].append(udr)
            experiment.log_metric('reward', np.mean(rewards_total), step=i_episode)
            print('\rEpisode {}\tAverage Reward: {:.2f}'.format(i_episode, np.mean(rewards_total)), end="")
            if i_episode % 100 == 0:
                print('\rEpisode {}\tAverage Reward: {:.2f}'.format(i_episode, np.mean(rewards_total)))
            mean_window_rewards = np.mean(rewards_window, axis=1)
            if (mean_window_rewards >= 30.0).all():
                print('\nEnvironment solved in {:d} episodes.  Average agent rewards: '
                      .format(i_episode), mean_window_rewards)
                torch.save(agent.policy.state_dict(), 'results/checkpoint.pth')
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DeepRL - Continuous control project')
    parser.add_argument('--examine',
                        action="store_true",
                        dest="examine",
                        help='Print environment information')
    parser.add_argument('--random',
                        action="store_true",
                        dest="random",
                        help='Start a random agent')
    parser.add_argument('--train',
                        action="store_true",
                        dest="train",
                        help='Train a new network')
    parser.add_argument('--test',
                        action="store_true",
                        dest="test",
                        help='Load an existing network and test it')
    parser.add_argument('--log',
                        action="store_true",
                        dest="log",
                        help='Log results to comet.ml')
    parser.add_argument('--single',
                        action="store_true",
                        dest="single",
                        help='Run the single agent version of the environment')
    args = parser.parse_args()
    if not any(vars(args).values()):
        parser.error('No arguments provided.')
    main(args)
