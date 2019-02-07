#  The udacity training materials, https://github.com/ShangtongZhang, and https://github.com/martinenzinger
#  were used as reference for the PPO algorithm.

import numpy as np
from collections import deque
import torch
from model import ActorCritic


class Agent():
    def __init__(self, env, hyper_params):
        self.env = env
        self.num_agents = env.num_agents
        self.action_size = env.action_space_size
        self.state_size = env.state_size
        self.hyper_params = hyper_params
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Learning objects
        self.memory = Memory(hyper_params['memory_size'], hyper_params['batch_size'])
        self.policy = ActorCritic(self.state_size, self.action_size, seed=0)
        self.opt = torch.optim.Adam(self.policy.parameters(), 3e-4)

        # Starting Values
        self.states = env.info.vector_observations
        self.clip_param = hyper_params['clip_param']
        self.beta = hyper_params['beta']

    def collect_trajectories(self):
        done = False
        states = self.env.info.vector_observations

        state_list = []
        reward_list = []
        prob_list = []
        action_list = []
        value_list = []

        # Random steps to start
        if self.hyper_params['traj_coll_random_steps'] > 0:
            for _ in range(self.hyper_params['traj_coll_random_steps']):
                actions = np.random.randn(self.num_agents, self.action_size)
                actions = np.clip(actions, -1, 1)
                env_info = self.env.step(actions)
                states = env_info.vector_observations

        # Finish trajectory using policy
        for t in range(self.hyper_params['tmax']):
            states = torch.FloatTensor(states).to(self.device)
            dist, values = self.policy(states)
            actions = dist.sample()
            probs = dist.log_prob(actions).sum(-1).unsqueeze(-1)

            env_info = self.env.step(actions.cpu().detach().numpy())
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done

            # store the result
            state_list.append(states)
            reward_list.append(rewards)
            prob_list.append(probs)
            action_list.append(actions)
            value_list.append(values)

            states = next_states

            if np.any(dones):
                done = True
                break

        value_arr = torch.stack(value_list)
        reward_arr = torch.FloatTensor(np.array(reward_list)[:, :, np.newaxis])

        advantage_list = []
        return_list = []

        _, next_value = self.policy(torch.FloatTensor(states).to(self.device))
        returns = next_value.detach()

        advantages = torch.FloatTensor(np.zeros((self.num_agents, 1)))
        for i in reversed(range(len(state_list))):
            returns = reward_arr[i] + self.hyper_params['discount_rate'] * returns
            td_error = reward_arr[i] + self.hyper_params['discount_rate'] * next_value - value_arr[i]
            advantages = advantages * self.hyper_params['gae_tau'] * self.hyper_params['discount_rate'] + td_error
            next_value = value_arr[i]
            advantage_list.insert(0, advantages.detach())
            return_list.insert(0, returns.detach())

        return_arr = torch.stack(return_list)
        indices = return_arr >= np.percentile(return_arr, self.hyper_params['curation_percentile'])
        indices = torch.squeeze(indices, dim=2)

        advantage_arr = torch.stack(advantage_list)
        state_arr = torch.stack(state_list)
        prob_arr = torch.stack(prob_list)
        action_arr = torch.stack(action_list)

        self.memory.add({'advantages': advantage_arr[indices],
                         'states': state_arr[indices],
                         'log_probs_old': prob_arr[indices],
                         'returns': return_arr[indices],
                         'actions': action_arr[indices]})

        rewards = np.sum(np.array(reward_list), axis=0)
        return rewards, done

    def update(self):
        advantages_batch, states_batch, log_probs_old_batch, returns_batch, actions_batch = self.memory.categories()
        actions_batch = actions_batch.detach()
        log_probs_old_batch = log_probs_old_batch.detach()
        advantages_batch = (advantages_batch - advantages_batch.mean()) / advantages_batch.std()

        batch_indices = self.memory.sample()

        # Gradient ascent
        for _ in range(self.hyper_params['num_epochs']):
            for batch_idx in batch_indices:
                batch_idx = torch.LongTensor(batch_idx)

                advantages_sample = advantages_batch[batch_idx]
                states_sample = states_batch[batch_idx]
                log_probs_old_sample = log_probs_old_batch[batch_idx]
                returns_sample = returns_batch[batch_idx]
                actions_sample = actions_batch[batch_idx]

                dist, values = self.policy(states_sample)

                log_probs_new = dist.log_prob(actions_sample.to(self.device)).sum(-1).unsqueeze(-1)
                entropy = dist.entropy().sum(-1).unsqueeze(-1).mean()
                value_function_loss = (returns_sample - values).pow(2).mean()

                ratio = (log_probs_new - log_probs_old_sample).exp()
                clipped_ratio = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param)
                clipped_surrogate_loss = -torch.min(ratio * advantages_sample, clipped_ratio * advantages_sample).mean()
                loss = clipped_surrogate_loss - self.beta * entropy + self.hyper_params['value_pred_loss_coefficient']\
                    * value_function_loss

                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.hyper_params['gradient_clip'])
                self.opt.step()

        # Reduce clipping and exploration
        self.clip_param *= .999
        self.beta *= .995


class Memory:
    def __init__(self, size, mini_batch_size):
        self.keys = ['advantages', 'states', 'log_probs_old', 'returns', 'actions']
        self.size = size
        self.mini_batch_size = mini_batch_size
        self.reset()

    def add(self, data):
        for k, vs in data.items():
            for i in range(vs.size()[0]):
                getattr(self, k).append(vs[i].unsqueeze(0))

    def reset(self):
        for key in self.keys:
            setattr(self, key, deque(maxlen=self.size))

    def categories(self, keys=['advantages', 'states', 'log_probs_old', 'returns', 'actions']):
        data = [list(getattr(self, k))[:] for k in keys]
        return map(lambda x: torch.cat(x, dim=0), data)

    def sample(self):
        batch_indices = np.random.permutation(len(getattr(self, self.keys[0])))[:len(getattr(
            self, self.keys[0])) // self.mini_batch_size * self.mini_batch_size].reshape(-1, self.mini_batch_size)
        return batch_indices
