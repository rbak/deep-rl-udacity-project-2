import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.weight.data.mul_(0.001)
        nn.init.constant_(m.bias.data, 0)


class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs, seed, hidden_size_1=64, hidden_size_2=64):
        super(ActorCritic, self).__init__()

        self.seed = torch.manual_seed(seed)

        self.shared = nn.Sequential(
            nn.Linear(num_inputs, hidden_size_1),
            nn.ReLU(),
            nn.Linear(hidden_size_1, hidden_size_2),
            nn.ReLU(),
        )

        self.actor = nn.Sequential(
            nn.Linear(hidden_size_2, num_outputs),
            nn.Tanh()
        )
        self.actor.apply(init_weights)

        self.critic = nn.Linear(hidden_size_2, 1)
        self.critic.apply(init_weights)

        self.std = nn.Parameter(torch.zeros(num_outputs))

    def forward(self, x):
        x = torch.FloatTensor(x)
        x = self.shared(x)
        value = self.critic(x)
        mu = self.actor(x)
        dist = Normal(mu, F.softplus(self.std))

        return dist, value
