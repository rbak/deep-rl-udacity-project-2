# Project 2 Report

## Algorithm
The learning algorithm is a PPO actor-critic with a clipping surrogate function and gae.

## Hyperparameters
Most of the hyperparameter values I used came from the original PPO paper. (https://arxiv.org/abs/1707.06347)

  * memory_size: 20000,      # replay buffer size
  * batch_size: 64,          # sample batch size
  * t_random: 3,             # random steps at start of trajectory
  * t_max: 1000,             # trajectory length
  * num_epochs: 10,          # number of updates
  * c_vf: 0.5,               # coefficent for vf loss (c1)
  * c_entropy: 0.001,        # starting value for coefficent for entropy (c2)
  * epsilon: 0.2,            # starting value for clipping parameter
  * gae_param: 0.95,         # gae param (λ)
  * discount: .99,           # discount (γ)
  * curation_percentile: 0,  # percent of trajectory data to drop
  * gradient_clip: 5,        # gradient clip

## Model Architecture
The model architecture is a combined actor-critic with two hidden layers, both of size 64.

## Rewards
With multiple agents, only about 120-130 episodes were required to consistently perform better than the goal.

![Final Results](https://github.com/rbak/deep-reinforcement-learning-project-2/blob/master/results/final.jpeg)

Over multiple runs the agent varied in how quickly it learned the model, but constently converged around the same timestep.

![Results Summary](https://github.com/rbak/deep-reinforcement-learning-project-2/blob/master/results/summary.jpeg)


## Future Improvements
Although they are not strictly improvements to my implementation, I would be interested in trying some of the other algorithms to see how they compare.  I initially implemented DDPG for this project but was unable to get it working, so I would like to revisit that, as well as implementing A3C and potentially the Q-prop algorithm that was mentioned, but not covered.  I would also like to get the single agent version of the evironment working, as none of the algorithms I tried on that environment learned.
