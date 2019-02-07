# Project 2 Report

## Algorithm
The learning algorithm is a PPO actor-critic.

## Hyperparameters
I initially implemented DDPG for this project, but could not get it to perform well regardless of the hyperparameters I used.  As a result, when I got an implementation of PPO working, I didn't mess with the hyperparameters much as I didn't want to break the model. As far as I could tell my model learned to get a near perfect score, however I would like to go back and play with the hyperparameters to see if I could influence the initial learning rate.

  * (memory_size): 20000
  * (batch_size): 64
  * (gae_tau): 0.95
  * (ppo_ratio_clip): 0.2
  * (entropy_weight): 0
  * (optimization_epochs): 10
  * (discount_rate): .99
  * (traj_coll_random_steps): 3
  * (value_pred_loss_coefficient): 0.5
  * (clip_param): 0.2
  * (beta): 0.001
  * (tmax): 1000
  * (num_epochs): 10
  * (curation_percentile): 0
  * (gradient_clip): 5

## Model Architecture
The model architecture is a combined actor-critic with two hidden layers, both of size 64.

## Rewards
Reward summary for the chosen model, trained over 2000 episodes.
The agent required only about 120-130 episodes to consistently perform better than the goal.

![Final Results](https://github.com/rbak/deep-reinforcement-learning-project-2/blob/master/results/final.jpeg)

Over multiple runs the agent varied in how quickly it learned the model, but constently converged around the same timestep.

![Results Summary](https://github.com/rbak/deep-reinforcement-learning-project-2/blob/master/results/summary.jpeg)


## Future Improvements
Although they are not strictly improvements to my implementation, I would be interested in trying some of the other actor critic algorithms to see how they compare.  I initially implemented DDPG for this project but was unable to get it working, so I would like to revisit that, as well as implementing A3C and potentially the Q-prop algorithm that was mentioned, but not covered.
