# Deep Reinforcement Learning Project 2
This is an implementation of PPO for the second project in Udacity's Deep Reinforcement Learning class.  The model is built and trained to solve Unity's [Reacher Environment](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) scenario.


## Environment Details
The Reacher learning environment has the following properties:

  * State Space - The state space for each agent has 33 dimensions.
  * Action Space - There action space has four dimensions, each with a continuous value between -1 and 1.
  * Goal - The goal for this project as to achieve a mean of 30 points across all agents for 100 episodes.

For more information see the [Unity Github repo](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher)

## Agent Details
The agent uses a PPO actor-critic network.  It is implemented in python3 using PyTorch, and uses a two hidden layers.

## Installation Requirements
  1. Create a python 3.6 virtual environment.  I used Anaconda for this. (`conda create -n yourenvname python=3.6`)
  2. After activating the environment, pip install the requirements file. (`pip install -r requirements.txt`)

Note: The agent is setup to use the MacOS Multiagent_Reacher.app environment file.

## Running
Run main.py from the repo directory. The main.py file can be run with any of the following flags.
e.g. `python main.py --test`

* `--examine`: Prints information on the learning environment.
* `--random`: Runs an agent that takes random actions.
* `--train`: Trains a new agent including saving a checkpoint of the model weights and printing out scores.
* `--test`: Runs an agent using the included checkpoint file.
* `--single`: Runs the single agent version of the reacher environment. The multi agent version runs by default.

Rather than the standard matplotlib plots from my last project, I set this one up to push the results to comet.ml
If you want to see a graph of the results from training, you can setup an account there and use the following flag.

* `--log`: Logs the results to comet.ml.  Must have set the environment variable COMET_API_KEY.
