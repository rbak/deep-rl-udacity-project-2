# Deep Reinforcement Learning Project 2
This is an implementation of PPO for the second project in Udacity's Deep Reinforcement Learning class.  The model is built and trained to solve Unity's [Reacher Environment](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) scenario.


## Environment Details


## Agent Details

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
