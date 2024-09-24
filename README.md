<h1>TD-MPC2 for Unitree Robotics H1</span></h1>

This repository provides an implementation of TD-MPC2 for the humanoid Unitree H1 robot, with the objective of training and evaluating TD-MPC2 agents on a variety of continuous control tasks for the humanoid Unitree H1 robot. 
In particular, it is a fork of the official implementation of TD-MPC2 by Nicklas Hansen, Hao Su, and Xiaolong Wang. The original repository can be found [here](https://github.com/nicklashansen/tdmpc2). 

The repository contains the following main components:
- the TD-MPC2 algorithm
- the Unitree H1 robot model
- the code to create an environment for the Unitree H1 robot
- the training and evaluation scripts for the Unitree H1 robot
---

## Overview

As reported in the official repo, TD-MPC**2** is a scalable, robust world model for continuous control tasks. It is also a model-based reinforcement learning algorithm that learns a world model from data collected by a model-free agent. The world model is then used to plan and execute actions in the environment.

---

## Getting started

A machine with a GPU and at least 12 GB of RAM with TD-MPC**2** is required for training and evaluation. Single-task online RL and evaluation require a GPU with at least 4 GB of memory. 
Multi-task online and offline RL has not been tested.

I provide a `pyproject.toml` with all the required libraries. 

More instructions on how to install the required libraries will be provided in the future. Or you can refer to the official repo for more information.

----


## Example usage

We provide examples on how to evaluate our provided TD-MPC**2** checkpoints, as well as how to train your own TD-MPC**2** agents, below.

----

### Evaluation

To evaluate the model, you need to specify the path to the model file you want to evaluate (checkpoint parameter) and the save_video argument. The save_video argument is used to save the video of the evaluation. The command to evaluate the model is the following:
```bash
python evaluate.py task=robot_name-task_name-reward_version checkpoint=checkpoint_path save_video=boolean
```
An example of how to evaluate the model is the following:
```bash
python evaluate.py task=h1-walk-v1 checkpoint=/path/to/file/step-0.pt save_video=true
```

----
### Training

#### How to train the model from scratch
To train the model from scratch, you need to specify the task you want to train the model on. The command to train the model from scratch is the following:
```bash
python train.py task=h1-walk-v1 experiment_name=experiment_name
```
#### How to train the model from a checkpoint
To train the model from a checkpoint, you need to specify the path to the checkpoint file and the experiment name. The experiment name is used to create a new directory in the outputs folder where the new experiment will be saved. The command to train the model from a checkpoint is the same as the one used to train the model from scratch, with the addition of the checkpoint and experiment_name arguments. The command template to train the model from a checkpoint is the following:
```bash
python train.py task=robot_name-task_name-reward_version checkpoint=checkpoint_path experiment_name=experiment_name
```
An example of how to train the model from a checkpoint is the following:
```bash
python train.py task=h1-walk-v1 checkpoint=/path/to/file/step-0.pt experiment_name=testing
```
or
```bash
python3 train.py task=h1-hybrid_walk-v0 frame_skip=5 max_episode_steps=1000 seed=52 checkpoint=/path/to/file/step-0.pt experiment_name=HybridWalkV0_resumeTrainig
```

WARNING!!! Notice in the same directory of the checkpoint file there should be:
- a file named "buffer.pkl" containing the replay buffer parameters
- a directory named "buffer_buffer.pkl" containing the replay buffer data
- a file named "eval.csv" (not mandatory) containing the evaluation data. It can be useful for plotting the training data.

All these files are automatically saved by the training script. So, you have to be careful to not delete them and specify a valid path that contains all these files. Notice that the model (the *.pt file) is the only mandatory file to train the model from a checkpoint, but the training will be strongly unstable without the replay buffer data.

---

## License

This project is licensed under the MIT License - see the `LICENSE` file for details. Note that the repository relies on third-party code, which is subject to their respective licenses.

---

## References

- The repository contains code from the TD-MPC2 repository, which was the starting point for the project.

- The Unitree Robotics H1 robot is a humanoid robot developed by Unitree Robotics. The robot model is taken from MuJoCo Menagerie, which is a collection of MuJoCo environments and models.

- Few modules code ware inspired by the HumanoidBench repository. However, the objectives of this project are completely different from the HumanoidBench project and the code was modified to fit the objectives of this project. 
--- 

## Known issues

When you evaluate the model, the program will save the video correctly, but it could crash in the end. This is due to a bug in the code that will be fixed in the future. The evaluation is still correct, and the video will be saved correctly.