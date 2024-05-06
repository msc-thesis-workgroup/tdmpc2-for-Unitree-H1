## How to evaluate the model
To evaluate the model, you need to specify the path to the model file you want to evaluate (checkpoint parameter) and the save_video argument. The save_video argument is used to save the video of the evaluation. The command to evaluate the model is the following:
```bash
python evaluate.py checkpoint=checkpoint_path save_video=boolean
```
An example of how to evaluate the model is the following:
```bash
python evaluate.py checkpoint=/home/davide/tdmpc2/tdmpc2/logs/humanoid_h1-walk-v0/1/tdmpc/models/base_3-2024-04-30-22-38-59/step-775175.pt save_video=true
```
## How to train the model from scratch
To train the model from scratch, you need to specify the task you want to train the model on. The command to train the model from scratch is the following:
```bash
python train.py task=humanoid_h1-walk-v0 experiment_name=experiment_name
```
## How to train the model from a checkpoint
To train the model from a checkpoint, you need to specify the path to the checkpoint file and the experiment name. The experiment name is used to create a new directory in the outputs folder where the new experiment will be saved. The command to train the model from a checkpoint is the same as the one used to train the model from scratch, with the addition of the checkpoint and experiment_name arguments. The command template to train the model from a checkpoint is the following:
```bash
python train.py task=humanoid_h1-walk-v0 checkpoint=checkpoint_path experiment_name=experiment_name
```
An example of how to train the model from a checkpoint is the following:
```bash
python train.py task=humanoid_h1-walk-v0 checkpoint=/home/davide/tdmpc2/tdmpc2/logs/humanoid_h1-walk-v0/1/tdmpc/models/base/step-750465.pt experiment_name=testing
```
WARNING!!! Notice in the same directory of the checkpoint file there should be:
- a file named "buffer.pkl" containing the replay buffer parameters
- a directory named "buffer_buffer.pkl" containing the replay buffer data
- a file named "eval.csv" (not mandatory) containing the evaluation data. It can be useful for plotting the training data.

All these files are automatically saved by the training script. So, you have to be careful to not delete them and specify a valid path that contains all these files. Notice that the model (the *.pt file) is the only mandatory file to train the model from a checkpoint, but the training will be strongly unstable without the replay buffer data.
 
