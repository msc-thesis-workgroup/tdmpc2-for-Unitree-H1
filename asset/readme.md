# Description
This folder contains all the robot models and the world files used in the simulation. The models are taken from Mujoco Menagerie project.

# How to add a new robot model
First, you have to create a new folder in `asset/` with the name in UPPERCASE of the robot you want to simulate. Then, you have to create a new folder in the robot folder with the name of the world you want to simulate.

In the robot model you have to create a `scene.xml` which contains or include the robot model and the world model.

# Example
If you want to simulate the robot H1 from Unitree Robotics, create a folder "H1" in `asset/`. In the folder you have just created, `asset/H1`, create `scene.xml` file that contains the world model and the robot model or include it.

