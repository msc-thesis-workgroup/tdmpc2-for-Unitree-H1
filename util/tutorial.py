#!/usr/bin/env python3
"""
Example for how to modifying the MuJoCo qpos during execution.
"""

import os
from mujoco_py import load_model_from_xml, MjSim, MjViewer
import time

import xml.etree.ElementTree as ET

MODEL_XML = "/home/davide/.mujoco/mujoco-py/unitree_h1/h1.xml"

# Check if the file exists
if not os.path.exists(MODEL_XML):
    print(f"The file {MODEL_XML} does not exist.")
else:
    # Check if the file is not empty
    if os.path.getsize(MODEL_XML) == 0:
        print(f"The file {MODEL_XML} is empty.")
    else:
        # Validate the XML
        try:
            ET.parse(MODEL_XML)
            print(f"The file {MODEL_XML} is a valid XML file.")
        except ET.ElementTree.ParseError as e:
            print(f"The file {MODEL_XML} is not a well-formed XML file. Error: {e}")

# Load the model
model = load_model_from_xml(open(MODEL_XML).read())



sim = MjSim(model)
viewer = MjViewer(sim)


# get the z-coordinate of the elbow frame
# right_elbow_id = sim.model.body_name2id('right_elbow_link')
# left_elbow_id = sim.model.body_name2id('left_elbow_link')

# Print all the existing x_pos
print(sim.data.body_xpos)


# Set the value of the qpos of the right_shoulder_roll
# right_shoulder_roll_id = sim.model.joint_name2id('right_shoulder_roll')
# print("right_shoulder_roll_id: ", right_shoulder_roll_id)
# set the joint valu

# Get all the joint names
for joint_name in sim.model.joint_names:
    joint_id = sim.model.joint_name2id(joint_name)
    print(f"Joint name: {joint_name}")
    print(f"Joint type: {sim.model.jnt_type[joint_id]}")
    print(f"Joint damping: {sim.model.dof_damping[joint_id]}")
    print(f"Joint limited: {sim.model.jnt_limited[joint_id]}")
    print(f"Joint range: {sim.model.jnt_range[joint_id]}")
    print("----")
#sim.data.qpos[right_shoulder_roll_id] = 1.5

left_shoulder_roll_addr = sim.model.get_joint_qpos_addr('left_shoulder_roll')

print("left_shoulder_roll_id: ", left_shoulder_roll_addr)


sim.data.qpos[left_shoulder_roll_addr] = 1.5

# Print joint configurations
# for joint_name in sim.model.joint_names:
#     joint_id = sim.model.joint_name2id(joint_name)
#     print(f"Joint name: {joint_name}")
#     print(f"Joint type: {sim.model.jnt_type[joint_id]}")
#     print(f"Joint damping: {sim.model.dof_damping[joint_id]}")
#     print(f"Joint limited: {sim.model.jnt_limited[joint_id]}")
#     print(f"Joint range: {sim.model.jnt_range[joint_id]}")
#     print("----")

value = 1.0
while True:
    # how to set the qpos of the elbow
    # right_elbow_qpos_idx = sim.model.get_joint_qpos_addr('right_elbow')
    # sim.data.qpos[right_elbow_qpos_idx] = value

    
    sim.forward()

    # Advance the simulation
    # sim.step()
    
    viewer.render()
    #value += 0.1
    #time.sleep(0.01)
    if os.getenv('TESTING') is not None:
        break