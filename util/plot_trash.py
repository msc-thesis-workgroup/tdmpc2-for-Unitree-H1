import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PATH_TO_FILE = "/home/davide/tdmpc2/tdmpc2/logs/h1-walk-v2/1/tdmpc/videos/h1-walk-v2-0.csv"

df = pd.read_csv(PATH_TO_FILE)

# I want to compare com_velocity_x and robot_velocity_x
# plt.figure()
# plt.plot(df["com_velocity_x"], label="com_velocity_x")
# plt.plot(df["robot_velocity_x"], label="robot_velocity_x")
# plt.legend()
# plt.title("com_velocity_x vs robot_velocity_x")
# plt.xlabel("time")
# plt.ylabel("velocity")

# I want to compare com_velocity_x and move
# plt.figure()
# plt.plot(df["com_velocity_x"], label="com_velocity_x")
# plt.plot(df["move"], label="move")
# plt.legend()
# plt.title("com_velocity_x vs move")
# plt.xlabel("time")
# plt.ylabel("velocity")

# I want to compare com_position_y and robot_position_y
# plt.figure()
# plt.plot(df["com_position_y"], label="com_position_y")
# plt.plot(df["robot_position_y"], label="robot_position_y")
# plt.legend()
# plt.title("com_position_y vs robot_position_y")
# plt.xlabel("time")
# plt.ylabel("position")

# I want to compare angle_x and stay_inline_reward
plt.figure()
plt.plot(df["orientation_x"], label="orientation_x")
plt.plot(df["stay_inline_reward"], label="stay_inline_reward")
plt.legend()
plt.title("orientation_x vs stay_inline_reward")
plt.xlabel("time")
plt.ylabel("angle")


plt.tight_layout()
plt.show()