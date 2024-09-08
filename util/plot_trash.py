import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PATH_TO_FILE = "/home/davide/tdmpc2/tdmpc2/logs/h1-hybrid_walk-v0/52/tdmpc/videos/h1-hybrid_walk-v0-0.csv"

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

# I want to plot many subfigures: angle_x, centered_reward, stay_inline_reward, move and small_control
plt.figure()
plt.subplot(5,1,1)
plt.plot(df["angle_x"], label="angle_x")
plt.legend()
plt.title("angle_x")
plt.xlabel("time")
plt.ylabel("angle")
# add grid
plt.grid()
# plot two horizontal lines -8 and 8
plt.axhline(y=-8, color='r', linestyle='-', label="-8")
plt.axhline(y=8, color='r', linestyle='-', label="8")


plt.subplot(5,1,2)
plt.plot(df["centered_reward"], label="centered_reward")
plt.legend()
plt.title("centered_reward")
plt.xlabel("time")
plt.ylabel("reward")

plt.subplot(5,1,3)
plt.plot(df["stay_inline_reward"], label="stay_inline_reward")
plt.legend()
plt.title("stay_inline_reward")
plt.xlabel("time")
plt.ylabel("reward")

plt.subplot(5,1,4)
plt.plot(df["move"], label="move")
plt.legend()
plt.title("move")
plt.xlabel("time")
plt.ylabel("reward")

plt.subplot(5,1,5)
plt.plot(df["small_control"], label="small_control")
plt.legend()
plt.title("small_control")
plt.xlabel("time")
plt.ylabel("reward")


# save the figure
plt.savefig("h1-hybrid_walk-v0-0.png")
#plt.tight_layout()
plt.show()