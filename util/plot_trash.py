import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ast import literal_eval

PATH_TO_FILE = "/home/davide/tdmpc2/tdmpc2/logs/humanoid_h1-walk-v0/1/tdmpc/videos/humanoid_h1-walk-v0-0.csv"

# Function to convert string representation of arrays into numpy arrays
def str_to_array(s):
    # Remove brackets and split the string by spaces
    elements = s.strip("[]").split()
    # Convert each element to float and create a numpy array
    return np.array([float(e) for e in elements])
# Read the CSV file
df = pd.read_csv(PATH_TO_FILE, converters={'lin_pos': str_to_array})#, 'quat_pos': str_to_array, 'lin_vel': str_to_array, 'ang_vel': str_to_array})

# Plotting
fig, axs = plt.subplots(4, 1, figsize=(10, 8))

# Linear Position
labels = ['x', 'y', 'z']
for i in range(3):
    axs[0].plot(df.index, df['lin_pos'].apply(lambda x: x[i]), label=f'Component {labels[i]}')
axs[0].set_title('Linear Position')
axs[0].legend()

# Quaternion Position
# labels = ['w', 'x', 'y', 'z']
# for i in range(4):
#     axs[1].plot(df.index, df['quat_pos'].apply(lambda x: x[i]), label=f'Component {labels[i]}')
# axs[1].set_title('Quaternion Position')
# axs[1].legend()

# Linear Velocity
# labels = ['v_x', 'v_y', 'v_z']
# for i in range(3):
#     axs[2].plot(df.index, df['lin_vel'].apply(lambda x: x[i]), label=f'Component {labels[i]}')
# axs[2].set_title('Linear Velocity')
# axs[2].legend()

# # Angular Velocity
# labels = ['w_x', 'w_y', 'w_z']
# for i in range(3):
#     axs[3].plot(df.index, df['ang_vel'].apply(lambda x: x[i]), label=f'Component {labels[i]}')
# axs[3].set_title('Angular Velocity')
# axs[3].legend()

plt.tight_layout()
plt.show()