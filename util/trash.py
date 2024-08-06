import numpy as np
from dm_control.utils import rewards


move_speed = 1.44 # 5.2 km/h
move_speed_lower_bound = 1.11 # 4 km/h 
move_speed_upper_bound = 1.78 # 6.4 km/h
com_velocity = np.arange(0, 3, 0.1)
move = rewards.tolerance(
            com_velocity,
            bounds=(move_speed_lower_bound, move_speed_upper_bound),
            margin=0.4,
            value_at_margin=0.1,
            sigmoid="gaussian",
        ) 

# plot the reward function for the given range
import matplotlib.pyplot as plt
plt.plot(com_velocity, move)
plt.show()
