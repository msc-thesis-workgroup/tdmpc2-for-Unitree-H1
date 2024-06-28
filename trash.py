import numpy as np
import os
# x = np.array([1,2,3])
# y = np.array([4,5,6])
# z = np.array([7,8,9])

# # create a numpy vector where x is the "key" and the value is 0

# weights = np.zeros((x.shape[0]+1, 5))
# print(weights)

# print(weights[1])
# weights[x] = 1


# weights[y] = 2
# weights[z] = 3

# print(weights)

behaviors_name = r"salerno_behaviors.npy"

###This is where the behaviors are loaded
behaviors = np.load(behaviors_name) 

fullDim = behaviors.shape[1]
N_contribs = behaviors.shape[0]

print("fullDim: ", fullDim)
print("N_contribs: ", N_contribs)
print("behaviors: ", behaviors.shape)

print(behaviors[0,0,0])