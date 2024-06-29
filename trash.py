import numpy as np
from scipy.stats import multivariate_normal


def create_multivariate_gaussian_function(state:np.array, variance: float): # TODO: Consider moving this to a utility class with many other typical functions
    """
    Create a multivariate Gaussian function centered at the given state with the specified variance.

    Parameters:
    - state: A NumPy array of length 59, serving as the mean of the multivariate Gaussian distribution.
    - variance: The variance for the Gaussian distribution, applied uniformly across all dimensions.

    Returns:
    - A function that calculates the multivariate Gaussian distribution's PDF for a given input x.
    """
    # Create a covariance matrix with the specified variance along the diagonal
    covariance_matrix = np.eye(len(state)) * variance
    
    # Define the multivariate normal distribution
    mv_normal = multivariate_normal(mean=state, cov=covariance_matrix)
    
    # Return the PDF function of the distribution
    return mv_normal.pdf

# create the pf 
state = np.zeros(59)
variance = 0.1
pdf = create_multivariate_gaussian_function(state, variance)

pf = pdf(state)
# normalize the pdf

print("PDF:")
print(pdf)

# take 10000 samples from the pdf
#x = np.linspace(0, 100, 1000)
samples = np.random.choice(state, size=(1000,59), p=pdf)
print("Samples from the pdf:")
print(samples)

print("Mean of the samples:")
print(samples.mean()) # expected output: 0.0
