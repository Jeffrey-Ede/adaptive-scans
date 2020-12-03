import numpy as np

def ornstein_uhlenbeck(input, theta=0.1, sigma=0.2):
    """Ornstein-Uhlembeck perturbation. Using Gaussian Wiener process."""
    noise_perturb = -theta*input + sigma*np.random.normal()
    return input + noise_perturb

noise = 0
for _ in range(20):
    noise = ornstein_uhlenbeck(noise)
    print(noise/(np.pi))