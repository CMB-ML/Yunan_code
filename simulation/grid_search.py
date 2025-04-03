import numpy as np
import healpy as hp
import itertools

planck_noise_path = {'30' : '/home/yunan/planck_release/ffp10_noise_030_full_map_mc_00000.fits',
                     '100': '/home/yunan/planck_release/ffp10_noise_100_full_map_mc_00000.fits',
                     '143': '/home/yunan/planck_release/ffp10_noise_143_full_map_mc_00000.fits',}
planck_noise_map = {}
for frequency in planck_noise_path:
    planck_noise_map[frequency] = hp.read_map(planck_noise_path[frequency])

frequencys = [100, 143]
planck_noise_spectra = {}
for frequency in frequencys:
    planck_noise_spectra[f'{frequency}'] = hp.anafast(planck_noise_map[f'{frequency}'])

# Define the noise model function
def noise_model(ell, params):
    ell_c, alpha, beta, gamma, delta, A, B = params
    n_ell = A * (100 / ell) ** alpha + B * ((ell / 1000) ** beta) / (1 + (ell / ell_c) ** gamma) ** delta
    return n_ell

# Define the objective function (e.g., mean squared error)
def objective_function(params, ell, data_y):
    model_y = noise_model(ell, params)
    loglike = - np.sum(np.log((data_y - model_y) ** 2))
    return loglike

steps = 3

# Define the parameter ranges for the grid search
param_ranges = {
    'ell_c': np.linspace(50, 150, steps),
    'alpha': np.linspace(1, 3, steps),
    'beta': np.linspace(0.1, 0.5, steps),
    'gamma': np.linspace(1e-7, 1e-5, steps),
    'delta': np.linspace(10, 20, steps),
    'A': np.linspace(1e-18, 1e-16, steps),
    'B': np.linspace(1e-11, 1e-9, steps)
}

# Generate all combinations of parameters
param_combinations = list(itertools.product(*param_ranges.values()))

# Example data (replace with actual data)
ell = np.linspace(2, 6144, 6142)
data_y = planck_noise_spectra['100'][2:]

# Perform grid search
best_params = None
best_mse = float('inf')

for params in param_combinations:
    sqe = objective_function(params, ell, data_y)
    if sqe < best_mse:
        best_mse = sqe
        best_params = params

# Output the best parameters and the corresponding MSE
print("Best parameters:", best_params)
print("Best MSE:", best_mse)