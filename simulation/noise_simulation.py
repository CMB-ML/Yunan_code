import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import noise_model_fit as nmf
from scipy.optimize import curve_fit

def fn_format(dir_Path, frequency, seed):
    return f'{dir_Path}/ffp10_noise_{frequency}_full_map_mc_{seed:05d}.fits'

def format_plot(ax, x, y, yerr=None, label=None, color=None, fill_between=None, fill_color=None):
    """Function to format plots uniformly."""
    if yerr is not None:
        ax.errorbar(x, y, yerr=yerr, label=label, fmt='o', markersize=3, color=color)
    else:
        ax.plot(x, y, label=label, color=color)
    
    if fill_between is not None:
        ax.fill_between(x, fill_between[0], fill_between[1], color=fill_color, alpha=0.3, label='1σ Interval')
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    ax.set_xlabel('$\ell$')
    ax.set_ylabel('$N_{\ell}$')

# Configuration
sim_noise_dir_path = '/home/yunan/planck_release'
frequency = 100  # Frequency of the noise map
seeds = np.arange(10)  # MC Seeds
nside = 2048
ellmin = 1

# Paths to noise maps
planck_noise_path = {f'{seed}': fn_format(sim_noise_dir_path, frequency, seed) for seed in seeds}

# Initialize variables
planck_noise_spectra = {}
planck_noise_map = {}
ps_total = np.zeros(3*nside)
popt_list = {}
pcov_list = {}

# Average the noise spectra
for seed in planck_noise_path:
    planck_noise_map[seed] = hp.read_map(planck_noise_path[seed])
    planck_noise_spectra[f'{seed}'] = hp.anafast(planck_noise_map[f'{seed}'])
    if seed == '0':
        ells_2 = np.arange(len(planck_noise_spectra[f'{seed}']))
    ps_total += planck_noise_spectra[f'{seed}']  # Accumulate total power spectrum
    ps = planck_noise_spectra[f'{seed}']
    cf = nmf.curvefit(ells_2[ellmin:], ps[ellmin:])
    popt, pcov = cf.run_fit(num_trial=5000000, model='polynomial')  # Fit individual power spectrum
    popt_list[f'{seed}'] = popt
    pcov_list[f'{seed}'] = pcov

# Compute average and standard deviation of the power spectra
ps_average = ps_total / len(planck_noise_spectra)
ps_std = np.std([planck_noise_spectra[f'{seed}'] for seed in planck_noise_spectra], axis=0)

# Fit the average power spectrum
cf = nmf.curvefit(ells_2[ellmin:], ps_average[ellmin:])
popt, pcov = cf.run_fit(num_trial=5000000, model='polynomial')
popt_list['average'] = popt
pcov_list['average'] = pcov

# Ensure popt_list['average'] has 11 parameters
if len(popt_list['average']) != 11:
    raise ValueError("popt_list['average'] must contain 11 parameters for the polynomial noise model")

# --- MCMC Sampling ----
# Initialize MCMC fitting class and sample the parameters
steps = 500000
step_vec = np.sqrt(np.diag(pcov_list['average']))
fitting_instance = nmf.MCfitting(ells_2[ellmin:], ps_average[ellmin:])
samples, likelihoods = fitting_instance.mcmc_sampler(popt_list['average'], ps_std[ellmin:], steps, step_vec, model='polynomial')

# Calculate the mean spectrum and uncertainties from MCMC samples
n_samples = len(samples)
spectra_samples_mcmc = np.zeros((n_samples, len(ells_2[ellmin:])))

# Compute spectra for each set of MCMC sampled parameters
for i in range(n_samples):
    spectra_samples_mcmc[i, :] = fitting_instance.polynomial_noise_model(ells_2[ellmin:], *samples[i])

# Calculate the mean and standard deviation from the MCMC samples
spectra_mean_mcmc = np.mean(spectra_samples_mcmc, axis=0)
spectra_std_mcmc = np.std(spectra_samples_mcmc, axis=0)

# Plot out to compare
fig, ax = plt.subplots(1, 1, figsize=(20, 12))
ells = np.arange(len(ps_average))

# Plot Planck noise average with error bars
format_plot(ax, ells, ps_average, yerr=ps_std, label='Planck Noise Average', color='black')

# Plot curve_fit result
cf_instance = nmf.curvefit(ells[ellmin:], ps_average[ellmin:])
format_plot(ax, ells[ellmin:], cf_instance.polynomial_noise_model(ells[ellmin:], *popt_list['average']), label='Average Fit')

# Plot MCMC mean spectrum with 1σ uncertainty
format_plot(ax, ells[ellmin:], spectra_mean_mcmc, fill_between=(spectra_mean_mcmc - spectra_std_mcmc, spectra_mean_mcmc + spectra_std_mcmc), label='MCMC Mean Spectrum', fill_color='blue')

# Format plot title
ax.set_title('Planck Noise Spectra fit Comparison')
plt.show()

# --- Monte Carlo Sampling for Noise Models with Variations ---
n_samples = len(samples)  # Number of MCMC samples
sample_mean = np.mean(samples, axis= 0)
sample_std = np.std(samples, axis = 0)
variated_samples = np.random.normal(loc= sample_mean,
                                    scale=10**10.5*sample_std, 
                                    size=(n_samples, len(sample_mean)))
cf_instance = nmf.curvefit(ells[1:], ps_average[1:])
spectra_samples_with_variations = np.array([cf_instance.polynomial_noise_model(ells[1:], *variated_samples[i]) for i in range(n_samples)])
# Calculate mean and standard deviation from the variated MCMC samples
spectra_mean_variated = np.mean(spectra_samples_with_variations, axis=0)
spectra_std_variated = np.std(spectra_samples_with_variations, axis=0)
# 68% confidence intervals (1 sigma) based on variated MCMC results
lower_bound_variated = spectra_mean_variated - spectra_std_variated
upper_bound_variated = spectra_mean_variated + spectra_std_variated

mean = popt_list['average']
std = np.sqrt(np.abs(np.diag(pcov_list['average'])))
variated_samples_2 = np.random.normal(loc=mean, 
                                      scale=std, 
                                      size=(n_samples, len(mean)))
cf_instance = nmf.curvefit(ells[1:], ps_average[1:])
spectra_samples_with_variations_2 = np.array([cf_instance.polynomial_noise_model(ells[1:], *variated_samples_2[i]) for i in range(n_samples)])
# Calculate mean and standard deviation from the variated MCMC samples
spectra_mean_variated_2 = np.mean(spectra_samples_with_variations_2, axis=0)
spectra_std_variated_2 = np.std(spectra_samples_with_variations_2, axis=0)
# 68% confidence intervals (1 sigma) based on variated MCMC results
lower_bound_variated_2 = spectra_mean_variated_2 - spectra_std_variated_2
upper_bound_variated_2 = spectra_mean_variated_2 + spectra_std_variated_2

# Plot the results with variations
plt.figure(figsize=(10, 6))

# Format plot for Planck Noise Average
format_plot(plt.gca(), ells, ps_average, yerr=ps_std, label='Planck Noise Average', color='black')

# Format plot for variated MCMC results
format_plot(plt.gca(), ells[1:], spectra_mean_variated, label = 'MCMC',fill_between=(lower_bound_variated, upper_bound_variated), fill_color='blue')

format_plot(plt.gca(), ells[1:], spectra_mean_variated_2, label = 'Curve fit',fill_between=(lower_bound_variated_2, upper_bound_variated_2), fill_color='red')


# Format plot title
plt.title('Planck Noise Spectra with Variated MCMC Fitting Results and curve fitting result')
plt.show()