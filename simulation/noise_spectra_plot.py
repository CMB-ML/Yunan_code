import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import noise_model_fit as nmf
from scipy.optimize import curve_fit

def noise_model_c (ell, ell_c, alpha, beta, gamma, delta, A, B):
    n_ell = A*(100/ell)**alpha + B*((ell/1000)**beta)/(1 + (ell/ ell_c)**gamma)**delta
    return n_ell


#Global Path
noise_dir = '/shared/data/Datasets/I_512_1-10_NoiseBreakout/Simulation/Test/sim0000/'
noise_asset_dit = '/home/yunan/planck_release/'

#Dictionary for the noise maps path
noise_path = {'30': noise_dir + 'noise_30_map.fits',
              '44': noise_dir + 'noise_44_map.fits',
              '70': noise_dir + 'noise_70_map.fits',
              '100': noise_dir + 'noise_100_map.fits',
              '143': noise_dir + 'noise_143_map.fits',
              '217': noise_dir + 'noise_217_map.fits',
              '353': noise_dir + 'noise_353_map.fits',
              '545': noise_dir + 'noise_545_map.fits',
              '857': noise_dir + 'noise_857_map.fits'
              }

planck_noise_path = {'30' : noise_asset_dit + 'ffp10_noise_030_full_map_mc_00000.fits',
                     '100': noise_asset_dit + 'ffp10_noise_100_full_map_mc_00000.fits',
                     '143': noise_asset_dit + 'ffp10_noise_143_full_map_mc_00000.fits'
                    }

#Load the noise maps and calculate the noise spectra
noise_map = {}
planck_noise_map = {}
noise_spectra = {}
planck_noise_spectra = {}
for freq in planck_noise_path:
    planck_noise_map[freq] = hp.read_map(planck_noise_path[freq])
    noise_map[freq] = hp.read_map(noise_path[freq])
    noise_spectra[freq] = hp.anafast(noise_map[freq])
    planck_noise_spectra[freq] = hp.anafast(planck_noise_map[freq])

freq_check = ['100', '143']
sampling_params = {}
#MCfitting parameters
initial_params = { '100' :  np.array([9.21526879e+01, 1.85794293e+00, 2.34682283e-01, 1.47763124e-06, 1.95499578e+01, 1.93578004e-17, 2.85518495e-10]),
                   '143' :  np.array([1.38260136e+01, 1.77422483e+00, 9.42788557e+00, 3.21724781e+01, 2.93346616e-01, 2.53080481e-17, 3.51471283e+01])
                 }
steps = 500000
step_vec = {'100' : np.array([1e-1, 1e-2, 1e-3, 1e-8, 1e-1, 1e-19, 1e-12]),
            '143' : np.array([1e-1, 1e-2, 1e-1, 1e-1, 1e-3, 1e-19, 1e-2])
        }

for freq in freq_check:
    #Test MCfitting class
    ells = np.arange(len(planck_noise_spectra[f'{freq}']))
    ps = planck_noise_spectra[f'{freq}']
    fitting_instance = nmf.MCfitting(ells[2:], ps[2:])
    #run the mcmc_sampler
    samples = fitting_instance.mcmc_sampler(initial_params[f'{freq}'], steps, step_vec[f'{freq}'])
    print(samples[-1])
    sampling_params[f'{freq}'] = samples[-1]

    #curve_fit test
    fitting_instance = nmf.curvefit(ells[2:], ps[2:])
    popt, pcov = fitting_instance.run_fit(steps)

    #Plot variables
    dot_size = 3
    ells_p = np.arange(len(planck_noise_spectra[f'{freq}']))
    ells_s = np.arange(len(noise_spectra[f'{freq}']))
    hp.mollview(planck_noise_map[f'{freq}'], title=f'Planck Noise {freq} GHz')
    hp.mollview(noise_map[f'{freq}'], title=f'Simulated Noise {freq} GHz')

    plt.figure()
    plt.plot(ells_p[2:], noise_model_c(ells_p[2:],  *sampling_params[f'{freq}']), label='Fitted Model:sampling', c = 'red' )
    plt.plot(ells_p[2:], noise_model_c(ells_p[2:],  *popt), label='Fitted Model:curve_fit', c = 'black')
    plt.scatter(ells_p, (planck_noise_spectra[freq]), s = dot_size, label = 'Planck Noise')
    plt.scatter(ells_s ,(noise_spectra[freq]), s =  dot_size, label = 'Simulated Noise')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('$\ell$')
    plt.ylabel('$N_{\ell}$')
    plt.title('Noise Spectra at ' + freq + ' GHz')
    plt.legend()
    plt.show()

