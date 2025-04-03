import numpy as np
import healpy as hp
from matplotlib import pyplot as plt
from astropy.io import fits
import math

def pse(map_path, ellmax, mask = None):
    map = hp.read_map(map_path)
    if mask is not None:
        map_ps = get_power(map,map,mask,lmax=ellmax)
    else:
        map_ps = hp.anafast(map,map,lmax=ellmax)
    return map_ps

def get_power(mapp1,mapp2,mask,lmax=None):
    mean1 = np.sum(mapp1*mask)/np.sum(mask)
    mean2 = np.sum(mapp2*mask)/np.sum(mask)
    fsky = np.sum(mask)/mask.shape[0]
    return hp.anafast(mask*(mapp1-mean1),mask*(mapp2-mean2),lmax=lmax)/fsky

def get_theory_ps(path):
    ps_theory = np.load(path)
    ell_theory = np.arange(len(ps_theory))
    return ell_theory, ps_theory

def ps2dl(ps,ell):
    return ell*(ell+1)*ps/(2*math.pi)

def dl_deconv(dl,beam):
    return dl/beam**2

def map_conv(map_in,nside,ellmax,beam):
    alm_in = hp.map2alm(map_in, lmax=ellmax)
    alm_conv = hp.almxfl(alm_in, beam[:ellmax])
    map_conv = hp.alm2map(alm_conv, nside=nside)
    return map_conv

def map_deconv(map_in,nside,ellmax,beam):
    alm_in = hp.map2alm(map_in, lmax=ellmax)
    alm_deconv = hp.almxfl(alm_in, 1/beam[:ellmax])
    map_deconv = hp.alm2map(alm_deconv, nside=nside)
    return map_deconv

def get_percent_diff(ps, benchenmark):
    temp_idx = len(ps)
    benchenmark = benchenmark[:temp_idx]
    return (ps - benchenmark)/benchenmark*100

def get_abs_diff(ps, benchenmark):
    temp_idx = len(ps)
    benchenmark = benchenmark[:temp_idx]
    return ps - benchenmark

def plot_power_spectra(ell, ps_list, labels, ell_theory, theory_ps, title = 'Power spectrum'):
    for i in range(len(ps_list)):
        plt.scatter(ell, ps_list[i], s = 1, label = labels[i])
    plt.scatter(ell_theory[:1535], theory_ps[:1535], s=1, label='Theory power spectrum', color='black')

    plt.xlabel(r'$\ell$')
    plt.ylabel(r'$\ell(\ell+1)C_\ell/(2\pi)$ $\;$ [$\mu K^2$]')
    plt.title(title)
    plt.ylim(1e-6, 6e3)

    large_s = 20  # size of the scatter points in the legend
    # Custom legend with increased scatter point size
    handles, labels = plt.gca().get_legend_handles_labels()
    new_handles = [plt.Line2D([], [], color=h.get_facecolor()[0], marker='o', linestyle='', markersize=np.sqrt(large_s)) for h in handles]
    plt.legend(handles=new_handles, labels=labels, loc='upper left', bbox_to_anchor=(1, 1), fontsize='large')
    plt.show()


def plot_diff(ell, ps_list, benchenmark, labels, title = 'Power spectrum'):
    for i in range(len(ps_list)):
        plt.scatter(ell, get_percent_diff(ps_list[i], benchenmark), s = 1, label = labels[i])
    plt.xlabel(r'$\ell$')
    plt.ylabel(r'$\Delta C_\ell/C_\ell$ $\;$ [%]')
    plt.title(title)
    plt.ylim(-50, 50)

    large_s = 20  # size of the scatter points in the legend
    # Custom legend with increased scatter point size
    handles, labels = plt.gca().get_legend_handles_labels()
    new_handles = [plt.Line2D([], [], color=h.get_facecolor()[0], marker='o', linestyle='', markersize=np.sqrt(large_s)) for h in handles]
    plt.legend(handles=new_handles, labels=labels, loc='upper left', bbox_to_anchor=(1, 1), fontsize='large')
    plt.show()

#Global parameters
nside = 512
ellmax = 1534

#Load prediction map
prediction_map_path = '/home/yunan/pyilc/output/IQU_512_1450_test/512_mask/0/CN_needletILCmap_component_CMB.fits'
prediction_map = hp.read_map(prediction_map_path)

#Load theory power spectra
theory_ps_path = '/shared/data/Datasets/IQU_512_1450/Analysis_Theory_Power_Spectra/Test/sim0000/theory_ps.npy'
ell_theory, theory_ps = get_theory_ps(theory_ps_path)

#Read mask
mask_path = '/home/yunan/planck_release/planck_mask_512.fits'
mask = hp.read_map(mask_path, field = 0)

#Common beam
beam_fwhm = 5
beam = hp.gauss_beam(beam_fwhm*np.pi/(180*60), ellmax)

#create ell array
ell = np.arange(ellmax+1)


#Deconvolve the map and calculate power spectrum
prediction_map_deconv = map_deconv(prediction_map,nside,ellmax,beam)
prediction_map_deconv_rming_dipole = hp.remove_dipole(prediction_map_deconv) #This will remove the large cold patch on the ma, The ps is not afftected too much by this process
ps_prediction_map_deconv = get_power(prediction_map_deconv_rming_dipole,
                                     prediction_map_deconv_rming_dipole, 
                                     mask = mask, 
                                     lmax = ellmax)
dl_prediction_map_deconv = ps2dl(ps_prediction_map_deconv, ell)

#Deconvolve the prediction power spectrum
ps_pred = pse(prediction_map_path, ellmax, mask = mask)
dl_pred = ps2dl(ps_pred, ell)
dl_prediction_deconv = dl_deconv(dl_pred, beam)

#Plot
plot_power_spectra(ell, 
                   [dl_prediction_map_deconv, dl_prediction_deconv], 
                   ['Prediction map deconvolved', 'Prediction power spectrum deconvolved'], 
                   ell_theory, theory_ps, 
                   title = 'Prediction map deconvolved vs prediction power spectrum deconvolved')
plot_diff(ell,
            [dl_prediction_map_deconv, dl_prediction_deconv],
            theory_ps,
            ['Prediction map deconvolved', 'Prediction power spectrum deconvolved'],
            title = 'Prediction map deconvolved vs prediction power spectrum deconvolved')


# #Convolve the map and calculate power spectrum
# pred_map_conv = map_conv(prediction_map_deconv, 512, 1534, beam)
# ps_pred_map_conv = get_power(pred_map_conv, pred_map_conv, mask, lmax=1534)
# dl_pred_map_conv = ps2dl(ps_pred_map_conv, ell)
# #Calculate the power spectrum of the convolved map from Pyilc
# ps_pred = get_power(prediction_map, prediction_map, mask, lmax=1534)
# dl_pred = ps2dl(ps_pred, ell)

##Plot
# plot_power_spectra(ell, 
#                    [dl_pred, dl_pred_map_conv], 
#                    ['Prediction map', 'Prediction power spectrum convolved'], 
#                    ell_theory, theory_ps, 
#                    title = 'Prediction map deconvolved vs prediction power spectrum deconvolved')
# plot_diff(ell,
#             [dl_pred, dl_pred_map_conv],
#             theory_ps,
#             ['Prediction map', 'Prediction power spectrum convolved'],
#             title = 'Prediction map deconvolved vs prediction power spectrum deconvolved')