import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

def cl2dl(cl):
    ell = np.arange(len(cl))
    dl = ell*(ell+1)*cl/(2*np.pi)
    return dl

def dl2cl(dl):
    cl = (0,0)
    ell = np.arange(2,len(dl))
    cl = dl[2:]*2*np.pi/ell/(ell+1)
    cl = np.insert(cl, 0, [0,0])
    return cl

def get_alms(map_in):
    nside_in = hp.get_nside(map_in)
    lmax_in = 3*nside_in - 1
    alm = hp.map2alm(map_in, lmax = lmax_in)
    return alm

def alm_udgrade(map_in, nside_out):
    alm_in = get_alms(map_in)
    lmax_out = 3*nside_out - 1
    alm_out = hp.almxfl(alm_in, np.ones(lmax_out + 1))
    map_out = hp.alm2map(alm_out, nside_out)
    return map_out

def theory_ps2map_realisation(theory_ps, nside):
    theory_cl = dl2cl(theory_ps)
    map_realisation = hp.synfast(theory_cl, nside)
    return map_realisation

def get_percent_diff(ps, benchenmark):
    temp_idx = len(ps)
    benchenmark = benchenmark[2:temp_idx]
    return (benchenmark - ps[2:])/benchenmark*100

def get_abs_diff(ps, benchenmark):
    temp_idx = len(ps)
    benchenmark = benchenmark[2:temp_idx]
    return benchenmark - ps[2:]

#############demo
nside_in = 2048
nside_out = 512
ellmax = 3*nside_out - 1
theory_ps = np.load("/shared/data/Datasets/IQU_512_1450/Analysis_Theory_Power_Spectra/Test/sim0000/theory_ps.npy")
ell_theory = np.arange(len(theory_ps))
ell = np.arange(ellmax + 1)

#Creating a CMB realisation map at 2048
map_realisation_at_2048 = theory_ps2map_realisation(theory_ps,2048)

#Downgrading the map to 512
map_realisation_at_512 = alm_udgrade(map_realisation_at_2048, 512)

#Calculating the power spectrum of the realisation maps and downgraded map
realisation_ps = hp.anafast(map_realisation_at_2048, lmax=ellmax)
realisation_ps_ud = hp.anafast(map_realisation_at_512, lmax=ellmax)

#normalising the power spectrum
realisation_dl = cl2dl(realisation_ps)
realisation_dl_ud = cl2dl(realisation_ps_ud)

#Calculating the percentage difference between the power spectrum of the realisation maps and the theory power spectrum
theory_realisation_percent_diff = get_percent_diff(realisation_dl,theory_ps)
theory_realisation_percent_diff_ud = get_percent_diff(realisation_dl_ud,theory_ps)
realisation_ud_percent_diff = get_percent_diff(realisation_dl_ud,realisation_dl)


#Plot maps(I was trying to use the function in cmb-ml to plot the maps but it was not working, so I had to use the healpy function to plot the maps)
hp.mollview(map_realisation_at_2048, title='Realisation map at 2048', unit='K', min=-500, max=500)
hp.mollview(map_realisation_at_512, title='Realisation map at 512', unit='K', min=-500, max=500)


fig, ax = plt.subplots(1,3, figsize=(18,5))

#Power spectrum
ax[0].scatter(ell,realisation_dl, s = 1, label = 'Realisation map power spectrum at 2048')
ax[0].scatter(ell,realisation_dl_ud, s = 1, label = 'Realisation map power spectrum at 512')
ax[0].scatter(ell_theory[:ellmax], theory_ps[:ellmax], s=1, label='Theory power spectrum', color='black')
ax[0].set_title('Power spectrum')
ax[0].set_xlabel(r'$\ell$')
ax[0].set_ylabel(r'$D_\ell$ $\;$ [$\mu K^2$]')
ax[0].set_xlim(0, 1534)
ax[0].set_ylim(1e-6, 6e3)
ax[0].legend()


#Percentage error with respect to theory
ax[1].scatter(ell[2:], theory_realisation_percent_diff, s = 1, label = 'Realisation map power spectrum at 2048')
ax[1].scatter(ell[2:], theory_realisation_percent_diff_ud, s = 1, label = 'Realisation map power spectrum at 512')
ax[1].set_title('Percentage error with respect to theory')
ax[1].set_xlabel(r'$\ell$')
ax[1].set_ylabel(r'$\Delta D_\ell/D_\ell$ $\;$ [%]')
ax[1].set_xlim(0, 1534)
ax[1].set_ylim(-50,50)
ax[1].legend()

#Percentage error with respect to realisation map at 2048
ax[2].scatter(ell[2:], realisation_ud_percent_diff, s = 1, label = 'Realisation map power spectrum at 512')
ax[2].set_title('Percentage error with respect to realisation map at 2048')
ax[2].set_xlabel(r'$\ell$')
ax[2].set_ylabel(r'$\Delta D_\ell/D_\ell$ $\;$ [%]')
ax[2].set_yscale('log')
ax[2].set_xlim(0, 1534)
ax[2].set_ylim( 1e-4,1e1)
ax[2].legend()

plt.tight_layout()
plt.show()