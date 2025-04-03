import numpy as np
import matplotlib.pyplot as plt
import healpy as hp

# PS Nomarlization
def cl2dl(cl, lmin, lmax):
    l = np.arange(lmin, lmax+1)
    if len(cl) < len(l):
        cl = cl[lmin:]
    dl = cl * l * (l + 1) / 2 / np.pi
    return dl
def dl2cl(dl, lmin, lmax):
    l = np.arange(lmin, lmax+1)
    if len(dl) < len(l):
        dl = dl[lmin:]
    cl = dl * 2 * np.pi / l / (l + 1)
    return cl

# Resolution
def radian2arcmin(radian):
    return radian * 60 * 180 / np.pi
def arcmin2radian(arcmin): 
    return arcmin * np.pi / (60 * 180)


# Plotting
def plot_power_spectrum(ell, power_spectra, labels, 
                        colors=None, theory_spectrum=None, theory_label='Theory input',
                        xlabel=r'$\ell$', ylabel=r'$\ell (\ell + 1) C_\ell / 2\pi$', 
                        xscale='log', yscale='linear', ylim=(0, 6000), 
                        title="Power Spectrum", legend_loc='upper left', legend_bbox=(1, 1)):
    plt.figure(figsize=(8, 6))
    
    for idx, spectrum in enumerate(power_spectra):
        color = colors[idx] if colors is not None else None
        plt.plot(ell, spectrum, label=labels[idx], color=color)
    
    if theory_spectrum is not None:
        plt.plot(ell, theory_spectrum, c='black', label=theory_label)
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xscale(xscale)
    plt.yscale(yscale)
    plt.ylim(ylim)
    plt.title(title)
    plt.legend(loc=legend_loc, bbox_to_anchor=legend_bbox)
    plt.show()
