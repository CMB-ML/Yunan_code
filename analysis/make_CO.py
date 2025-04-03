import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from astropy.io import fits

class downgraded_co:
    def __init__(self, nside, co_map):
        self.nside = nside
        self.co_map = co_map

    def alm_downgrade(self):
        nside_in = hp.get_nside(self.co_map)
        alm = hp.map2alm(self.co_map, lmax=3*nside_in-1)
        alm = hp.almxfl(alm, np.ones(3*self.nside-1))
        map_downgraded = hp.alm2map(alm, self.nside)
        return map_downgraded
    
    def downgrade_co(self):
        map_downgraded = self.alm_downgrade()
        return map_downgraded


class sampling_co:
    def __init__(self, nside, co_map):
        self.nside = nside
        self.co_map = co_map

    def gaussian_sampling(self):
        nside_in = hp.get_nside(self.co_map)
        raw_map = np.random.normal(0, 1, hp.nside2npix(nside_in))
        map_sampled = raw_map * self.co_map
        return map_sampled
    
    def sample_co(self):
        map_sampled = self.gaussian_sampling()
        return map_sampled
    
    def alm_downgrade(self, map_in):
        nside_in = hp.get_nside(map_in)
        alm = hp.map2alm(map_in, lmax=3*nside_in-1)
        alm = hp.almxfl(alm, np.ones(3*self.nside-1))
        map_downgraded = hp.alm2map(alm, self.nside)
        return map_downgraded
    
    def downgrade_co(self):
        map_sampled = self.sample_co()
        map_downgraded = self.alm_downgrade(map_sampled)
        return map_downgraded