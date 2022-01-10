import sys, os
import numpy as np
import scipy as s
from scipy.io import loadmat
import matplotlib.pyplot as plt

import scipy.stats as st

sys.path.insert(0, '../isofit/')

from isofit.core.forward import ForwardModel
from isofit.core.geometry import Geometry
from isofit.inversion.inverse import Inversion
from isofit.configs.configs import Config  
from isofit.surface.surface_multicomp import MultiComponentSurface

class Setup:
    '''
    Contains functions to generate training and test samples
    from isofit.
    '''
    def __init__(self, wv, ref, radiance, config, resultsDir, setupDir):

        print('Setup in progress...')
        self.wavelengths = wv
        self.reflectance = ref

        # specify storage directories 
        self.setupDir = setupDir
        self.resultsDir = '../resultsGibbs/' + resultsDir + '/'
        
        # initialize Isofit with config 
        self.config = config
        self.windows = config['implementation']['inversion']['windows']
        self.surfaceFile = config['forward_model']['surface']['surface_file']
        fullconfig = Config(config)
        self.fm = ForwardModel(fullconfig)
        self.geom = Geometry()
        self.mu_x, self.gamma_x = self.getPrior(fullconfig)

        self.luts = np.load('lut_grid.npy').item()
        
        # get Isofit noise model and simulate radiance
        atmSim = [0.1, 1.5] 
        self.truth = np.concatenate((ref, atmSim))
        # self.truth = np.random.multivariate_normal(self.mu_x, self.gamma_x) ############
        # self.truth[-2:] = atmSim
        # plt.plot(self.truth[:-2])
        # plt.show()

        rad = self.fm.calc_rdn(self.truth, self.geom)
        self.noisecov = self.fm.Seps(self.truth, rad, self.geom)
        eps = np.random.multivariate_normal(np.zeros(len(rad)), self.noisecov)
        self.radianceSim = rad + eps

        if np.all((radiance == 0)): #radiance == np.zeros(radiance.shape):#.all() == 0:
            self.radiance =  self.radianceSim
        else:
            self.radiance = radiance
        
        # inversion using simulated radiance
        self.isofitMuPos, self.isofitGammaPos = self.invModel(self.radiance)
        self.nx = self.truth.shape[0]
        self.ny = self.radiance.shape[0]
        
        # get indices that are in the window (i.e. take out deep water spectra)
        wl = self.wavelengths
        w = self.windows
        bands = []
        for i in range(wl.size):
            # if (wl[i] > 380 and wl[i] < 1300) or (wl[i] > 1450 and wl[i] < 1780) or (wl[i] > 1950 and wl[i] < 2450):
            if (wl[i] > w[0][0] and wl[i] < w[0][1]) or (wl[i] > w[1][0] and wl[i] < w[1][1]) or (wl[i] > w[2][0] and wl[i] < w[2][1]):
                bands = bands + [i]
        self.bands = bands
        self.bandsX = bands + [self.nx-2,self.nx-1]
        

    # def saveConfig(self):
    #     np.save(self.resultsDir + 'wavelength.npy', self.wavelengths)
    #     np.save(self.resultsDir + 'radiance.npy', self.radiance)
    #     np.save(self.resultsDir + 'truth.npy', self.truth)
    #     np.save(self.resultsDir + 'bands.npy', self.bands)
    #     np.save(self.resultsDir + 'mu_x.npy', self.mu_x)
    #     np.save(self.resultsDir + 'gamma_x.npy', self.gamma_x)
    #     np.save(self.resultsDir + 'isofitMuPos.npy', self.isofitMuPos)
    #     np.save(self.resultsDir + 'isofitGammaPos.npy', self.isofitGammaPos)
    #     np.save(self.resultsDir + 'posPredictive.npy', self.fm.calc_rdn(self.isofitMuPos, self.geom))

    def getPrior(self, fullconfig):
        # get index of prior used in inversion
        mcs = MultiComponentSurface(fullconfig)
        # indPr = mcs.component(self.truth, self.geom)
        indPr = mcs.component(self.reflectance, self.geom)
        print('Prior Index:', indPr)
        # Get prior mean and covariance
        surfmat = loadmat(self.surfaceFile)
        wl = surfmat['wl'][0]
        refwl = np.squeeze(surfmat['refwl'])
        idx_ref = [np.argmin(abs(wl-w)) for w in np.squeeze(refwl)]
        idx_ref = np.array(idx_ref)
        refnorm = np.linalg.norm(self.reflectance[idx_ref])

        mu_priorsurf = self.fm.surface.components[indPr][0] * refnorm
        mu_priorRT = self.fm.RT.xa()
        mu_priorinst = self.fm.instrument.xa()
        mu_x = np.concatenate((mu_priorsurf, mu_priorRT, mu_priorinst), axis=0)
        
        gamma_priorsurf = self.fm.surface.components[indPr][1] * (refnorm ** 2)
        gamma_priorRT = self.fm.RT.Sa()[:, :]
        gamma_priorinst = self.fm.instrument.Sa()[:, :]
        gamma_x = s.linalg.block_diag(gamma_priorsurf, gamma_priorRT, gamma_priorinst)
        
        return mu_x, gamma_x

    def invModel(self, radiance):
        inversion_settings = self.config
        inverse_config = Config(inversion_settings)
        iv = Inversion(inverse_config, self.fm)
        state_trajectory = iv.invert(radiance, self.geom)
        state_est = state_trajectory[-1]
        rfl_est, rdn_est, path_est, S_hat, K, G = iv.forward_uncertainty(state_est, radiance, self.geom)

        return state_est, S_hat
    
    def lookupLUT(self, point):
        ret = {}
        for key, lut in self.luts.items():
            ret[key] = np.array(lut(point)).ravel()
        return ret

    def unpackLUTparam(self, atm):
        lutparam = self.lookupLUT(atm)
        rhoatm = lutparam['rhoatm']
        sphalb = lutparam['sphalb']
        transm = lutparam['transm']
        coszen = np.load('coszen.npy')   
        solar_irr = np.load('solar_irr.npy')   
        return rhoatm, sphalb, transm, coszen, solar_irr

    def linOper(self, sphalb, transm, coszen, solar_irr): # Conditioned on the atmospheric parameters!
        # rhoatm, sphalb, transm, coszen, solar_irr = self.unpackLUTparam(atm)
        # G = np.zeros([self.ny, self.nx])
        xMAP = self.isofitMuPos[:self.ny]
        # xMAP = self.mu_x[:self.ny]
        # G[:self.ny, :self.ny]
        G = coszen / np.pi * np.diag(solar_irr * transm / (1 - sphalb * xMAP))
        return G
    
    def applyLinOper(self, ref, atm): # Conditioned on the atmospheric parameters!
        # atm = x[-2:]
        rhoatm, sphalb, transm, coszen, solar_irr = self.unpackLUTparam(atm)
        G = self.linOper(sphalb, transm, coszen, solar_irr)
        # x = np.concatenate((ref, atm))
        x = ref
        y_lut = G @ x + coszen / np.pi * solar_irr * rhoatm 
        return y_lut
    