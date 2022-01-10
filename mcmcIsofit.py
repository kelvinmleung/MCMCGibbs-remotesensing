import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal, gaussian_kde
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

from mcmcGibbs import MCMCGibbs

class MCMCIsofit:
    '''
    Wrapper class to perform MCMC on Isofit
    Contains functions to perform MCMC sampling
    '''

    def __init__(self, setup, Nsamp, burn, x0, alg='AM', thinning=1):

        self.resultsDir = setup.resultsDir
    
        # initialize problem parameters
        self.wavelengths = setup.wavelengths
        self.reflectance = setup.reflectance # true reflectance
        self.truth = setup.truth # true state (ref + atm)
        self.yobs = setup.radiance # true radiance
        self.bands = setup.bands # reflectance indices excluding deep water spectra
        self.bandsX = setup.bandsX # same indices including atm parameters

        # isofit parameters and functions
        self.mu_x = setup.mu_x
        self.gamma_x = setup.gamma_x
        self.mupos_isofit = setup.isofitMuPos
        self.gammapos_isofit = setup.isofitGammaPos
        self.noisecov = setup.noisecov
        self.fm = setup.fm
        self.geom = setup.geom

        self.luts = setup.luts
        
        # MCMC parameters to initialize
        self.Nsamp = Nsamp
        self.x0 = x0
        self.alg = alg
        self.thinning = thinning
        self.burn = int(burn / self.thinning)
        
        self.nx = self.gamma_x.shape[0] # parameter dimension
        self.ny = self.noisecov.shape[0] # data dimension

    def initMCMC(self, rank=2, constrain=False):
        
        # create folder
        if not os.path.exists(self.resultsDir):
            os.makedirs(self.resultsDir)

        # define upper and lower bounds 
        if constrain == True:
            lowbound = np.concatenate((np.zeros(self.nx-2), [0, 1]))
            upbound = np.concatenate((np.ones(self.nx-2), [0.5, 2]))
        else:
            lowbound = np.ones(self.nx) * np.NINF
            upbound = np.ones(self.nx) * np.inf

        self.mcmcConfig = {
            "startX": self.x0,
            "Nsamp": self.Nsamp,
            "burn": self.burn,
            "sd": 2.38 ** 2 / rank,
            "propcov": np.array([[1,0],[0,1]]) * 0.001,#self.gamma_x * (2.38 ** 2) / rank,# self.gammapos_isofit * (2.38 ** 2) / rank,
            "lowbound": lowbound,
            "upbound": upbound,
            "rank": rank,
            "mu_x": self.mu_x,
            "gamma_x": self.gamma_x,
            "noisecov": self.noisecov,
            "yobs": self.yobs,
            "fm": self.fm,
            "geom": self.geom,
            "resultsDir": self.resultsDir,
            "thinning": self.thinning,
            "luts": self.luts,
            "truth": self.truth,
            "bands": self.bands,
            "bandsX": self.bandsX
            }
        self.mcmc = MCMCGibbs(self.mcmcConfig)
        self.saveMCMCConfig()

    def runAM(self):
        self.mcmc.adaptm(self.alg)   

    def saveMCMCConfig(self):
        # np.save(self.resultsDir + 'wavelength.npy', self.wavelengths)
        np.save(self.resultsDir + 'radiance.npy', self.yobs)
        np.save(self.resultsDir + 'truth.npy', self.truth)
        np.save(self.resultsDir + 'bands.npy', self.bands)
        np.save(self.resultsDir + 'mu_x.npy', self.mu_x)
        np.save(self.resultsDir + 'gamma_x.npy', self.gamma_x)
        np.save(self.resultsDir + 'isofitMuPos.npy', self.mupos_isofit)
        np.save(self.resultsDir + 'isofitGammaPos.npy', self.gammapos_isofit)
        np.save(self.resultsDir + 'posPredictive.npy', self.fm.calc_rdn(self.mupos_isofit, self.geom))
        np.save(self.resultsDir + 'Nsamp.npy', self.Nsamp)
        np.save(self.resultsDir + 'burn.npy', self.burn)
        np.save(self.resultsDir + 'thinning.npy', self.thinning)

    def calcMeanCov(self):
        self.MCMCmean, self.MCMCcov = self.mcmc.calcMeanCov()
        return self.MCMCmean, self.MCMCcov 

    def autocorr(self, ind):
        return self.mcmc.autocorr(ind)