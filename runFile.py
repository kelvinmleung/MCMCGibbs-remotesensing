import time
import numpy as np
import matplotlib.pyplot as plt

from fileProcessing import FileProcessing
from isofitSetup import Setup
from mcmcIsofit import MCMCIsofit

np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
# lines 583-584 of modtran.py (end of build_lut)
# np.save('../MCMCGibbs-remotesensing/coszen.npy', self.coszen)                                        
# np.save('../MCMCGibbs-remotesensing/lut_grid.npy', self.luts)


##### CONFIG #####
resultsDir = '177_SNR50_DetRefl'
setupDir = 'ang20140612'#'ang20170228'#

Nsamp = 6000
burn = 1000
thinning = 20
##### CONFIG #####

f = FileProcessing(setupDir='setup/' + setupDir)
# f.loadWavelength('data/wavelengths.txt')
# f.loadReflectance('data/beckmanlawn/insitu.txt')
# f.loadRadiance('data/beckmanlawn/ang20171108t184227_data_v2p11_BeckmanLawn.mat')
# f.loadConfig('config/config_inversion.json')
f.loadWavelength('data/wavelengths.txt')
f.loadReflectance('data/177/insitu.txt')
f.loadRadiance('data/177/ang20140612t215931_data_dump.mat')
f.loadConfig('config/config_inversion.json')
wv, ref, radiance, config = f.getFiles()

# radiance=0
setup = Setup(wv, ref, radiance, config, resultsDir=resultsDir, setupDir=setupDir)


## MCMC ##
x0 = setup.isofitMuPos
# x0 = setup.mu_x
# x0 = setup.truth

m = MCMCIsofit(setup, Nsamp, burn, x0, 'AM', thinning=thinning)
m.initMCMC(constrain=True, rank=2) 

start_time = time.time()
m.runAM()
np.savetxt(setup.resultsDir + 'runtime.txt', np.array([time.time() - start_time]))


# test 
# N = 1000
# atm = [0.1, 1.5]
# samp = np.zeros([N, setup.ny])
# err = np.zeros(setup.ny)
# for i in range(N):
#     z = np.random.normal(size=setup.ny)
#     gamma_x_chol = np.linalg.cholesky(setup.gamma_x[:-2,:-2])
#     samp[i,:] = gamma_x_chol @ z + setup.mu_x[:-2]
#     sampFull = np.concatenate((samp[i,:], atm))
#     y_lin = setup.applyLinOper(samp[i,:], atm)
#     y_fwd = setup.fm.calc_rdn(sampFull, setup.geom)
#     err += abs(y_lin - y_fwd) / y_fwd
# err = err/N
# plt.plot(setup.wavelengths, err, label='Error')
# plt.title('Error in Linearized Forward Model')

# plt.plot(setup.wavelengths, y_fwd, label='Forward Model')
# plt.plot(setup.wavelengths, y_lin, label='From LUT')
# plt.legend()
# # plt.show()

# radiance=0
# setup = Setup(wv, ref, radiance, config, resultsDir=resultsDir, setupDir=setupDir)
# xAtm = setup.truth[-2:]
# mu_x = setup.mu_x[:-2]
# gamma_x = setup.gamma_x[:-2,:-2]
# inv_gamma_x = np.linalg.inv(gamma_x)
# inv_noisecov = np.linalg.inv(setup.noisecov)
# rhoatm, sphalb, transm, coszen, solar_irr = setup.unpackLUTparam(xAtm)

# xInput = np.concatenate((setup.isofitMuPos[:-2], xAtm))# np.concatenate((setup.mu_x[:-2], xAtm))#
# G = setup.linOper(sphalb, transm, coszen, solar_irr)
# K = setup.fm.K(xInput, setup.geom)[:setup.ny, :setup.ny]
# yobs_adjust = setup.radiance - coszen / np.pi * solar_irr * rhoatm 
# gamma_refl = np.linalg.inv(G.T @ inv_noisecov @ G + inv_gamma_x)
# gamma_refl_K = np.linalg.inv(K.T @ inv_noisecov @ K + inv_gamma_x)
# mu_refl = gamma_refl @ (G.T @ inv_noisecov @ yobs_adjust + inv_gamma_x @ mu_x)
# mu_refl_K = gamma_refl_K @ (K.T @ inv_noisecov @ yobs_adjust + inv_gamma_x @ mu_x)
# # diffGK = (np.diag(G)-np.diag(K))/np.diag(K)
# # print(diffGK[setup.bands])

# plt.figure(3)
# # plt.plot(mu_refl, label='Linear Inversion')
# plt.plot(setup.wavelengths[setup.bands], mu_refl[setup.bands], '.', label='Linear G')
# plt.plot(setup.wavelengths[setup.bands], mu_refl_K[setup.bands], '.', label='Jacobian K')
# plt.plot(setup.wavelengths[setup.bands], setup.isofitMuPos[setup.bands], '.', label='Isofit')
# plt.title('Posterior Mean')
# plt.legend()


# plt.figure(4)
# plt.plot(setup.wavelengths[setup.bands], np.sqrt(np.diag(gamma_refl)[setup.bands]), label='Linear G')
# plt.plot(setup.wavelengths[setup.bands], np.sqrt(np.diag(gamma_refl_K)[setup.bands]), label='Jacobian K')
# plt.plot(setup.wavelengths[setup.bands], np.sqrt(np.diag(setup.isofitGammaPos)[setup.bands]), label='Isofit')
# plt.title('Marginal Variance')
# plt.legend()
# plt.show()









