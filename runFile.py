import subprocess
import time
import numpy as np
import matplotlib.pyplot as plt

from fileProcessing import FileProcessing
from isofitSetup import Setup
from genSamples import GenerateSamples
from regression import Regression
from analysis import Analysis
from mcmcIsofit import MCMCIsofit


##### CONFIG #####
Nsamp = 6000
burn = 2000
init = 'MAP'
rank = 100
LIS = True
mcmcfolder = 'H11S_test'
thinning = 1#20
setupDir = 'ang20140612'#'ang20170228'
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

radiance=0
setup = Setup(wv, ref, radiance, config, mcmcdir=mcmcfolder, setupDir=setupDir)
g = GenerateSamples(setup)


# surf_mu, surf_gamma = f.loadSurfModel('data/surface.mat')
# atm_mu = setup.mu_x[432:]
# atm_gamma = setup.gamma_x[432:, 432:]
# atm_bounds = np.array([[0.001, 0.5],[1.3100563704967498, 1.586606174707413]])

# g.genTrainTest(surf_mu, surf_gamma, atm_mu, atm_gamma, atm_bounds, f, 'train', NperPrior=6000)
# g.genTrainTest(surf_mu, surf_gamma, atm_mu, atm_gamma, atm_bounds, f, 'test', NperPrior=2000)

# X_train = np.load('../results/Regression/samples/ang20140612/X_train.npy')
# X_test = np.load('../results/Regression/samples/ang20140612/X_test.npy')
# g.genY(X_train, X_test)



r = Regression(setup)

# for i in [10,20,100,120,250,260,400]:
#     print('yElem = ', i)
#     r.tuneLasso(params=[1e-4,1e-3,1e-2,1e-1], yElem=i, plot=False)
# r.fullLasso(params=np.ones(432)*1e-3)
# r.plotFullLasso()
# plt.show()

# Y_train = np.diag(np.sqrt(r.varY)) @ r.Y_train.T + np.outer(r.meanY, np.ones(25000))
# Y_train = Y_train.T
# plt.figure()
# for i in range(200):   
#     plt.plot(setup.wavelengths, Y_train[np.random.randint(0,24000),:432], 'b', linewidth=0.5)
# plt.plot(setup.wavelengths, setup.radiance, 'r', linewidth=2, label='Observed radiance')
# plt.title('Training data vs Real radiance data')
# plt.ylabel('Radiance')
# plt.legend()
# plt.show()

a = Analysis(setup, r)

# linPosMu, linPosGamma = a.posterior(setup.radiance)
# plt.figure()
# plt.plot(setup.wavelengths[setup.bands], setup.truth[setup.bands], 'k.', label='Truth')
# plt.plot(setup.wavelengths[setup.bands], setup.isofitMuPos[setup.bands], 'r.', label='Isofit')
# plt.plot(setup.wavelengths[setup.bands], linPosMu[setup.bands], 'b', label='Linear Posterior')
# plt.title('Reflectance')
# plt.legend()
# plt.show()

# plt.figure()
# plt.plot(setup.wavelengths[setup.bands], setup.truth[setup.bands], 'k.', label='Truth')
# plt.plot(setup.wavelengths[setup.bands], setup.isofitMuPos[setup.bands], 'r.', label='Isofit')
# plt.show()



## MCMC #
if init == 'MAP':
    x0 = setup.isofitMuPos
elif init == 'truth':
    x0 = setup.truth
elif init == 'midMAPtruth':
    x0 = 0.5 * (setup.isofitMuPos + setup.truth)
elif init == 'linpos':
    linMuPos, linGammaPos = a.posterior(setup.radiance)
    x0 = linMuPos

mcmcfolder = mcmcfolder + '_init' + init + '_rank' + str(rank)

m = MCMCIsofit(setup, a, Nsamp, burn, x0, 'AM', thinning=thinning)
m.initMCMC(LIS=LIS, rank=rank, constrain=True) # specify LIS parameters

start_time = time.time()
m.runAM()
np.savetxt(setup.mcmcDir + 'runtime.txt', np.array([time.time() - start_time]))








