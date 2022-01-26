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
resultsDir = 'mars_SNR50_RandWalkIsofitCovEps0_11_1M'
setupDir = 'ang20140612'#'ang20170228'#
method = 'RandWalk'

Nsamp = 1000000
burn = 100000
thinning = 10
##### CONFIG #####

f = FileProcessing(setupDir='setup/' + setupDir)
# f.loadWavelength('data/wavelengths.txt')
# f.loadReflectance('data/beckmanlawn/insitu.txt')
# f.loadRadiance('data/beckmanlawn/ang20171108t184227_data_v2p11_BeckmanLawn.mat')
# f.loadConfig('config/config_inversion.json')
f.loadWavelength('data/wavelengths.txt')
f.loadReflectance('data/mars/insitu.txt')
f.loadRadiance('data/mars/ang20140612t215931_data_dump.mat')
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
m.runAM(method)
np.savetxt(setup.resultsDir + 'runtime.txt', np.array([time.time() - start_time]))










