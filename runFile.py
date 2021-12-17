import subprocess
import time
import numpy as np
import matplotlib.pyplot as plt

from fileProcessing import FileProcessing
from isofitSetup import Setup
# from genSamples import GenerateSamples
# from regression import Regression
# from analysis import Analysis
# from mcmcIsofit import MCMCIsofit


##### CONFIG #####
resultsDir = 'Test1'
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

# radiance=0
setup = Setup(wv, ref, radiance, config, resultsDir=resultsDir, setupDir=setupDir)

plt.plot(setup.wavelengths, setup.isofitMuPos[:432])
plt.plot(setup.wavelengths, setup.reflectance)
plt.show()
# ## MCMC #
# if init == 'MAP':
#     x0 = setup.isofitMuPos
# elif init == 'truth':
#     x0 = setup.truth
# elif init == 'midMAPtruth':
#     x0 = 0.5 * (setup.isofitMuPos + setup.truth)
# elif init == 'linpos':
#     linMuPos, linGammaPos = a.posterior(setup.radiance)
#     x0 = linMuPos

# mcmcfolder = mcmcfolder + '_init' + init + '_rank' + str(rank)

# m = MCMCIsofit(setup, a, Nsamp, burn, x0, 'AM', thinning=thinning)
# m.initMCMC(LIS=LIS, rank=rank, constrain=True) # specify LIS parameters

# start_time = time.time()
# m.runAM()
# np.savetxt(setup.mcmcDir + 'runtime.txt', np.array([time.time() - start_time]))








